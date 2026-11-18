"""
Closed-Loop Maritime Drone Tracking and Accompany Simulation Engine.
Synchronizes visual detection, EKF state estimation, 3D target projection,
and drone flight dynamics.
"""

import cv2
import numpy as np
from src.estimation.maritime_ekf import MaritimeEKF
from src.gnc.drone_kinematics import DroneKinematics
from src.gnc.guidance_controller import GuidanceController
from src.vision.visual_tracker import VisualTracker
from src.benchmark.metrics import MaritimeBenchmark


class ClosedLoopSimulation:
    def __init__(
        self,
        video_path: str,
        groundtruth_path: str = None,
        standoff_dist: float = 25.0,
        standoff_alt: float = 15.0,
        dt: float = 0.05,
        meters_per_pixel: float = 0.15 # Scale factor mapping pixel displacement to metric sea coordinates
    ):
        self.video_path = video_path
        self.groundtruth_path = groundtruth_path
        self.dt = dt
        self.m_per_px = meters_per_pixel

        # Modules
        self.ekf = MaritimeEKF(dt=dt)
        self.drone = DroneKinematics(
            init_pos=(-15.0, -40.0, standoff_alt),
            init_yaw=np.radians(45),
            max_speed=30.0,
            time_constant=0.20
        )
        self.guidance = GuidanceController(
            standoff_distance=standoff_dist,
            standoff_altitude=standoff_alt
        )
        self.tracker = VisualTracker(tracker_type="CSRT")
        self.benchmark = MaritimeBenchmark()

        # Load ground truth if available
        self.groundtruth = []
        if groundtruth_path:
            self._load_groundtruth(groundtruth_path)

        # Telemetry log
        self.logs = []
        self.frame_idx = 0

    def _load_groundtruth(self, path: str):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [float(v) for v in line.replace(",", " ").split()]
                if len(parts) >= 4:
                    x, y, w, h = parts[:4]
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    self.groundtruth.append((cx, cy, w, h))

    def reset(self, initial_bbox, initial_frame: np.ndarray = None):
        """
        initial_bbox: (cx, cy, w, h)
        """
        self.ekf.initialize(initial_bbox)
        cx, cy, w, h = initial_bbox
        if initial_frame is not None:
            self.tracker.init(initial_frame, (cx - w / 2.0, cy - h / 2.0, w, h))

        target_x_m = cx * self.m_per_px
        target_y_m = cy * self.m_per_px
        self.drone.pos = np.array([
            target_x_m - self.guidance.standoff_distance,
            target_y_m - self.guidance.standoff_distance,
            self.guidance.standoff_altitude
        ])
        self.drone.vel = np.zeros(3)
        self.drone.history = [self.drone.pos.copy()]
        self.frame_idx = 0
        self.logs = []

    def process_frame(
        self,
        frame: np.ndarray,
        detector_measurement=None,
        detector_conf: float = 0.0,
        simulate_spray_dropout: bool = False
    ):
        """
        Runs one step of the closed-loop tracking & flight cycle.
        """
        self.frame_idx += 1
        gt_bbox = self.groundtruth[self.frame_idx - 1] if self.frame_idx <= len(self.groundtruth) else None

        # 1. EKF Motion Prediction
        self.ekf.predict(self.dt)

        # 2. Visual Tracking / Detection association
        measurement = None
        conf = 0.0

        if simulate_spray_dropout:
            # Drop optical tracker state during severe occlusion/spray
            self.tracker.is_tracking = False
        else:
            # Check optical tracker
            if self.tracker.is_tracking:
                ok, opt_bbox, opt_conf = self.tracker.update(frame)
                if ok:
                    measurement = opt_bbox
                    conf = opt_conf

            # If detector provided candidate or optical tracker failed
            if detector_measurement is not None:
                if measurement is None or detector_conf >= conf:
                    measurement = detector_measurement
                    conf = detector_conf
                    # Re-seed optical tracker to target
                    cx, cy, w, h = measurement
                    self.tracker.init(frame, (cx - w / 2.0, cy - h / 2.0, w, h))

        # 3. EKF Measurement Update (handles coasting if measurement is None or occluded)
        self.ekf.update(measurement, confidence=conf)
        est_bbox, (x1, y1, x2, y2) = self.ekf.get_bbox()
        vx_px, vy_px = self.ekf.get_velocity_vector()

        # 4. Project Estimated Target into 3D Metric Coordinates
        target_x_m = est_bbox[0] * self.m_per_px
        target_y_m = est_bbox[1] * self.m_per_px
        target_pos_3d = np.array([target_x_m, target_y_m, 0.0])

        target_vx_m = vx_px * self.m_per_px
        target_vy_m = vy_px * self.m_per_px
        target_vel_3d = np.array([target_vx_m, target_vy_m, 0.0])
        target_heading = self.ekf.x[3]

        # 5. GNC Guidance & Flight Setpoint
        v_cmd_3d, yaw_cmd, telem = self.guidance.compute_flight_command(
            drone_pos=self.drone.pos,
            drone_vel=self.drone.vel,
            target_pos_3d=target_pos_3d,
            target_vel_3d=target_vel_3d,
            target_heading=target_heading
        )

        # 6. Step Drone Flight Dynamics
        self.drone.step(v_cmd_3d, yaw_cmd, dt=self.dt)

        # 7. Benchmark Recording
        self.benchmark.record_frame(
            pred_bbox=est_bbox,
            gt_bbox=gt_bbox,
            status=self.ekf.status,
            standoff_err=telem["standoff_error"],
            speed_err=telem["relative_speed_error"]
        )

        # 8. Record telemetry log
        log_entry = {
            "frame": self.frame_idx,
            "status": self.ekf.status,
            "pred_bbox": est_bbox,
            "gt_bbox": gt_bbox,
            "target_pos_3d": target_pos_3d.tolist(),
            "target_vel_3d": target_vel_3d.tolist(),
            "target_speed_knots": float(np.linalg.norm(target_vel_3d[:2]) * 1.94384), # m/s to knots
            "drone_pos_3d": self.drone.pos.tolist(),
            "drone_vel_3d": self.drone.vel.tolist(),
            "drone_speed_knots": float(np.linalg.norm(self.drone.vel[:2]) * 1.94384),
            "standoff_error_m": telem["standoff_error"],
            "relative_speed_error_ms": telem["relative_speed_error"],
            "drone_yaw_deg": float(np.degrees(self.drone.attitude[2])),
            "target_yaw_deg": float(np.degrees(target_heading))
        }
        self.logs.append(log_entry)

        # 9. Render Tactical HUD
        annotated = self._render_hud(frame, est_bbox, gt_bbox, log_entry)

        return annotated, log_entry

    def _render_hud(self, frame: np.ndarray, est_bbox, gt_bbox, telem: dict) -> np.ndarray:
        hud = frame.copy()
        cx, cy, w, h = [int(v) for v in est_bbox]
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        # Status color
        if telem["status"] == "LOCKED":
            color = (0, 255, 0)      # Green
            status_text = "AUTO-LOCK: ACTIVE"
        elif telem["status"] == "COASTING":
            color = (0, 165, 255)    # Orange / Amber
            status_text = "COASTING (WAKE/SPRAY REJECTION)"
        else:
            color = (0, 0, 255)      # Red
            status_text = "TARGET LOST"

        # Draw Ground Truth in cyan dashed/thin if available
        if gt_bbox is not None:
            gcx, gcy, gw, gh = [int(v) for v in gt_bbox]
            cv2.rectangle(
                hud,
                (int(gcx - gw / 2), int(gcy - gh / 2)),
                (int(gcx + gw / 2), int(gcy + gh / 2)),
                (255, 255, 0),
                1
            )
            cv2.putText(hud, "GT", (int(gcx - gw / 2), int(gcy - gh / 2) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Draw Tactical Target Reticle (Corner Brackets)
        cv2.rectangle(hud, (x1, y1), (x2, y2), color, 2)
        reticle_len = max(6, int(min(w, h) * 0.25))
        # Top-left corner
        cv2.line(hud, (x1, y1), (x1 + reticle_len, y1), color, 3)
        cv2.line(hud, (x1, y1), (x1, y1 + reticle_len), color, 3)
        # Top-right corner
        cv2.line(hud, (x2, y1), (x2 - reticle_len, y1), color, 3)
        cv2.line(hud, (x2, y1), (x2, y1 + reticle_len), color, 3)
        # Bottom-left corner
        cv2.line(hud, (x1, y2), (x1 + reticle_len, y2), color, 3)
        cv2.line(hud, (x1, y2), (x1, y2 - reticle_len), color, 3)
        # Bottom-right corner
        cv2.line(hud, (x2, y2), (x2 - reticle_len, y2), color, 3)
        cv2.line(hud, (x2, y2), (x2, y2 - reticle_len), color, 3)
        # Center crosshair
        cv2.drawMarker(hud, (cx, cy), color, cv2.MARKER_CROSS, 14, 1)

        # Draw CTRV Predicted trajectory arc
        future_pts = self.ekf.get_predicted_trajectory(horizon_steps=15)
        for i in range(len(future_pts) - 1):
            p_a = (int(future_pts[i][0]), int(future_pts[i][1]))
            p_b = (int(future_pts[i + 1][0]), int(future_pts[i + 1][1]))
            cv2.line(hud, p_a, p_b, (255, 200, 0), 2)

        # Top Banner HUD Overlay
        overlay = hud.copy()
        cv2.rectangle(overlay, (15, 15), (420, 155), (20, 25, 30), -1)
        cv2.addWeighted(overlay, 0.75, hud, 0.25, 0, hud)
        cv2.rectangle(hud, (15, 15), (420, 155), color, 1)

        # Telemetry text
        cv2.putText(hud, status_text, (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(hud, f"VESSEL SPEED: {telem['target_speed_knots']:.1f} kts",
                    (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1)
        cv2.putText(hud, f"DRONE CHASE SPEED: {telem['drone_speed_knots']:.1f} kts",
                    (25, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1)
        cv2.putText(hud, f"STANDOFF DISTANCE: {self.guidance.standoff_distance:.1f}m (Err: {telem['standoff_error_m']:.2f}m)",
                    (25, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1)
        cv2.putText(hud, f"GNC STATUS: ACCOMPANY / SPEED MATCH",
                    (25, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)

        return hud
