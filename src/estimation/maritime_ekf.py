"""
Maritime Extended Kalman Filter (EKF) with CTRV (Coordinated Turn Rate and Velocity) Model.
Specifically engineered for high-speed speedboats and maritime vessels subject to wave spray,
glare occlusions, wake clutter, and rapid maneuvering.
"""

import numpy as np


class MaritimeEKF:
    def __init__(self, dt: float = 0.05, process_noise_std=(1.5, 0.8, 0.1, 0.5, 0.5)):
        """
        State Vector: x = [x, y, v, theta, omega, w, h]^T
        - x, y: target center position (pixels or meters)
        - v: linear speed
        - theta: heading angle (radians)
        - omega: yaw rate / angular turn rate (rad/s)
        - w, h: target bounding box dimensions
        """
        self.dt = dt
        self.dim_x = 7
        self.dim_z = 4  # [x, y, w, h]

        # State vector
        self.x = np.zeros(self.dim_x)

        # State covariance matrix P
        self.P = np.diag([50.0, 50.0, 10.0, 1.0, 0.5, 20.0, 20.0])

        # Process noise covariance Q
        # process_noise_std: (pos_accel, speed_accel, yaw_accel, w_std, h_std)
        s_a, s_v, s_w, s_bw, s_bh = process_noise_std
        self.Q = np.diag([
            0.5 * s_a * dt**2,
            0.5 * s_a * dt**2,
            s_v * dt,
            0.5 * s_w * dt**2,
            s_w * dt,
            s_bw * dt,
            s_bh * dt
        ])

        # Measurement noise covariance R (calibrated for high-res maritime aerial video)
        self.R = np.diag([25.0, 25.0, 36.0, 36.0])

        # Measurement matrix H
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y
        self.H[2, 5] = 1.0  # w
        self.H[3, 6] = 1.0  # h

        # Previous measurement for velocity estimation
        self.prev_z = None
        self.is_initialized = False
        self.frames_coasting = 0
        self.max_coasting_frames = 45  # ~1.5 to 2 seconds of zero detection before declaring lost
        self.status = "UNINITIALIZED"   # LOCKED, COASTING, LOST
        self.nis = 0.0                  # Normalized innovation squared

    def initialize(self, bbox, initial_velocity=0.0, initial_heading=0.0):
        """
        Initialize EKF from bounding box [cx, cy, w, h].
        """
        cx, cy, w, h = bbox
        self.x = np.array([cx, cy, initial_velocity, initial_heading, 0.0, w, h], dtype=float)
        self.P = np.diag([10.0, 10.0, 15.0, 1.0, 0.5, 5.0, 5.0])
        self.prev_z = (cx, cy)
        self.is_initialized = True
        self.frames_coasting = 0
        self.status = "LOCKED"

    def predict(self, dt: float = None):
        """
        CTRV Motion Model propagation step.
        """
        if not self.is_initialized:
            return self.x

        if dt is None:
            dt = self.dt

        px, py, v, theta, omega, w, h = self.x

        # State transition function f(x)
        if abs(omega) > 1e-4:
            # Curved coordinated turn
            new_px = px + (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
            new_py = py + (v / omega) * (-np.cos(theta + omega * dt) + np.cos(theta))
        else:
            # Straight-line motion
            new_px = px + v * np.cos(theta) * dt
            new_py = py + v * np.sin(theta) * dt

        new_theta = (theta + omega * dt + np.pi) % (2 * np.pi) - np.pi
        new_v = v
        new_omega = omega
        new_w = w
        new_h = h

        self.x = np.array([new_px, new_py, new_v, new_theta, new_omega, new_w, new_h])

        # Jacobian F_j = df/dx
        F = np.eye(self.dim_x)
        if abs(omega) > 1e-4:
            s_diff = np.sin(theta + omega * dt) - np.sin(theta)
            c_diff = -np.cos(theta + omega * dt) + np.cos(theta)
            s_tp = np.sin(theta + omega * dt)
            c_tp = np.cos(theta + omega * dt)

            F[0, 2] = s_diff / omega
            F[0, 3] = (v / omega) * (c_tp - np.cos(theta))
            F[0, 4] = (v * dt * c_tp) / omega - (v * s_diff) / (omega**2)

            F[1, 2] = c_diff / omega
            F[1, 3] = (v / omega) * (s_tp - np.sin(theta))
            F[1, 4] = (v * dt * s_tp) / omega - (v * c_diff) / (omega**2)
        else:
            F[0, 2] = np.cos(theta) * dt
            F[0, 3] = -v * np.sin(theta) * dt
            F[1, 2] = np.sin(theta) * dt
            F[1, 3] = v * np.cos(theta) * dt

        F[3, 4] = dt

        # Propagate covariance P = F P F^T + Q
        self.P = F @ self.P @ F.T + self.Q
        return self.x

    def update(self, measurement=None, confidence: float = 1.0, gating_threshold: float = 45.0):
        """
        Measurement Update step.
        measurement: [cx, cy, w, h] or None (if occluded/missed detection)
        """
        if not self.is_initialized:
            if measurement is not None:
                self.initialize(measurement)
            return self.x

        if measurement is None or confidence < 0.25:
            # Coasting: no valid visual measurement available
            self.frames_coasting += 1
            if self.frames_coasting > self.max_coasting_frames:
                self.status = "LOST"
            else:
                self.status = "COASTING"
            return self.x

        z = np.array(measurement, dtype=float)
        # Expected measurement
        z_pred = self.H @ self.x

        # Innovation (residual)
        y = z - z_pred

        # Innovation covariance S = H P H^T + R
        # Adjust R inversely with optical detection confidence
        R_adaptive = self.R * (1.0 / max(confidence, 0.2))
        S = self.H @ self.P @ self.H.T + R_adaptive

        # Normalized Innovation Squared (NIS) for outlier/spray gating
        inv_S = np.linalg.inv(S)
        self.nis = float(y.T @ inv_S @ y)

        if self.nis > gating_threshold:
            # Outlier rejection (e.g. spurious wave foam or sun glint false positive)
            self.frames_coasting += 1
            if self.frames_coasting > self.max_coasting_frames:
                self.status = "LOST"
            else:
                self.status = "COASTING"
            return self.x

        # Kalman Gain K = P H^T S^-1
        K = self.P @ self.H.T @ inv_S

        # State update
        self.x = self.x + K @ y
        # Wrap theta to [-pi, pi]
        self.x[3] = (self.x[3] + np.pi) % (2 * np.pi) - np.pi

        # Update velocity, heading, and yaw rate kinematics from consecutive position measurements
        if self.prev_z is not None:
            dx = z[0] - self.prev_z[0]
            dy = z[1] - self.prev_z[1]
            dist = np.hypot(dx, dy)
            speed_obs = dist / max(self.dt, 1e-4)

            # If vessel is visibly moving
            if speed_obs > 0.5:
                heading_obs = np.arctan2(dy, dx)
                # Heading difference
                d_heading = (heading_obs - self.x[3] + np.pi) % (2 * np.pi) - np.pi
                omega_obs = d_heading / max(self.dt, 1e-4)

                # Alpha-filter smoothing into EKF state
                alpha = min(0.45, max(confidence, 0.2))
                self.x[2] = (1 - alpha) * self.x[2] + alpha * speed_obs
                self.x[3] = (self.x[3] + alpha * d_heading + np.pi) % (2 * np.pi) - np.pi
                self.x[4] = (1 - alpha) * self.x[4] + alpha * omega_obs

        self.prev_z = (z[0], z[1])

        # Update covariance using Joseph form for numerical stability:
        # P = (I - KH) P (I - KH)^T + K R K^T
        I = np.eye(self.dim_x)
        I_KH = I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_adaptive @ K.T

        # Update speed and heading if we have motion history
        self.frames_coasting = 0
        self.status = "LOCKED"
        return self.x

    def get_bbox(self):
        """Returns [cx, cy, w, h] and corner [x1, y1, x2, y2]"""
        cx, cy, v, theta, omega, w, h = self.x
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return (cx, cy, w, h), (x1, y1, x2, y2)

    def get_velocity_vector(self):
        """Returns (vx, vy) in pixels or coordinate units per second"""
        v = self.x[2]
        theta = self.x[3]
        return v * np.cos(theta), v * np.sin(theta)

    def get_predicted_trajectory(self, horizon_steps: int = 20, dt: float = None):
        """Predicts ahead future trajectory points for display / guidance preview"""
        if dt is None:
            dt = self.dt
        pts = []
        cur_x = self.x.copy()
        px, py, v, theta, omega, w, h = cur_x

        for _ in range(horizon_steps):
            if abs(omega) > 1e-4:
                px += (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
                py += (v / omega) * (-np.cos(theta + omega * dt) + np.cos(theta))
            else:
                px += v * np.cos(theta) * dt
                py += v * np.sin(theta) * dt
            theta = (theta + omega * dt + np.pi) % (2 * np.pi) - np.pi
            pts.append((px, py))
        return pts
