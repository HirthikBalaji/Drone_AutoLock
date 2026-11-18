"""
Standoff Chase & Accompany Flight Guidance Controller.
Computes real-time 3D flight control setpoints (velocity and heading)
to keep the drone locked onto the speedboat at a constant standoff distance and altitude,
matching the vessel's speed and heading.
"""

import numpy as np


class GuidanceController:
    def __init__(
        self,
        standoff_distance: float = 25.0,  # Distance trailing behind the boat (meters)
        standoff_altitude: float = 15.0,  # Altitude above sea level (meters)
        lateral_offset: float = 0.0,      # Flanking offset (0 = directly behind, + = starboard, - = port)
        kp_pos: float = 1.2,              # Proportional position gain
        kd_pos: float = 0.4,              # Derivative damping gain
        kp_alt: float = 1.5,              # Altitude proportional gain
    ):
        self.standoff_distance = standoff_distance
        self.standoff_altitude = standoff_altitude
        self.lateral_offset = lateral_offset

        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_alt = kp_alt

    def compute_flight_command(
        self,
        drone_pos: np.ndarray,
        drone_vel: np.ndarray,
        target_pos_3d: np.ndarray,
        target_vel_3d: np.ndarray,
        target_heading: float
    ):
        """
        Calculates 3D velocity setpoint and yaw command for the drone.
        
        target_pos_3d: [x_b, y_b, z_b] (z_b is surface level, usually 0)
        target_vel_3d: [vx_b, vy_b, vz_b]
        target_heading: heading angle of boat in radians
        """
        # 1. Boat heading vectors
        cos_h = np.cos(target_heading)
        sin_h = np.sin(target_heading)
        u_fwd = np.array([cos_h, sin_h, 0.0])
        u_lat = np.array([-sin_h, cos_h, 0.0])

        # 2. Desired 3D position relative to the boat
        # Trailing behind: -standoff_distance * u_fwd
        # Flank offset: lateral_offset * u_lat
        p_desired = (
            target_pos_3d
            - self.standoff_distance * u_fwd
            + self.lateral_offset * u_lat
        )
        p_desired[2] = self.standoff_altitude

        # 3. Position and velocity error
        pos_error = p_desired - drone_pos
        vel_error = target_vel_3d - drone_vel

        # 4. Feed-forward target velocity + PD tracking feedback
        # vx, vy horizontal control
        v_xy_cmd = (
            target_vel_3d[:2]
            + self.kp_pos * pos_error[:2]
            + self.kd_pos * vel_error[:2]
        )

        # vz altitude control
        vz_cmd = self.kp_alt * pos_error[2]

        v_cmd_3d = np.array([v_xy_cmd[0], v_xy_cmd[1], vz_cmd])

        # 5. Yaw command: Point camera / drone directly at the target boat
        dx = target_pos_3d[0] - drone_pos[0]
        dy = target_pos_3d[1] - drone_pos[1]
        yaw_to_target = np.arctan2(dy, dx)

        # Diagnostics & Tracking Metrics
        actual_distance = float(np.linalg.norm(drone_pos[:2] - target_pos_3d[:2]))
        standoff_error = abs(actual_distance - self.standoff_distance)
        relative_speed_error = float(np.linalg.norm(drone_vel[:2] - target_vel_3d[:2]))

        telemetry = {
            "desired_position": p_desired,
            "actual_distance": actual_distance,
            "standoff_error": standoff_error,
            "altitude_error": abs(drone_pos[2] - self.standoff_altitude),
            "relative_speed_error": relative_speed_error,
            "target_speed": float(np.linalg.norm(target_vel_3d[:2])),
            "drone_speed": float(np.linalg.norm(drone_vel[:2])),
        }

        return v_cmd_3d, yaw_to_target, telemetry
