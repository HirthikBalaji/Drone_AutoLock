"""
6-DOF Drone Kinematics and Flight Dynamics Simulator.
Simulates realistic UAV flight physics, velocity response lag, drag, and tilt angles.
"""

import numpy as np


class DroneKinematics:
    def __init__(
        self,
        init_pos=(0.0, -30.0, 15.0), # (x, y, z) in meters (z = altitude above water)
        init_yaw=0.0,
        max_speed=25.0,              # Max speed in m/s (approx 90 km/h for pursuit)
        max_accel=7.5,               # Max acceleration in m/s^2
        max_yaw_rate=np.radians(90), # 90 deg/s max turn rate
        time_constant=0.25           # Motor/velocity controller response lag (seconds)
    ):
        # 3D position [x, y, z] (meters)
        self.pos = np.array(init_pos, dtype=float)
        # 3D velocity [vx, vy, vz] (m/s)
        self.vel = np.zeros(3, dtype=float)
        # Attitude [roll, pitch, yaw] (radians)
        self.attitude = np.array([0.0, 0.0, init_yaw], dtype=float)

        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_yaw_rate = max_yaw_rate
        self.tau = time_constant

        # Trajectory history for visualization
        self.history = [self.pos.copy()]

    def step(self, vel_cmd: np.ndarray, yaw_cmd: float, dt: float = 0.05):
        """
        Integrates drone kinematics over time step dt.
        vel_cmd: desired [vx, vy, vz]
        yaw_cmd: desired yaw angle in radians
        """
        # 1. Saturate commanded velocity
        speed_cmd = np.linalg.norm(vel_cmd)
        if speed_cmd > self.max_speed:
            vel_cmd = (vel_cmd / speed_cmd) * self.max_speed

        # 2. Acceleration with first-order motor lag
        raw_accel = (vel_cmd - self.vel) / self.tau
        accel_mag = np.linalg.norm(raw_accel)
        if accel_mag > self.max_accel:
            accel = (raw_accel / accel_mag) * self.max_accel
        else:
            accel = raw_accel

        # 3. Integrate velocity and position
        self.vel += accel * dt
        self.pos += self.vel * dt

        # Ground safety clamp (drone cannot go below sea level)
        if self.pos[2] < 1.0:
            self.pos[2] = 1.0
            self.vel[2] = max(0.0, self.vel[2])

        # 4. Yaw control with rate limiting
        current_yaw = self.attitude[2]
        yaw_error = (yaw_cmd - current_yaw + np.pi) % (2 * np.pi) - np.pi
        yaw_rate = np.clip(yaw_error / self.tau, -self.max_yaw_rate, self.max_yaw_rate)
        new_yaw = (current_yaw + yaw_rate * dt + np.pi) % (2 * np.pi) - np.pi
        self.attitude[2] = new_yaw

        # 5. Approximate roll and pitch from horizontal acceleration
        # Roll ~ -a_lateral / g, Pitch ~ a_forward / g
        g = 9.81
        cos_y, sin_y = np.cos(new_yaw), np.sin(new_yaw)
        a_fwd = accel[0] * cos_y + accel[1] * sin_y
        a_lat = -accel[0] * sin_y + accel[1] * cos_y
        self.attitude[1] = np.clip(a_fwd / g, -np.radians(35), np.radians(35))
        self.attitude[0] = np.clip(-a_lat / g, -np.radians(35), np.radians(35))

        self.history.append(self.pos.copy())
        return self.pos, self.vel, self.attitude

    def get_state(self):
        return {
            "position": self.pos.copy(),
            "velocity": self.vel.copy(),
            "speed": float(np.linalg.norm(self.vel)),
            "attitude": self.attitude.copy(),
            "yaw_deg": float(np.degrees(self.attitude[2]))
        }
