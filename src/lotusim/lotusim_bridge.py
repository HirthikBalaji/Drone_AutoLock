"""
LOTUSim Telemetry & ROS 2 / WebSocket Bridge.
Bridges our Maritime AutoLock GNC pipeline to Naval Group's LOTUSim simulation runtime.
Supports:
1. Direct WebSocket streaming to LOTUSim UI Backend (port 5000).
2. Standalone Hydrodynamic Simulation compatible with LOTUSim state packets.
"""

import json
import time
import numpy as np


class LOTUSimBridge:
    def __init__(self, host: str = "localhost", port: int = 5000):
        self.host = host
        self.port = port
        self.ws_url = f"ws://{host}:{port}"
        self.is_connected = False

    def export_scenario_payload(self, scenario_dict: dict) -> str:
        """Serializes scenario into LOTUSim REST API payload"""
        return json.dumps(scenario_dict, indent=2)

    def format_ros2_cmd_vel(self, vx: float, vy: float, vz: float, yaw_rate: float) -> dict:
        """
        Formats geometry_msgs/msg/Twist for LOTUSim drone flight controller:
        topic: /surveillance_drone/cmd_vel
        """
        return {
            "topic": "/surveillance_drone/cmd_vel",
            "msg_type": "geometry_msgs/msg/Twist",
            "data": {
                "linear": {
                    "x": float(vx),
                    "y": float(vy),
                    "z": float(vz)
                },
                "angular": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": float(yaw_rate)
                }
            },
            "timestamp": time.time()
        }

    def format_lotusim_agent_state(self, name: str, pos_3d: np.ndarray, vel_3d: np.ndarray, heading_rad: float) -> dict:
        """
        Formats agent state in LOTUSim UI WebSocket format.
        """
        return {
            "name": name,
            "position": {
                "north": float(pos_3d[0]),
                "east": float(pos_3d[1]),
                "altitude": float(pos_3d[2])
            },
            "velocity": {
                "u": float(vel_3d[0]),
                "v": float(vel_3d[1]),
                "w": float(vel_3d[2])
            },
            "heading": float(heading_rad),
            "heading_deg": float(np.degrees(heading_rad)),
            "speed_knots": float(np.linalg.norm(vel_3d[:2]) * 1.94384)
        }
