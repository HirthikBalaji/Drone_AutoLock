"""
Naval Group LOTUSim Scenario & Model Definition Generator.
Generates compliant LOTUSim Scenarios (JSON), Gazebo SDFs, and Xdyn 6-DOF Hydrodynamic Configurations.
"""

import json
import os


def generate_lotusim_scenario(
    output_path: str = "data/lotusim/scenario_autolock_accompany.json",
    ref_lat: float = 43.12,    # Toulon Naval Base / Mediterranean Sea testbed
    ref_lon: float = 5.92,
    boat_speed_knots: float = 32.0,
    standoff_dist: float = 25.0,
    standoff_alt: float = 15.0
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    scenario = {
        "name": "HighSpeed_Vessel_AutoLock_Accompany",
        "description": "Autonomous aerial drone lock-on and accompany mission on a high-speed yacht/USV under wave disturbances.",
        "version": "1.0.0",
        "referencePosition": {
            "latitude": ref_lat,
            "longitude": ref_lon,
            "altitude": 0.0
        },
        "environment": {
            "wind": {
                "speed": 8.5,        # m/s (~16.5 knots, Beaufort 5)
                "direction": 45.0     # degrees
            },
            "current": {
                "speed": 1.2,        # m/s (~2.3 knots surface current)
                "direction": 90.0
            },
            "waves": {
                "significant_height": 1.25, # meters (Sea State 3)
                "peak_period": 5.5,         # seconds
                "direction": 45.0
            }
        },
        "agents": [
            {
                "name": "Target_HighSpeed_Yacht",
                "model": "high_speed_usv",
                "domain": "surface",
                "position": {
                    "north": 0.0,
                    "east": 0.0,
                    "altitude": 0.0
                },
                "heading": 0.0, # radians
                "linear_velocity": {
                    "u": boat_speed_knots * 0.514444, # knots to m/s
                    "v": 0.0,
                    "w": 0.0
                },
                "physicsInterface": {
                    "engine": "xdyn",
                    "mass": 3200.0, # kg (fast sport yacht hull)
                    "added_mass": [
                        [120.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1850.0, 0.0, 0.0, 0.0, 420.0],
                        [0.0, 0.0, 2900.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 450.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 3100.0, 0.0],
                        [0.0, 420.0, 0.0, 0.0, 0.0, 2200.0]
                    ],
                    "linear_damping": [-75.0, -1200.0, -1800.0, -220.0, -1400.0, -980.0],
                    "quadratic_damping": [-18.0, -450.0, -620.0, -85.0, -520.0, -380.0]
                },
                "waypointInterface": {
                    "active": True,
                    "waypoints": [
                        {"north": 0.0, "east": 0.0, "speed": 15.0},
                        {"north": 120.0, "east": 180.0, "speed": 17.0},
                        {"north": 280.0, "east": 240.0, "speed": 16.5},
                        {"north": 450.0, "east": 150.0, "speed": 18.0}
                    ]
                }
            },
            {
                "name": "Surveillance_Drone",
                "model": "quadcopter_uav",
                "domain": "aerial",
                "position": {
                    "north": -standoff_dist,
                    "east": 0.0,
                    "altitude": standoff_alt
                },
                "heading": 0.0,
                "sensors": [
                    {
                        "type": "gimbal_camera",
                        "topic": "/surveillance_drone/camera/image_raw",
                        "update_rate": 30.0,
                        "fov": 1.22, # ~70 degrees
                        "resolution": {"width": 1920, "height": 1080}
                    },
                    {
                        "type": "gnss_ins",
                        "topic": "/surveillance_drone/nav/fix",
                        "update_rate": 50.0
                    }
                ],
                "controlInterface": {
                    "type": "ROS2",
                    "cmd_vel_topic": "/surveillance_drone/cmd_vel",
                    "pose_topic": "/surveillance_drone/pose"
                }
            }
        ]
    }

    with open(output_path, "w") as f:
        json.dump(scenario, f, indent=2)

    print(f"Generated LOTUSim scenario at {output_path}")
    return scenario


def generate_lotusim_gazebo_sdf(output_path: str = "data/lotusim/drone_autolock_world.sdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sdf_content = """<?xml version="1.0" ?>
<!--
  LOTUSim Maritime World: High-Speed Vessel AutoLock & Accompany Mission
  Compatible with Naval Group LOTUSim, Gazebo Fortress/Garden, and ROS 2
-->
<sdf version="1.9">
  <world name="lotusim_maritime_autolock">
    <physics name="1ms" type="ignored">
      <max_step_size>0.002</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Sun Lighting with Water Specular Glint -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 100 0.5 0.3 0</pose>
      <diffuse>0.9 0.9 0.95 1</diffuse>
      <specular>0.7 0.7 0.8 1</specular>
      <direction>-0.4 0.2 -0.9</direction>
    </light>

    <!-- Ocean Surface Plane with Wave Visuals -->
    <model name="ocean_surface">
      <static>true</static>
      <link name="ocean_link">
        <visual name="ocean_visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>2000 2000</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.05 0.15 0.25 0.9</ambient>
            <diffuse>0.1 0.35 0.55 0.9</diffuse>
            <specular>0.8 0.8 0.9 0.9</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- LOTUSim Hydrodynamics Plugin Interface -->
    <plugin
      filename="libLOTUSimPhysicsEngineInterface.so"
      name="lotusim::gazebo::PhysicsEngineInterface">
      <ros_namespace>lotusim</ros_namespace>
    </plugin>

    <!-- Aerial MAS Entity Manager -->
    <plugin
      filename="libAerialMAS.so"
      name="lotusim::gazebo::AerialMAS">
      <aerial_namespace>aerialWorld</aerial_namespace>
    </plugin>
  </world>
</sdf>
"""
    with open(output_path, "w") as f:
        f.write(sdf_content)

    print(f"Generated LOTUSim SDF world at {output_path}")


if __name__ == "__main__":
    generate_lotusim_scenario()
    generate_lotusim_gazebo_sdf()
