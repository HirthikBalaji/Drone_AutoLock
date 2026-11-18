#!/usr/bin/env python3
"""
ROS 2 Maritime Drone AutoLock & Accompany Simulation Node.
Runs on ROS 2 Jazzy.

Publishes:
  - /drone/camera/image_locked    (sensor_msgs/Image): Real-time optical gimbal tracking HUD feed
  - /maritime/markers            (visualization_msgs/MarkerArray): 3D Yacht, Drone, Frustum, Wake, CTRV Spline
  - /drone/path                  (nav_msgs/Path): Drone flight trajectory
  - /yacht/path                  (nav_msgs/Path): Yacht sea trajectory
  - TF: world -> yacht, world -> drone, drone -> gimbal_camera
"""

import math
import os
import sys
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster


# Euler to Quaternion helper
def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


class MaritimeAutoLockNode(Node):
    def __init__(self):
        super().__init__('maritime_autolock_node')
        self.get_logger().info("Initializing Maritime Drone AutoLock ROS 2 Simulation...")

        # Publishers
        self.image_pub = self.create_publisher(Image, '/drone/camera/image_locked', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/maritime/markers', 10)
        self.drone_path_pub = self.create_publisher(Path, '/drone/path', 10)
        self.yacht_path_pub = self.create_publisher(Path, '/yacht/path', 10)

        # TF Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # CV Bridge
        self.bridge = CvBridge()

        # Paths
        self.drone_path = Path()
        self.drone_path.header.frame_id = 'world'
        self.yacht_path = Path()
        self.yacht_path.header.frame_id = 'world'

        # Load video frames for the HUD camera stream
        self._load_video_stream()

        # Simulation state
        self.sim_time = 0.0
        self.dt = 0.05  # 20 Hz

        # Speedboat motion parameters (Racetrack maneuver)
        self.vessel_speed = 14.5  # m/s (~28.2 knots)
        self.vessel_yaw = 0.0
        self.vessel_yaw_rate = 0.0
        self.vessel_pos = np.array([0.0, 0.0, 0.0])
        self.wake_history = []

        # Drone 6-DOF state
        self.drone_pos = np.array([-25.0, -15.0, 18.0])
        self.drone_vel = np.array([0.0, 0.0, 0.0])
        self.drone_attitude = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.rotor_spin = 0.0

        # Guidance parameters
        self.standoff_dist = 28.0
        self.standoff_alt = 18.0

        # Timer: 20 Hz
        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info("Maritime Drone AutoLock Simulation Node READY. Publishing at 20 Hz.")

    def _load_video_stream(self):
        # Look for the generated locked video, or fallback to source video
        cand_paths = [
            "/mnt/c/Users/Suz Machine Tech/Documents/projects/Drone_AutoLock/data/outputs/real_maritime_boat1_h264.mp4",
            "/mnt/c/Users/Suz Machine Tech/Documents/projects/Drone_AutoLock/data/outputs/real_maritime_boat1_locked.mp4",
            "/mnt/c/Users/Suz Machine Tech/Documents/projects/Drone_AutoLock/data/sample_videos/real_maritime_boat1.mp4",
            "data/outputs/real_maritime_boat1_h264.mp4",
            "data/outputs/real_maritime_boat1_locked.mp4"
        ]
        self.video_cap = None
        for p in cand_paths:
            if os.path.exists(p):
                self.video_cap = cv2.VideoCapture(p)
                self.get_logger().info(f"Loaded camera video stream from: {p}")
                break

        if not self.video_cap or not self.video_cap.isOpened():
            self.get_logger().warn("Could not open video file. Will generate synthetic camera feed.")
            self.video_cap = None

    def timer_callback(self):
        now = self.get_clock().now()
        t = self.sim_time
        self.sim_time += self.dt

        # --- 1. Vessel Hydrodynamic CTRV Motion ---
        # Smooth high-speed S-curve and racetrack path
        curve_period = 25.0
        self.vessel_yaw_rate = 0.12 * math.cos(2.0 * math.pi * t / curve_period)
        self.vessel_yaw += self.vessel_yaw_rate * self.dt

        vx = self.vessel_speed * math.cos(self.vessel_yaw)
        vy = self.vessel_speed * math.sin(self.vessel_yaw)
        self.vessel_pos[0] += vx * self.dt
        self.vessel_pos[1] += vy * self.dt
        # Slight pitch and roll due to sea waves
        vessel_pitch = 0.04 * math.sin(3.0 * t)
        vessel_roll = -0.15 * (self.vessel_yaw_rate / 0.15)  # Inward banking

        # Wake trail memory
        if len(self.wake_history) == 0 or np.linalg.norm(self.vessel_pos[:2] - self.wake_history[-1][:2]) > 2.0:
            self.wake_history.append(self.vessel_pos.copy())
            if len(self.wake_history) > 40:
                self.wake_history.pop(0)

        # --- 2. Drone 3D Standoff Guidance & Flight Dynamics ---
        # Desired position: trailing behind the yacht along vessel heading
        trail_angle = self.vessel_yaw + math.pi  # Directly aft
        p_des = np.array([
            self.vessel_pos[0] + self.standoff_dist * math.cos(trail_angle),
            self.vessel_pos[1] + self.standoff_dist * math.sin(trail_angle),
            self.standoff_alt
        ])

        # Feedforward vessel velocity + proportional position correction
        p_err = p_des - self.drone_pos
        v_des = np.array([vx, vy, 0.0]) + 0.85 * p_err
        speed_des = np.linalg.norm(v_des[:2])
        if speed_des > 32.0:
            v_des[:2] = (v_des[:2] / speed_des) * 32.0

        # Drone physical lag (tau = 0.22s)
        alpha = self.dt / (0.22 + self.dt)
        self.drone_vel += alpha * (v_des - self.drone_vel)
        self.drone_pos += self.drone_vel * self.dt

        # Drone attitude (faces vessel + banks into turns)
        los_vector = self.vessel_pos - self.drone_pos
        desired_yaw = math.atan2(los_vector[1], los_vector[0])
        # Smooth yaw
        self.drone_attitude[2] += 0.20 * (desired_yaw - self.drone_attitude[2])
        # Bank roll and pitch proportional to horizontal acceleration
        accel_x = (v_des[0] - self.drone_vel[0]) / 0.22
        accel_y = (v_des[1] - self.drone_vel[1]) / 0.22
        self.drone_attitude[0] = -0.05 * accel_y  # roll
        self.drone_attitude[1] = 0.05 * accel_x   # pitch

        self.rotor_spin = (self.rotor_spin + 1.2) % (2.0 * math.pi)

        # --- 3. Publish TFs ---
        yacht_q = euler_to_quaternion(vessel_roll, vessel_pitch, self.vessel_yaw)
        t_yacht = TransformStamped()
        t_yacht.header.stamp = now.to_msg()
        t_yacht.header.frame_id = 'world'
        t_yacht.child_frame_id = 'yacht'
        t_yacht.transform.translation.x = float(self.vessel_pos[0])
        t_yacht.transform.translation.y = float(self.vessel_pos[1])
        t_yacht.transform.translation.z = float(self.vessel_pos[2])
        t_yacht.transform.rotation = yacht_q
        self.tf_broadcaster.sendTransform(t_yacht)

        drone_q = euler_to_quaternion(self.drone_attitude[0], self.drone_attitude[1], self.drone_attitude[2])
        t_drone = TransformStamped()
        t_drone.header.stamp = now.to_msg()
        t_drone.header.frame_id = 'world'
        t_drone.child_frame_id = 'drone'
        t_drone.transform.translation.x = float(self.drone_pos[0])
        t_drone.transform.translation.y = float(self.drone_pos[1])
        t_drone.transform.translation.z = float(self.drone_pos[2])
        t_drone.transform.rotation = drone_q
        self.tf_broadcaster.sendTransform(t_drone)

        # --- 4. Trajectory Paths ---
        pose_y = PoseStamped()
        pose_y.header.stamp = now.to_msg()
        pose_y.header.frame_id = 'world'
        pose_y.pose.position.x = float(self.vessel_pos[0])
        pose_y.pose.position.y = float(self.vessel_pos[1])
        pose_y.pose.position.z = 0.1
        self.yacht_path.poses.append(pose_y)
        if len(self.yacht_path.poses) > 400:
            self.yacht_path.poses.pop(0)
        self.yacht_path.header.stamp = now.to_msg()
        self.yacht_path_pub.publish(self.yacht_path)

        pose_d = PoseStamped()
        pose_d.header.stamp = now.to_msg()
        pose_d.header.frame_id = 'world'
        pose_d.pose.position.x = float(self.drone_pos[0])
        pose_d.pose.position.y = float(self.drone_pos[1])
        pose_d.pose.position.z = float(self.drone_pos[2])
        self.drone_path.poses.append(pose_d)
        if len(self.drone_path.poses) > 400:
            self.drone_path.poses.pop(0)
        self.drone_path.header.stamp = now.to_msg()
        self.drone_path_pub.publish(self.drone_path)

        # --- 5. Publish 3D Markers (Yacht, Drone, Gimbal Cone, Spline, Wake, HUD Text) ---
        markers = self._generate_markers(now)
        self.marker_pub.publish(markers)

        # --- 6. Publish Real-Time Camera HUD Feed ---
        self._publish_camera_feed(now)

    def _generate_markers(self, now) -> MarkerArray:
        arr = MarkerArray()
        t_stamp = now.to_msg()

        # Marker 1: Yacht Hull (Sleek deep navy blue hull)
        m_hull = Marker()
        m_hull.header.frame_id = 'yacht'
        m_hull.header.stamp = t_stamp
        m_hull.ns = 'vessel'
        m_hull.id = 1
        m_hull.type = Marker.CUBE
        m_hull.action = Marker.ADD
        m_hull.pose.position.x = 0.0
        m_hull.pose.position.y = 0.0
        m_hull.pose.position.z = 0.6
        m_hull.pose.orientation.w = 1.0
        m_hull.scale.x = 10.0  # Length
        m_hull.scale.y = 3.2   # Beam
        m_hull.scale.z = 1.6   # Depth
        m_hull.color.r = 0.12
        m_hull.color.g = 0.22
        m_hull.color.b = 0.38
        m_hull.color.a = 1.0
        arr.markers.append(m_hull)

        # Marker 2: Yacht Cabin (White superstructure)
        m_cabin = Marker()
        m_cabin.header.frame_id = 'yacht'
        m_cabin.header.stamp = t_stamp
        m_cabin.ns = 'vessel'
        m_cabin.id = 2
        m_cabin.type = Marker.CUBE
        m_cabin.action = Marker.ADD
        m_cabin.pose.position.x = -0.5
        m_cabin.pose.position.y = 0.0
        m_cabin.pose.position.z = 1.8
        m_cabin.pose.orientation.w = 1.0
        m_cabin.scale.x = 4.8
        m_cabin.scale.y = 2.4
        m_cabin.scale.z = 1.2
        m_cabin.color.r = 0.95
        m_cabin.color.g = 0.96
        m_cabin.color.b = 0.98
        m_cabin.color.a = 1.0
        arr.markers.append(m_cabin)

        # Marker 3: Bow tip wedge (Marker.LINE_LIST / pointed nose)
        m_bow = Marker()
        m_bow.header.frame_id = 'yacht'
        m_bow.header.stamp = t_stamp
        m_bow.ns = 'vessel'
        m_bow.id = 3
        m_bow.type = Marker.LINE_STRIP
        m_bow.action = Marker.ADD
        m_bow.scale.x = 0.35
        m_bow.color.r = 0.0
        m_bow.color.g = 0.9
        m_bow.color.b = 1.0
        m_bow.color.a = 1.0
        m_bow.points = [
            Point(x=5.0, y=-1.6, z=1.2),
            Point(x=7.2, y=0.0, z=1.6),
            Point(x=5.0, y=1.6, z=1.2)
        ]
        arr.markers.append(m_bow)

        # Marker 4: Expanding Twin Wake Foam Trails
        m_wake = Marker()
        m_wake.header.frame_id = 'world'
        m_wake.header.stamp = t_stamp
        m_wake.ns = 'wake'
        m_wake.id = 4
        m_wake.type = Marker.LINE_LIST
        m_wake.action = Marker.ADD
        m_wake.scale.x = 0.4
        m_wake.color.r = 0.8
        m_wake.color.g = 0.95
        m_wake.color.b = 1.0
        m_wake.color.a = 0.75
        pts_wake = []
        for i, wp in enumerate(self.wake_history):
            age_factor = (i + 1) / float(len(self.wake_history) + 1)
            spread = (1.0 - age_factor) * 4.5 + 1.2
            pts_wake.append(Point(x=wp[0], y=wp[1] - spread, z=0.05))
            pts_wake.append(Point(x=wp[0], y=wp[1] + spread, z=0.05))
        m_wake.points = pts_wake
        arr.markers.append(m_wake)

        # Marker 5: Drone Airframe Hub
        m_drone = Marker()
        m_drone.header.frame_id = 'drone'
        m_drone.header.stamp = t_stamp
        m_drone.ns = 'drone'
        m_drone.id = 5
        m_drone.type = Marker.SPHERE
        m_drone.action = Marker.ADD
        m_drone.pose.position.z = 0.0
        m_drone.pose.orientation.w = 1.0
        m_drone.scale.x = 0.7
        m_drone.scale.y = 0.7
        m_drone.scale.z = 0.3
        m_drone.color.r = 0.1
        m_drone.color.g = 0.1
        m_drone.color.b = 0.15
        m_drone.color.a = 1.0
        arr.markers.append(m_drone)

        # Marker 6: Drone Arms
        m_arms = Marker()
        m_arms.header.frame_id = 'drone'
        m_arms.header.stamp = t_stamp
        m_arms.ns = 'drone'
        m_arms.id = 6
        m_arms.type = Marker.LINE_LIST
        m_arms.action = Marker.ADD
        m_arms.scale.x = 0.12
        m_arms.color.r = 0.3
        m_arms.color.g = 0.35
        m_arms.color.b = 0.4
        m_arms.color.a = 1.0
        arm_d = 0.8
        m_arms.points = [
            Point(x=-arm_d, y=-arm_d, z=0.0), Point(x=arm_d, y=arm_d, z=0.0),
            Point(x=-arm_d, y=arm_d, z=0.0), Point(x=arm_d, y=-arm_d, z=0.0)
        ]
        arr.markers.append(m_arms)

        # Marker 7: 4 Spinning Propeller Discs
        m_props = Marker()
        m_props.header.frame_id = 'drone'
        m_props.header.stamp = t_stamp
        m_props.ns = 'drone'
        m_props.id = 7
        m_props.type = Marker.CYLINDER
        m_props.action = Marker.ADD
        m_props.pose.position.z = 0.08
        m_props.pose.orientation.w = 1.0
        m_props.scale.x = 1.8
        m_props.scale.y = 1.8
        m_props.scale.z = 0.03
        m_props.color.r = 0.0
        m_props.color.g = 1.0
        m_props.color.b = 0.6
        m_props.color.a = 0.4
        arr.markers.append(m_props)

        # Marker 8: Active Gimbal Optical Frustum Cone (translucent cyan pyramid)
        m_cone = Marker()
        m_cone.header.frame_id = 'world'
        m_cone.header.stamp = t_stamp
        m_cone.ns = 'frustum'
        m_cone.id = 8
        m_cone.type = Marker.LINE_LIST
        m_cone.action = Marker.ADD
        m_cone.scale.x = 0.18
        m_cone.color.r = 0.0
        m_cone.color.g = 1.0
        m_cone.color.b = 0.8
        m_cone.color.a = 0.85

        dp = Point(x=float(self.drone_pos[0]), y=float(self.drone_pos[1]), z=float(self.drone_pos[2]))
        # 4 corners of yacht deck target
        c1 = Point(x=float(self.vessel_pos[0] + 5.0), y=float(self.vessel_pos[1] - 2.0), z=0.8)
        c2 = Point(x=float(self.vessel_pos[0] + 5.0), y=float(self.vessel_pos[1] + 2.0), z=0.8)
        c3 = Point(x=float(self.vessel_pos[0] - 5.0), y=float(self.vessel_pos[1] + 2.0), z=0.8)
        c4 = Point(x=float(self.vessel_pos[0] - 5.0), y=float(self.vessel_pos[1] - 2.0), z=0.8)

        # Rays from drone to 4 corners
        m_cone.points = [
            dp, c1,
            dp, c2,
            dp, c3,
            dp, c4,
            c1, c2,
            c2, c3,
            c3, c4,
            c4, c1
        ]
        arr.markers.append(m_cone)

        # Marker 9: CTRV Future Prediction Trajectory Arc
        m_pred = Marker()
        m_pred.header.frame_id = 'world'
        m_pred.header.stamp = t_stamp
        m_pred.ns = 'prediction'
        m_pred.id = 9
        m_pred.type = Marker.LINE_STRIP
        m_pred.action = Marker.ADD
        m_pred.scale.x = 0.35
        m_pred.color.r = 1.0
        m_pred.color.g = 0.8
        m_pred.color.b = 0.0
        m_pred.color.a = 0.95

        pred_pts = []
        cur_x = self.vessel_pos[0]
        cur_y = self.vessel_pos[1]
        cur_yaw = self.vessel_yaw
        omega = self.vessel_yaw_rate
        speed = self.vessel_speed

        for step in range(25):
            cur_yaw += omega * 0.1
            cur_x += speed * math.cos(cur_yaw) * 0.1
            cur_y += speed * math.sin(cur_yaw) * 0.1
            pred_pts.append(Point(x=float(cur_x), y=float(cur_y), z=0.4))
        m_pred.points = pred_pts
        arr.markers.append(m_pred)

        # Marker 10: 3D Tactical Status Text HUD
        m_text = Marker()
        m_text.header.frame_id = 'world'
        m_text.header.stamp = t_stamp
        m_text.ns = 'hud_text'
        m_text.id = 10
        m_text.type = Marker.TEXT_VIEW_FACING
        m_text.action = Marker.ADD
        m_text.pose.position.x = float(self.vessel_pos[0])
        m_text.pose.position.y = float(self.vessel_pos[1])
        m_text.pose.position.z = 4.5
        m_text.scale.z = 1.2  # Font size
        m_text.color.r = 0.0
        m_text.color.g = 1.0
        m_text.color.b = 0.4
        m_text.color.a = 1.0
        target_kts = self.vessel_speed * 1.94384
        drone_kts = np.linalg.norm(self.drone_vel[:2]) * 1.94384
        actual_standoff = np.linalg.norm(self.drone_pos[:2] - self.vessel_pos[:2])
        m_text.text = f"[AUTO-LOCK ACTIVE]\nVessel: {target_kts:.1f} kts | Drone: {drone_kts:.1f} kts\nStandoff: {actual_standoff:.1f}m (Cmd: {self.standoff_dist:.1f}m)"
        arr.markers.append(m_text)

        return arr

    def _publish_camera_feed(self, now):
        if self.video_cap and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if not ret:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_cap.read()

            if ret:
                # Resize if high resolution for smooth 20 Hz ROS publishing
                if frame.shape[1] > 960:
                    frame = cv2.resize(frame, (960, 540))

                # Add live ROS timestamp HUD
                ts_str = f"ROS2 CLOCK: {self.sim_time:.2f}s | TACTICAL FEED"
                cv2.putText(frame, ts_str, (frame.shape[1] - 380, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)

                msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                msg.header.stamp = now.to_msg()
                msg.header.frame_id = 'gimbal_camera'
                self.image_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MaritimeAutoLockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.video_cap:
            node.video_cap.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
