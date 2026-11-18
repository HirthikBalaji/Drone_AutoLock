# Drone AutoLock & Standoff Tracking System 🚁🛥️

Autonomous aerial visual lock-on, Extended Kalman Filter (EKF) motion estimation, 3D standoff flight guidance, and ROS 2 / RViz2 & Gazebo simulation for high-speed maritime vessels (speedboats and yachts).

---

## 🌊 The Problem
In maritime surveillance and interception:
- Speedboats and yachts travel at high speeds (20–45+ knots) and carve sharp, non-linear S-turns.
- Heavy white-water wake, engine spray, wave bobbing, and harsh sun glint cause standard object detectors (like naive YOLO) to momentarily lose detection.
- Once detection drops, standard drones lose the lock and drift away, failing to follow or accompany the vessel.

---

## 💡 The Solution
This repository implements an end-to-end, mathematically verified GNC (Guidance, Navigation & Control) pipeline:

```
[Camera Video Feed / LOTUSim Sensor Stream]
        │
        ▼
[YOLO Detector + High-Rate Optical Tracker]
        │   (Detection or Bounding Box)
        ▼
[Maritime CTRV Extended Kalman Filter (EKF)]
        │ ├── Coordinated Turn & Rate Velocity Motion Model
        │ ├── Outlier Gating via Normalized Innovation Squared (NIS)
        │ └── Autonomous COASTING Mode through spray/glare dropouts
        ▼
[3D Metric Standoff Guidance Controller]
        │ ├── Relative Standoff Vector (Distance, Altitude, Flank Offset)
        │ └── Feed-forward Speed Matching + PD Attitude Command
        ▼
[6-DOF Drone Flight Kinematics / ROS 2 /cmd_vel]
```

---

## 🎮 Simulation & Showcase Options

### 1. RViz2 3D Tactical Simulation (ROS 2 Jazzy + WSLg)
Experience full 3D hardware-accelerated simulation with embedded live camera feed:
- **Double-click**: `launch_rviz.bat`
- **Or command**:
  ```bash
  wsl -d Ubuntu-24.04 bash scripts/launch_rviz.sh
  ```
- **What RViz2 Displays**:
  - **3D Fast Yacht**: Navy blue hull, white cabin, pointed bow wedge, and dynamic expanding wake foam trails.
  - **3D Quadcopter Drone**: Central hub, 4 diagonal motor arms, and 4 spinning propeller discs with banking dynamics.
  - **Active Optical Gimbal Frustum**: Cyan line-of-sight light cone locked onto the vessel deck.
  - **CTRV Trajectory Prediction Spline**: 3D future path arc predicted by the EKF.
  - **3D Floating Telemetry HUD**: Live knot speed, standoff distance, and lock state.
  - **Embedded Live Camera Feed**: RViz2 Image panel subscribed to `/drone/camera/image_locked` showing the real-time locking video!

---

### 2. Standalone Real-Time Video Player (OpenCV 60 FPS)
Smooth real-time video locking with HUD reticle, CTRV trajectory arc, and interactive keyboard controls:
- **Double-click**: `run_realtime_locking.bat`
- **Or command**:
  ```bash
  python scripts/show_realtime_locking.py
  ```
- **Interactive Keys**:
  - `[Space]` : Pause / Resume
  - `[S]` : Toggle Wave Spray / Glare Occlusion (demonstrates autonomous EKF Coasting)
  - `[R]` : Reset to start
  - `[1]` / `[2]` : Switch between Real Sea Yacht & Speedboat S-Curve Maneuvers
  - `[Q]` / `[Esc]` : Exit

---

### 3. Gazebo Harmonic Simulation (Naval Group LOTUSim)
- **Double-click**: `launch_gazebo.bat`
- **Or command**:
  ```bash
  wsl -d Ubuntu-24.04 bash scripts/launch_gazebo.sh
  ```
- Uses `data/lotusim/drone_autolock_world.sdf` generated with 6-DOF hydrodynamic added mass and damping matrices according to the LOTUSim Xdyn specification.

---

### 4. Interactive Web Dashboard (Streamlit)
```bash
streamlit run app.py
```
Open `http://localhost:8501` to access:
- **Tab 1**: Live 3D WebGL Digital Twin (WebGL/Three.js with wave shaders, drone chase cam, and gimbal view).
- **Tab 2**: Real-Time Video Locking Feed (Auto-looping 1080p H.264 stream, 20 FPS live simulation stream, and timeline scrubber).
- **Tab 3**: Benchmark & Proof of Optimality comparison tables.
- **Tab 4**: ROS 2 / RViz2 & Gazebo launchers with architecture overview.

---

## 📊 Benchmark Verification Results

Evaluated on **Real Maritime Drone Video** from the **Maritime Visual Tracking Dataset (MVTD)**:

| Metric | Baseline Naive Tracker | Maritime AutoLock (EKF + GNC) |
|---|---|---|
| **Lock Retention Rate** | 74.0% (Drops during occlusions) | **100.0% (Continuous Lock)** |
| **IoU Success Rate (IoU > 0.5)** | 74.0% | **82.0%** |
| **Mean Center Error (CLE)** | Diverged on drops | **3.89 pixels** |
| **Mean Flight Standoff Error** | Diverged (No GNC) | **5.96 meters** |
| **Recovery from Glare / Spray** | ❌ Fails / Target Lost | **✅ 100% Seamless EKF Coasting** |

---

## 📁 Repository Structure
```
Drone_AutoLock/
├── app.py                         # Interactive Streamlit Web Dashboard
├── run_benchmark.py               # Comparative benchmark evaluation
├── launch_rviz.bat                # 1-Click RViz2 launcher (Windows)
├── launch_gazebo.bat              # 1-Click Gazebo launcher (Windows)
├── run_realtime_locking.bat       # 1-Click Real-time video player (Windows)
├── config/
│   └── maritime_autolock.rviz     # Preconfigured RViz2 layout
├── data/
│   ├── sample_videos/             # Real MVTD footage & speedboat maneuvers
│   ├── outputs/                   # H.264 real-time locked video exports
│   └── lotusim/                   # LOTUSim scenario JSON & Gazebo SDF
├── src/
│   ├── ros2/
│   │   └── maritime_autolock_node.py # ROS 2 node publishing 3D markers & video
│   ├── estimation/
│   │   └── maritime_ekf.py        # CTRV EKF with autonomous coasting
│   ├── gnc/
│   │   ├── drone_kinematics.py    # 6-DOF Drone physics simulator
│   │   └── guidance_controller.py # Accompany & velocity matching guidance
│   ├── vision/
│   │   ├── detector.py            # YOLO Maritime Vessel Detector
│   │   └── visual_tracker.py      # High-rate optical appearance tracker
│   ├── simulation/
│   │   └── closed_loop_sim.py     # Synchronized vision-flight engine
│   ├── benchmark/
│   │   └── metrics.py             # CLE, IoU, and Standoff Error metrics
│   └── visualization/
│       └── lotusim_3d_showcase.html # 3D WebGL Digital Twin
└── scripts/
    ├── generate_locked_video.py   # Video exporter (H.264 MP4)
    ├── show_realtime_locking.py   # Native OpenCV real-time player
    ├── launch_rviz.sh             # ROS 2 RViz2 launcher script
    └── launch_gazebo.sh           # Gazebo Harmonic launcher script
```
