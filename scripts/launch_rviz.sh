#!/usr/bin/env bash
# ==============================================================================
# Maritime Drone AutoLock - ROS 2 Jazzy & RViz2 3D Simulation Launcher
# ==============================================================================

set -e

source /opt/ros/jazzy/setup.bash

PROJECT_DIR="/mnt/c/Users/Suz Machine Tech/Documents/projects/Drone_AutoLock"
cd "$PROJECT_DIR"

echo "======================================================================"
echo " Starting Maritime Drone AutoLock ROS 2 Simulation Node..."
echo "======================================================================"

# Start the simulation node in background
python3 src/ros2/maritime_autolock_node.py &
SIM_PID=$!

cleanup() {
    echo ""
    echo "Stopping Maritime Simulation Node (PID $SIM_PID)..."
    kill $SIM_PID 2>/dev/null || true
    echo "Simulation stopped cleanly."
}
trap cleanup EXIT INT TERM

sleep 1.5

echo "======================================================================"
echo " Launching RViz2 Tactical Simulation Window (WSLg Hardware Accelerated)..."
echo " Displays:"
echo "   - 3D Fast Yacht (Navy Blue Hull, White Superstructure, Bow wedge)"
echo "   - 3D Drone (Quadcopter, 4 Motor Arms, Spinning Propellers)"
echo "   - Active Optical Gimbal Frustum Light Cone (Cyan Line-of-Sight)"
echo "   - Twin Wake Foam Trails & CTRV Trajectory Prediction Spline"
echo "   - Floating 3D Tactical Telemetry HUD Marker"
echo "   - Embedded Live Drone Camera AutoLock Feed (/drone/camera/image_locked)"
echo "======================================================================"

rviz2 -d config/maritime_autolock.rviz

echo "RViz2 closed by user."
