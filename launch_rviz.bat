@echo off
title Maritime Drone AutoLock - RViz2 3D Simulation
echo ======================================================================
echo Launching Maritime Drone AutoLock in RViz2 (ROS 2 Jazzy + WSLg)
echo ======================================================================
echo This launches:
echo   1. ROS 2 Node (CTRV EKF + 6-DOF Drone Kinematics + Standoff Guidance)
echo   2. Real-time Camera Feed (/drone/camera/image_locked)
echo   3. RViz2 Tactical 3D Digital Twin (Yacht, Drone, Gimbal Frustum, Wake)
echo ======================================================================

wsl -d Ubuntu-24.04 bash "/mnt/c/Users/Suz Machine Tech/Documents/projects/Drone_AutoLock/scripts/launch_rviz.sh"

pause
