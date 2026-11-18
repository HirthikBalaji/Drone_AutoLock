#!/usr/bin/env bash
# ==============================================================================
# Maritime Drone AutoLock - Gazebo Harmonic (GZ Sim) Launcher
# ==============================================================================

set -e

source /opt/ros/jazzy/setup.bash

PROJECT_DIR="/mnt/c/Users/Suz Machine Tech/Documents/projects/Drone_AutoLock"
cd "$PROJECT_DIR"

WORLD_FILE="$PROJECT_DIR/data/lotusim/drone_autolock_world.sdf"

echo "======================================================================"
echo " Launching Gazebo Harmonic Maritime Simulation World..."
echo " World File: $WORLD_FILE"
echo "======================================================================"

gz sim "$WORLD_FILE" -r
