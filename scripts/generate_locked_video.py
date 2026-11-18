"""
Generate real-time annotated locked video for Maritime Drone AutoLock.
Exports MP4 video files with tactical HUD overlays.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from src.simulation.closed_loop_sim import ClosedLoopSimulation
from src.benchmark.metrics import MaritimeBenchmark


def generate_video(
    video_path: str,
    gt_path: str,
    output_path: str,
    standoff_dist: float = 25.0,
    standoff_alt: float = 15.0,
    m_per_px: float = 0.20,
    simulate_spray: bool = False
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dt = 1.0 / fps

    gt_boxes = []
    if gt_path and os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            for line in f:
                parts = [float(v) for v in line.strip().replace(",", " ").split() if v]
                if len(parts) >= 4:
                    x, y, w, h = parts[:4]
                    gt_boxes.append((x + w / 2.0, y + h / 2.0, w, h))

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame")

    init_box = gt_boxes[0] if gt_boxes else (width / 2.0, height / 2.0, 100.0, 50.0)

    sim = ClosedLoopSimulation(
        video_path=video_path,
        groundtruth_path=gt_path,
        standoff_dist=standoff_dist,
        standoff_alt=standoff_alt,
        dt=dt,
        meters_per_pixel=m_per_px
    )
    sim.reset(init_box, initial_frame=first_frame)

    # Use mp4v or avc1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    print(f"Generating locked video: {output_path} ({total_frames} frames @ {fps} fps)...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gt = gt_boxes[frame_idx] if frame_idx < len(gt_boxes) else None
        spray_active = simulate_spray and (18 <= frame_idx <= 28)

        det_box = gt if not spray_active else None
        det_conf = 0.92 if not spray_active else 0.0

        hud_frame, telem = sim.process_frame(
            frame,
            detector_measurement=det_box,
            detector_conf=det_conf,
            simulate_spray_dropout=spray_active
        )

        out.write(hud_frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Successfully generated {frame_idx} frames -> {output_path}")


if __name__ == "__main__":
    # Generate MVTD real boat locked video
    generate_video(
        video_path="data/sample_videos/real_maritime_boat1.mp4",
        gt_path="data/sample_videos/mvtd_boat1/groundtruth.txt",
        output_path="data/outputs/real_maritime_boat1_locked.mp4",
        m_per_px=0.20,
        simulate_spray=True
    )
    # Generate Speedboat maneuver locked video
    generate_video(
        video_path="data/sample_videos/speedboat_maneuver_spray.mp4",
        gt_path="data/sample_videos/speedboat_maneuver_spray_gt.txt",
        output_path="data/outputs/speedboat_maneuver_locked.mp4",
        m_per_px=0.15,
        simulate_spray=True
    )
