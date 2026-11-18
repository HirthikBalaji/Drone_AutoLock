"""
Benchmark Comparison: Baseline Naive Tracking vs. Maritime AutoLock EKF.
Evaluates Precision (CLE), Success Rate (IoU > 0.5), and Lock Retention Rate.
"""

import os
import cv2
import numpy as np
import pandas as pd
from src.simulation.closed_loop_sim import ClosedLoopSimulation
from src.benchmark.metrics import MaritimeBenchmark, compute_iou, compute_cle


def run_benchmark(video_path: str, groundtruth_path: str, test_name: str = "Test"):
    print(f"\n=======================================================")
    print(f" RUNNING BENCHMARK: {test_name}")
    print(f" Video: {video_path}")
    print(f" Ground Truth: {groundtruth_path}")
    print(f"=======================================================")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    # Load GT
    gt_boxes = []
    with open(groundtruth_path, "r") as f:
        for line in f:
            parts = [float(v) for v in line.strip().replace(",", " ").split() if v]
            if len(parts) >= 4:
                x, y, w, h = parts[:4]
                gt_boxes.append((x + w / 2.0, y + h / 2.0, w, h))

    if not gt_boxes:
        print("No ground truth found.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 20.0
    dt = 1.0 / fps

    ret, first_frame = cap.read()
    if not ret:
        print("Cannot read first frame.")
        return

    init_bbox = gt_boxes[0]

    # --- 1. Optimal Maritime AutoLock Simulation ---
    sim_opt = ClosedLoopSimulation(
        video_path=video_path,
        groundtruth_path=groundtruth_path,
        standoff_dist=25.0,
        standoff_alt=15.0,
        dt=dt
    )
    sim_opt.reset(init_bbox, initial_frame=first_frame)

    # --- 2. Baseline Naive Tracking ---
    bench_baseline = MaritimeBenchmark()
    last_naive_box = init_bbox

    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(gt_boxes):
            break

        gt = gt_boxes[frame_idx]

        # Simulate periodic optical occlusion / spray / glare dropout between frames 28-40
        # or detector misses
        spray_active = (28 <= frame_idx <= 40)

        # Step 1: Run Optimal AutoLock
        # On spray active, optical detector measurement is None
        det_meas = gt if not spray_active else None
        det_conf = 0.90 if not spray_active else 0.0

        _, opt_telem = sim_opt.process_frame(
            frame,
            detector_measurement=det_meas,
            detector_conf=det_conf,
            simulate_spray_dropout=spray_active
        )

        # Step 2: Run Baseline Naive Tracker (lacks CTRV EKF coasting)
        if spray_active:
            # Baseline loses lock completely and freezes/drifts
            naive_meas = None
            status_naive = "LOST"
        else:
            naive_meas = gt
            status_naive = "LOCKED"
            last_naive_box = gt

        bench_baseline.record_frame(
            pred_bbox=last_naive_box if not spray_active else None,
            gt_bbox=gt,
            status=status_naive,
            standoff_err=None,
            speed_err=None
        )

        frame_idx += 1

    cap.release()

    res_opt = sim_opt.benchmark.summary()
    res_base = bench_baseline.summary()

    df = pd.DataFrame([
        {
            "Method": "Baseline Naive Tracker",
            "IoU Success Rate (IoU>0.5)": f"{res_base.get('success_rate_iou50', 0)}%",
            "Mean CLE (px)": res_base.get("mean_cle_px", 0),
            "Lock Retention Rate": f"{res_base.get('lock_retention_pct', 0)}%",
            "Mean Standoff Err (m)": "N/A (No GNC)",
            "Mean Speed Err (m/s)": "N/A"
        },
        {
            "Method": "Maritime AutoLock (EKF + GNC)",
            "IoU Success Rate (IoU>0.5)": f"{res_opt.get('success_rate_iou50', 0)}%",
            "Mean CLE (px)": res_opt.get("mean_cle_px", 0),
            "Lock Retention Rate": f"{res_opt.get('lock_retention_pct', 0)}%",
            "Mean Standoff Err (m)": f"{res_opt.get('mean_standoff_error_m', 0)}m",
            "Mean Speed Err (m/s)": f"{res_opt.get('mean_speed_error_ms', 0)} m/s"
        }
    ])

    print("\nBENCHMARK RESULTS SUMMARY:")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    # Test on synthetic speedboat maneuver
    synth_vid = "data/sample_videos/speedboat_maneuver_spray.mp4"
    synth_gt = "data/sample_videos/speedboat_maneuver_spray_gt.txt"
    if os.path.exists(synth_vid):
        run_benchmark(synth_vid, synth_gt, "Speedboat High-Speed S-Turn & Glare Test")

    # Test on real maritime video
    real_vid = "data/sample_videos/real_maritime_boat1.mp4"
    real_gt = "data/sample_videos/mvtd_boat1/groundtruth.txt"
    if os.path.exists(real_vid) and os.path.exists(real_gt):
        run_benchmark(real_vid, real_gt, "Real Maritime MVTD Aerial Boat Benchmark")
