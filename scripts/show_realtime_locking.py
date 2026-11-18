"""
Interactive Real-Time Video Player for Maritime Drone AutoLock.
Shows smooth real-time video locking with tactical HUD reticle, CTRV prediction,
and anti-spray coasting demonstration.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from src.simulation.closed_loop_sim import ClosedLoopSimulation


def run_realtime_player(video_choice: str = "real"):
    if video_choice == "real":
        v_path = os.path.join(PROJECT_ROOT, "data", "sample_videos", "real_maritime_boat1.mp4")
        g_path = os.path.join(PROJECT_ROOT, "data", "sample_videos", "mvtd_boat1", "groundtruth.txt")
        m_scale = 0.20
        title = "Maritime Drone AutoLock - Real Sea Yacht (MVTD Dataset)"
    else:
        v_path = os.path.join(PROJECT_ROOT, "data", "sample_videos", "speedboat_maneuver_spray.mp4")
        g_path = os.path.join(PROJECT_ROOT, "data", "sample_videos", "speedboat_maneuver_spray_gt.txt")
        m_scale = 0.15
        title = "Maritime Drone AutoLock - Speedboat High-G S-Curve Maneuvers"

    if not os.path.exists(v_path):
        print(f"Error: Video file not found: {v_path}")
        return

    # Load ground truth if exists
    gt_boxes = []
    if os.path.exists(g_path):
        with open(g_path, "r") as f:
            for line in f:
                parts = [float(v) for v in line.strip().replace(",", " ").split() if v]
                if len(parts) >= 4:
                    x, y, w, h = parts[:4]
                    gt_boxes.append((x + w / 2.0, y + h / 2.0, w, h))

    cap = cv2.VideoCapture(v_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_delay = 1.0 / fps

    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    init_box = gt_boxes[0] if gt_boxes else (first_frame.shape[1] / 2, first_frame.shape[0] / 2, 80, 40)

    sim = ClosedLoopSimulation(
        video_path=v_path,
        groundtruth_path=g_path,
        standoff_dist=30.0,
        standoff_alt=18.0,
        dt=frame_delay,
        meters_per_pixel=m_scale
    )
    sim.reset(init_box, initial_frame=first_frame)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1280, 720)

    is_paused = False
    simulate_spray = False
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("\n=======================================================")
    print(f"  {title}")
    print("=======================================================")
    print(" Controls:")
    print("   [Space] : Pause / Resume Playback")
    print("   [S]     : Toggle Wave Spray / Glare Occlusion (Coasting Test)")
    print("   [R]     : Restart Video from Frame 0")
    print("   [1]     : Switch to Real Sea MVTD Yacht Footage")
    print("   [2]     : Switch to Speedboat Spray Maneuver Footage")
    print("   [Q/Esc] : Quit")
    print("=======================================================\n")

    pulse_timer = 0

    while True:
        t_start = time.time()

        if not is_paused:
            ok, frame = cap.read()
            if not ok:
                # Loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                sim.reset(init_box, initial_frame=first_frame)
                continue

            gt = gt_boxes[frame_idx] if frame_idx < len(gt_boxes) else None

            # Check spray occlusion: either forced by user or auto-injected at turn
            spray_active = simulate_spray or (20 <= frame_idx <= 32 and video_choice == "real")

            det_box = gt if not spray_active else None
            det_conf = 0.94 if not spray_active else 0.0

            hud_frame, telem = sim.process_frame(
                frame,
                detector_measurement=det_box,
                detector_conf=det_conf,
                simulate_spray_dropout=spray_active
            )

            # Add pulsing lock animation & instruction footer
            pulse_timer = (pulse_timer + 1) % 20
            footer_bg = hud_frame.copy()
            h, w = hud_frame.shape[:2]
            cv2.rectangle(footer_bg, (0, h - 45), (w, h), (15, 18, 22), -1)
            cv2.addWeighted(footer_bg, 0.8, hud_frame, 0.2, 0, hud_frame)

            msg = f"[SPACE]: Pause  |  [S]: Force Spray Occlusion ({'ON' if spray_active else 'OFF'})  |  [R]: Reset  |  [1/2]: Switch Source  |  [Q]: Exit"
            cv2.putText(hud_frame, msg, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 240), 1)

            cv2.imshow(title, hud_frame)
            frame_idx += 1

        # Keyboard handling
        key = cv2.waitKey(max(1, int(frame_delay * 1000) if not is_paused else 50)) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            break
        elif key == ord(' '):
            is_paused = not is_paused
        elif key in [ord('s'), ord('S')]:
            simulate_spray = not simulate_spray
            print(f">> Simulated Wave Spray / Glare Occlusion: {'ENABLED' if simulate_spray else 'DISABLED'}")
        elif key in [ord('r'), ord('R')]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            sim.reset(init_box, initial_frame=first_frame)
            print(">> Playback restarted.")
        elif key == ord('1'):
            cv2.destroyAllWindows()
            run_realtime_player(video_choice="real")
            return
        elif key == ord('2'):
            cv2.destroyAllWindows()
            run_realtime_player(video_choice="speedboat")
            return

        # Frame rate sync
        elapsed = time.time() - t_start
        if not is_paused and elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else "real"
    run_realtime_player(choice)
