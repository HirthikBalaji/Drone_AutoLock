"""
Generates synthetic high-speed maritime scenario with wave texture,
dynamic boat S-turns, turbulent wake, and intermittent sun glare/spray occlusion.
"""

import os
import cv2
import numpy as np


def generate_speedboat_scenario(
    output_path: str = "data/sample_videos/speedboat_maneuver_spray.mp4",
    num_frames: int = 100,
    width: int = 1280,
    height: int = 720,
    fps: int = 20
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    gt_path = output_path.replace(".mp4", "_gt.txt")
    gt_lines = []

    # Boat trajectory parameters (S-turn carving through water)
    duration_sec = num_frames / float(fps)
    t_vals = np.linspace(0, duration_sec, num_frames)
    boat_x = 180 + 85 * t_vals
    boat_y = height / 2.0 + 130 * np.sin(1.2 * t_vals)

    wake_history = []

    for i in range(num_frames):
        # Ocean base background with wave ripples
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Deep blue ocean gradient
        frame[:, :] = (120, 60, 20)  # BGR

        # Wave ripples
        y_grid, x_grid = np.ogrid[:height, :width]
        wave_noise = (np.sin(x_grid * 0.03 + y_grid * 0.05 + i * 0.2) * 15).astype(np.int16)
        frame[:, :, 0] = np.clip(frame[:, :, 0].astype(np.int16) + wave_noise, 0, 255).astype(np.uint8)
        frame[:, :, 1] = np.clip(frame[:, :, 1].astype(np.int16) + wave_noise // 2, 0, 255).astype(np.uint8)

        # Current boat position and heading
        bx = boat_x[i]
        by = boat_y[i]
        wake_history.append((bx, by))

        # Heading tangent
        if i < num_frames - 1:
            dx = boat_x[i + 1] - boat_x[i]
            dy = boat_y[i + 1] - boat_y[i]
        else:
            dx = boat_x[i] - boat_x[i - 1]
            dy = boat_y[i] - boat_y[i - 1]
        heading = np.arctan2(dy, dx)

        # Draw white-water wake trailing behind
        for w_idx in range(max(0, len(wake_history) - 35), len(wake_history) - 1):
            wx, wy = wake_history[w_idx]
            age = len(wake_history) - w_idx
            wake_radius = int(2 + age * 0.6)
            alpha = max(0.1, 1.0 - age / 35.0)
            wake_color = (int(240 * alpha + 120 * (1 - alpha)),
                          int(240 * alpha + 60 * (1 - alpha)),
                          int(220 * alpha + 20 * (1 - alpha)))
            cv2.circle(frame, (int(wx), int(wy)), wake_radius, wake_color, -1)

        # Draw Boat (sleek speedboat hull)
        boat_len = 54
        boat_width = 22
        hull_pts = np.array([
            [boat_len / 2.0, 0],              # Bow (front tip)
            [boat_len / 4.0, boat_width / 2.0],
            [-boat_len / 2.0, boat_width / 2.2], # Stern port
            [-boat_len / 2.0, -boat_width / 2.2], # Stern starboard
            [boat_len / 4.0, -boat_width / 2.0]
        ], dtype=np.float32)

        # Rotate hull by heading
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        rot_hull = (hull_pts @ R.T) + np.array([bx, by])
        cv2.fillPoly(frame, [rot_hull.astype(np.int32)], (240, 240, 245))
        cv2.polylines(frame, [rot_hull.astype(np.int32)], True, (30, 30, 40), 2)

        # Cabin / Windshield
        cabin_pts = np.array([
            [boat_len * 0.1, 0],
            [-boat_len * 0.2, boat_width * 0.3],
            [-boat_len * 0.2, -boat_width * 0.3]
        ], dtype=np.float32)
        rot_cabin = (cabin_pts @ R.T) + np.array([bx, by])
        cv2.fillPoly(frame, [rot_cabin.astype(np.int32)], (180, 50, 30))

        # Ground truth bounding box (x, y, w, h)
        x_min = np.min(rot_hull[:, 0])
        y_min = np.min(rot_hull[:, 1])
        x_max = np.max(rot_hull[:, 0])
        y_max = np.max(rot_hull[:, 1])
        bw = x_max - x_min
        bh = y_max - y_min
        gt_lines.append(f"{x_min:.1f},{y_min:.1f},{bw:.1f},{bh:.1f}\n")

        # Simulate harsh sun glare & water spray burst between frames 30 and 42
        if 30 <= i <= 42:
            glare_center = (int(bx + np.random.uniform(-10, 10)), int(by + np.random.uniform(-10, 10)))
            glare_radius = int(60 + 20 * np.sin((i - 30) / 12.0 * np.pi))
            glare_mask = np.zeros((height, width), dtype=np.float32)
            cv2.circle(glare_mask, glare_center, glare_radius, 1.0, -1)
            glare_mask = cv2.GaussianBlur(glare_mask, (61, 61), 25)
            # Add intense white washout
            for c in range(3):
                frame[:, :, c] = np.clip(frame[:, :, c] * (1.0 - glare_mask * 0.85) + 255 * (glare_mask * 0.85), 0, 255).astype(np.uint8)

        out.write(frame)

    out.release()
    with open(gt_path, "w") as f:
        f.writelines(gt_lines)
    print(f"Generated {output_path} with {num_frames} frames.")


if __name__ == "__main__":
    generate_speedboat_scenario()
