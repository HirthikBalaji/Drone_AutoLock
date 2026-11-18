"""
Maritime Benchmark Evaluation Metrics.
Calculates Center Location Error (CLE), IoU Success Rate, Lock Retention,
and Guidance Standoff Tracking Performance.
"""

import numpy as np


def compute_iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) between two boxes.
    Boxes can be (cx, cy, w, h) or (x1, y1, x2, y2).
    """
    if boxA is None or boxB is None:
        return 0.0

    # Convert cx, cy, w, h to x1, y1, x2, y2 if needed
    def to_corners(b):
        cx, cy, w, h = b
        return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]

    xA1, yA1, xA2, yA2 = to_corners(boxA)
    xB1, yB1, xB2, yB2 = to_corners(boxB)

    # Intersection rectangle
    xi1 = max(xA1, xB1)
    yi1 = max(yA1, yB1)
    xi2 = min(xA2, xB2)
    yi2 = min(yA2, yB2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    boxA_area = (xA2 - xA1) * (yA2 - yA1)
    boxB_area = (xB2 - xB1) * (yB2 - yB1)
    union_area = boxA_area + boxB_area - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def compute_cle(pred_center, gt_center):
    """
    Computes Center Location Error (pixels or meters).
    """
    if pred_center is None or gt_center is None:
        return float("nan")
    return float(np.hypot(pred_center[0] - gt_center[0], pred_center[1] - gt_center[1]))


class MaritimeBenchmark:
    def __init__(self, iou_thresh: float = 0.5, cle_thresh: float = 20.0):
        self.iou_thresh = iou_thresh
        self.cle_thresh = cle_thresh
        self.ious = []
        self.cles = []
        self.lock_statuses = []
        self.standoff_errors = []
        self.speed_errors = []

    def record_frame(self, pred_bbox, gt_bbox, status: str, standoff_err=None, speed_err=None):
        """
        pred_bbox, gt_bbox: (cx, cy, w, h)
        """
        if pred_bbox is not None and gt_bbox is not None:
            iou = compute_iou(pred_bbox, gt_bbox)
            cle = compute_cle((pred_bbox[0], pred_bbox[1]), (gt_bbox[0], gt_bbox[1]))
            self.ious.append(iou)
            self.cles.append(cle)
        else:
            self.ious.append(0.0)
            self.cles.append(float("nan"))

        self.lock_statuses.append(status)
        if standoff_err is not None:
            self.standoff_errors.append(standoff_err)
        if speed_err is not None:
            self.speed_errors.append(speed_err)

    def summary(self):
        total_frames = len(self.lock_statuses)
        if total_frames == 0:
            return {}

        valid_cles = [c for c in self.cles if not np.isnan(c)]
        mean_cle = float(np.mean(valid_cles)) if valid_cles else 0.0
        median_cle = float(np.median(valid_cles)) if valid_cles else 0.0

        mean_iou = float(np.mean(self.ious)) if self.ious else 0.0
        success_rate = float(np.mean([1.0 if i >= self.iou_thresh else 0.0 for i in self.ious]))
        precision_rate = float(np.mean([1.0 if c <= self.cle_thresh else 0.0 for c in valid_cles])) if valid_cles else 0.0

        locked_or_coasting = [s for s in self.lock_statuses if s in ["LOCKED", "COASTING"]]
        lock_retention_rate = float(len(locked_or_coasting) / total_frames)

        mean_standoff_err = float(np.mean(self.standoff_errors)) if self.standoff_errors else 0.0
        mean_speed_err = float(np.mean(self.speed_errors)) if self.speed_errors else 0.0

        return {
            "total_frames": total_frames,
            "mean_iou": round(mean_iou, 3),
            "success_rate_iou50": round(success_rate * 100.0, 1),
            "precision_rate_cle20": round(precision_rate * 100.0, 1),
            "mean_cle_px": round(mean_cle, 2),
            "median_cle_px": round(median_cle, 2),
            "lock_retention_pct": round(lock_retention_rate * 100.0, 1),
            "mean_standoff_error_m": round(mean_standoff_err, 2),
            "mean_speed_error_ms": round(mean_speed_err, 2),
        }
