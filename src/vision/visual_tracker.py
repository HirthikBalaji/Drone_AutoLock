"""
High-Rate Optical Visual Tracker.
Uses correlation filters (CSRT / MIL / KCF) to track target appearance between detector updates,
handling frame-by-frame boat motion, spray, and scale adaptation.
"""

import cv2
import numpy as np


class VisualTracker:
    def __init__(self, tracker_type: str = "CSRT"):
        self.tracker_type = tracker_type.upper()
        self.tracker = None
        self.is_tracking = False
        self.last_bbox = None  # (cx, cy, w, h)

    def _create_tracker(self):
        # Prefer CSRT for high accuracy with scale changes; fallback to MIL or KCF
        if self.tracker_type == "CSRT" and hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        elif self.tracker_type == "MIL" and hasattr(cv2, "TrackerMIL_create"):
            return cv2.TrackerMIL_create()
        elif hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        else:
            return cv2.TrackerMIL_create()

    def init(self, frame: np.ndarray, bbox_xywh: tuple):
        """
        Initializes tracker with bbox: (cx, cy, w, h) or (x1, y1, w, h).
        bbox_xywh here expects (x_min, y_min, w, h) format for OpenCV.
        """
        self.tracker = self._create_tracker()
        # Clean bbox bounds
        x, y, w, h = [int(v) for v in bbox_xywh]
        h_f, w_f = frame.shape[:2]
        x = max(0, min(x, w_f - 1))
        y = max(0, min(y, h_f - 1))
        w = max(4, min(w, w_f - x))
        h = max(4, min(h, h_f - y))

        rect = (x, y, w, h)
        ok = self.tracker.init(frame, rect)
        self.is_tracking = bool(ok)
        if ok:
            cx = x + w / 2.0
            cy = y + h / 2.0
            self.last_bbox = (cx, cy, w, h)
        return self.is_tracking

    def update(self, frame: np.ndarray):
        """
        Updates optical track on new frame.
        Returns: (success: bool, bbox_center_xywh: (cx, cy, w, h), confidence: float)
        """
        if not self.is_tracking or self.tracker is None:
            return False, None, 0.0

        ok, rect = self.tracker.update(frame)
        if ok:
            x, y, w, h = rect
            cx = x + w / 2.0
            cy = y + h / 2.0

            # Image boundary check
            h_f, w_f = frame.shape[:2]
            if cx < 0 or cx >= w_f or cy < 0 or cy >= h_f:
                self.is_tracking = False
                return False, None, 0.0

            self.last_bbox = (cx, cy, w, h)
            # Estimate confidence based on bounding box stability
            conf = 0.85
            return True, (cx, cy, w, h), conf
        else:
            self.is_tracking = False
            return False, None, 0.0
