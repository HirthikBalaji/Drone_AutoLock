"""
YOLO Maritime Vessel Detector.
Detects boats, speedboats, and yachts in aerial maritime footage.
Supports high-resolution full-frame and tiled ROI inference for small target recovery.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class MaritimeDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.20):
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh
        # COCO class 8 is 'boat'
        self.target_class_ids = [8]

    def detect(self, frame: np.ndarray, imgsz: int = 1280):
        """
        Runs object detection on frame.
        Returns list of dicts:
        [{'bbox': (cx, cy, w, h), 'xyxy': (x1, y1, x2, y2), 'conf': float, 'class': str}]
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            imgsz=imgsz,
            verbose=False
        )

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # If COCO boat class or high-confidence vehicle on water
                if cls_id in self.target_class_ids or (cls_id in [0, 1, 2, 3, 5, 7] and conf > 0.4):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w / 2.0
                    cy = y1 + h / 2.0
                    detections.append({
                        "bbox": (cx, cy, w, h),
                        "xyxy": (x1, y1, x2, y2),
                        "conf": conf,
                        "class_name": self.model.names.get(cls_id, "vessel")
                    })

        return detections

    def detect_best_match(self, frame: np.ndarray, prior_bbox=None, search_radius=150.0):
        """
        Finds the detection closest to prior predicted location (EKF gating association).
        """
        all_dets = self.detect(frame)
        if not all_dets:
            return None

        if prior_bbox is None:
            # Return detection with highest confidence
            return max(all_dets, key=lambda d: d["conf"])

        pcx, pcy, _, _ = prior_bbox
        best_match = None
        min_dist = float("inf")

        for d in all_dets:
            cx, cy, _, _ = d["bbox"]
            dist = np.hypot(cx - pcx, cy - pcy)
            if dist < search_radius and dist < min_dist:
                min_dist = dist
                best_match = d

        return best_match
