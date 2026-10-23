import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class SpatioTemporalLocalizer:
    """
    Spatio-Temporal Anomaly Localization Module.
    Fuses Weakly Supervised Video-Level Temporal Anomaly Scores with
    Spatial Motion Saliency and Object Region Proposals to localize
    abnormal events both in TIME and in SPACE.
    """

    def __init__(
        self,
        min_contour_area: int = 600,
        max_boxes: int = 5,
        saliency_threshold: float = 30.0,
    ):
        self.min_contour_area = min_contour_area
        self.max_boxes = max_boxes
        self.saliency_threshold = saliency_threshold
        self.prev_gray: Optional[np.ndarray] = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=25, detectShadows=False
        )

    def extract_salient_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Extracts candidate spatial regions (x, y, w, h, motion_energy) using
        dynamic spatial motion energy and background subtraction.
        """
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # 1. Background subtraction mask
        fg_mask = self.bg_subtractor.apply(gray)
        _, fg_thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_cleaned = cv2.morphologyEx(fg_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 2. Extract contour bounding boxes
        contours, _ = cv2.findContours(fg_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_contour_area:
                x, y, box_w, box_h = cv2.boundingRect(cnt)
                # Filter out extreme aspect ratios (full screen noise)
                if box_w > w * 0.9 and box_h > h * 0.9:
                    continue
                roi = fg_cleaned[y : y + box_h, x : x + box_w]
                motion_energy = float(np.mean(roi))
                candidates.append((x, y, box_w, box_h, motion_energy))

        # Sort candidates by motion energy in descending order
        candidates.sort(key=lambda item: item[4], reverse=True)
        self.prev_gray = gray
        return candidates[: self.max_boxes]

    def render_spatiotemporal_hud(
        self,
        frame: np.ndarray,
        anomaly_score: float,
        threshold: float,
        fps: float = 30.0,
        model_name: str = "MultiModel_Ensemble",
    ) -> np.ndarray:
        """
        Renders complete real-time heads-up display overlay on video frame:
          - Spatial bounding boxes around anomalous or active regions.
          - Top surveillance alert status bar.
          - Bottom telemetry banner (score gauge, FPS, model name).
        """
        annotated = frame.copy()
        h, w, _ = annotated.shape
        is_anomaly = anomaly_score >= threshold

        # 1. Extract dynamic spatial candidate regions
        boxes = self.extract_salient_regions(frame)

        # 2. Draw spatial bounding boxes
        if is_anomaly:
            # Highlight anomalous regions with high-contrast glowing red boxes
            for i, (x, y, box_w, box_h, energy) in enumerate(boxes):
                color = (0, 0, 230)  # Bright Red
                thickness = 2
                cv2.rectangle(annotated, (x, y), (x + box_w, y + box_h), color, thickness)

                # Corner brackets for futuristic tactical CCTV look
                corner_len = min(20, box_w // 4, box_h // 4)
                # Top-left
                cv2.line(annotated, (x, y), (x + corner_len, y), (0, 0, 255), 3)
                cv2.line(annotated, (x, y), (x, y + corner_len), (0, 0, 255), 3)
                # Bottom-right
                cv2.line(annotated, (x + box_w, y + box_h), (x + box_w - corner_len, y + box_h), (0, 0, 255), 3)
                cv2.line(annotated, (x + box_w, y + box_h), (x + box_w, y + box_h - corner_len), (0, 0, 255), 3)

                # Label tag
                tag = f"🚨 SUSPICIOUS ACTOR [p={anomaly_score:.2f}]"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (x, max(0, y - 20)), (x + tw + 6, max(0, y)), (0, 0, 200), -1)
                cv2.putText(
                    annotated, tag, (x + 3, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
                )
        else:
            # Normal surveillance tracking in calm green/cyan
            for i, (x, y, box_w, box_h, energy) in enumerate(boxes[:2]):
                color = (60, 180, 60)  # Calm Green
                cv2.rectangle(annotated, (x, y), (x + box_w, y + box_h), color, 1)
                tag = f"NORMAL [p={anomaly_score:.2f}]"
                cv2.putText(
                    annotated, tag, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )

        # 3. Top Status Banner
        banner_color = (0, 0, 200) if is_anomaly else (30, 130, 30)
        cv2.rectangle(annotated, (0, 0), (w, 50), banner_color, -1)
        status_text = "🚨 CRITICAL CRIME / ANOMALY DETECTED" if is_anomaly else "🟢 SURVEILLANCE ACTIVE - ALL NORMAL"
        cv2.putText(
            annotated,
            status_text,
            (16, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 4. Bottom Telemetry Bar
        cv2.rectangle(annotated, (0, h - 40), (w, h), (20, 20, 20), -1)
        telemetry = f"Model: {model_name} | FPS: {fps:.1f} | p(t)={anomaly_score:.3f} | τ={threshold:.2f}"
        cv2.putText(
            annotated,
            telemetry,
            (16, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # 5. Live Probability Gauge Bar (right corner of bottom bar)
        gauge_width = 150
        gauge_height = 16
        gx = w - gauge_width - 20
        gy = h - 28
        cv2.rectangle(annotated, (gx, gy), (gx + gauge_width, gy + gauge_height), (60, 60, 60), 1)
        fill_w = int(np.clip(anomaly_score, 0.0, 1.0) * gauge_width)
        bar_color = (0, 0, 220) if is_anomaly else (0, 200, 100)
        cv2.rectangle(annotated, (gx + 1, gy + 1), (gx + fill_w, gy + gauge_height - 1), bar_color, -1)

        # Threshold marker on gauge
        tx = int(gx + threshold * gauge_width)
        cv2.line(annotated, (tx, gy - 2), (tx, gy + gauge_height + 2), (255, 255, 0), 2)

        return annotated
