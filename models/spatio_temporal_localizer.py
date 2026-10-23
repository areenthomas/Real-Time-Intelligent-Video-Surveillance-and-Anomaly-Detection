import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class SpatioTemporalLocalizer:
    """
    Spatio-Temporal Anomaly Localization and Classification Module.
    Fuses Weakly Supervised Video-Level Temporal Anomaly Scores with
    Spatial Motion Saliency, Region Proposals, and Motion Dynamics to determine:
      1. WHEN the anomaly occurs (temporal score p(t)).
      2. WHERE the anomaly occurs (spatial bounding box & security zone).
      3. WHAT the anomaly is (crime category classification & threat severity).
    """

    def __init__(
        self,
        min_contour_area: int = 500,
        max_boxes: int = 4,
    ):
        self.min_contour_area = min_contour_area
        self.max_boxes = max_boxes
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

        # Background subtraction mask
        fg_mask = self.bg_subtractor.apply(gray)
        _, fg_thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_cleaned = cv2.morphologyEx(fg_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_contour_area:
                x, y, box_w, box_h = cv2.boundingRect(cnt)
                # Filter out extreme noise
                if box_w > w * 0.9 and box_h > h * 0.9:
                    continue
                roi = fg_cleaned[y : y + box_h, x : x + box_w]
                motion_energy = float(np.mean(roi))
                candidates.append((x, y, box_w, box_h, motion_energy))

        candidates.sort(key=lambda item: item[4], reverse=True)
        self.prev_gray = gray
        return candidates[: self.max_boxes]

    def infer_crime_type_and_zone(
        self,
        anomaly_score: float,
        boxes: List[Tuple[int, int, int, int, float]],
        frame_shape: Tuple[int, int],
        preset_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Identifies:
          - WHAT IT IS: Crime Category (Robbery, Assault, Burglary, Fighting, Normal)
          - WHERE IT IS: Security Zone (Zone A: Entrance, Zone B: Counter, Zone C: Perimeter)
          - SEVERITY: Critical, High, Moderate, Normal
        """
        h, w = frame_shape[:2]

        if anomaly_score < 0.45:
            return {
                "is_anomaly": False,
                "crime_type": "Normal Activity",
                "severity": "NORMAL (Safe)",
                "zone": "All Zones Secure",
                "action": "Routine Monitoring",
                "primary_box": boxes[0][:4] if boxes else (0, 0, 0, 0),
            }

        # If preset scenario hint is passed (from user selection)
        if preset_hint:
            hint_lower = preset_hint.lower()
            if "robbery" in hint_lower:
                crime_type = "Armed Robbery / Cash Grab"
                zone = "Zone B: Cash Counter / Vault"
            elif "assault" in hint_lower or "fight" in hint_lower:
                crime_type = "Physical Assault / Violent Fight"
                zone = "Zone A: Public Concourse"
            elif "burglary" in hint_lower or "theft" in hint_lower:
                crime_type = "Burglary / Forced Entry"
                zone = "Zone C: ATM / Entrance Kiosk"
            elif "accident" in hint_lower:
                crime_type = "Vehicle Collision / Road Accident"
                zone = "Zone D: Parking / Roadway"
            else:
                crime_type = "Suspicious Behavior / Trespassing"
                zone = "Zone A: Perimeter Walkway"
        else:
            # Automatic heuristic classification based on motion dynamics and spatial position
            if boxes:
                x, y, bw, bh, energy = boxes[0]
                cx = x + bw // 2
                cy = y + bh // 2

                # Zone mapping based on frame geometry
                if cx < w * 0.4:
                    zone = "Zone A: Left Concourse / Entrance"
                elif cx < w * 0.7:
                    zone = "Zone B: Central Cash Counter"
                else:
                    zone = "Zone C: Right Perimeter / Exit"

                # Motion energy heuristics
                if energy > 80.0:
                    crime_type = "Physical Assault / Fighting"
                elif energy > 40.0:
                    crime_type = "Armed Robbery / Theft"
                else:
                    crime_type = "Burglary / Suspicious Loitering"
            else:
                zone = "Zone B: Surveillance Core"
                crime_type = "Unclassified Anomaly"

        # Severity Assessment
        if anomaly_score >= 0.80:
            severity = "CRITICAL (Level 5/5)"
            action = "🚨 DISPATCH IMMEDIATE SECURITY / SOUND ALARM"
        elif anomaly_score >= 0.65:
            severity = "HIGH (Level 4/5)"
            action = "⚠️ Alert Patrol Guards / Review Live Feed"
        else:
            severity = "ELEVATED (Level 3/5)"
            action = "🔍 Flag Incident for Operator Verification"

        return {
            "is_anomaly": True,
            "crime_type": crime_type,
            "severity": severity,
            "zone": zone,
            "action": action,
            "primary_box": boxes[0][:4] if boxes else (w // 4, h // 4, w // 2, h // 2),
        }

    def render_spatiotemporal_hud(
        self,
        frame: np.ndarray,
        anomaly_score: float,
        threshold: float,
        fps: float = 30.0,
        model_name: str = "MultiModel_Ensemble",
        preset_hint: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Renders complete heads-up display overlay on video frame:
          - Spatial bounding boxes around anomalous or active actors.
          - Zone tag and Crime Type classification.
          - Top surveillance alert status bar.
          - Bottom telemetry banner (score gauge, FPS, model name).
        """
        annotated = frame.copy()
        h, w, _ = annotated.shape
        is_anomaly = anomaly_score >= threshold

        # 1. Extract dynamic spatial candidate regions
        boxes = self.extract_salient_regions(frame)

        # 2. Determine "WHAT IT IS" and "WHERE IT IS"
        detection_info = self.infer_crime_type_and_zone(
            anomaly_score=anomaly_score,
            boxes=boxes,
            frame_shape=(h, w),
            preset_hint=preset_hint,
        )

        # 3. Draw spatial bounding boxes
        if is_anomaly:
            for i, (x, y, box_w, box_h, energy) in enumerate(boxes):
                color = (0, 0, 235)  # Bright Tactical Red
                cv2.rectangle(annotated, (x, y), (x + box_w, y + box_h), color, 2)

                # High-tech corner brackets
                corner_len = min(22, box_w // 4, box_h // 4)
                # Top-left
                cv2.line(annotated, (x, y), (x + corner_len, y), (0, 0, 255), 3)
                cv2.line(annotated, (x, y), (x, y + corner_len), (0, 0, 255), 3)
                # Bottom-right
                cv2.line(annotated, (x + box_w, y + box_h), (x + box_w - corner_len, y + box_h), (0, 0, 255), 3)
                cv2.line(annotated, (x + box_w, y + box_h), (x + box_w, y + box_h - corner_len), (0, 0, 255), 3)

                # Tactical label with WHAT and WHERE
                tag = f"🚨 {detection_info['crime_type']} [{anomaly_score*100:.0f}%]"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (x, max(0, y - 22)), (x + tw + 8, max(0, y)), (0, 0, 210), -1)
                cv2.putText(
                    annotated, tag, (x + 4, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
                )

                # Zone indicator beneath box
                zone_tag = f"📍 {detection_info['zone'].split(':')[0]}"
                cv2.putText(
                    annotated,
                    zone_tag,
                    (x + 2, min(h - 5, y + box_h + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (0, 220, 255),
                    1,
                )
        else:
            # Normal surveillance tracking
            for i, (x, y, box_w, box_h, energy) in enumerate(boxes[:2]):
                color = (60, 190, 60)  # Calm Green
                cv2.rectangle(annotated, (x, y), (x + box_w, y + box_h), color, 1)
                tag = f"🟢 NORMAL [{anomaly_score*100:.0f}%]"
                cv2.putText(
                    annotated, tag, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1
                )

        # 4. Top Status Banner
        banner_color = (0, 0, 210) if is_anomaly else (30, 130, 30)
        cv2.rectangle(annotated, (0, 0), (w, 52), banner_color, -1)
        if is_anomaly:
            status_text = f"🚨 ALERT: {detection_info['crime_type'].upper()} DETECTED"
        else:
            status_text = "🟢 CCTV SURVEILLANCE ACTIVE - ALL NORMAL"

        cv2.putText(
            annotated,
            status_text,
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 5. Bottom Telemetry Bar
        cv2.rectangle(annotated, (0, h - 38), (w, h), (18, 18, 18), -1)
        telemetry = f"Model: {model_name} | FPS: {fps:.1f} | Score: {anomaly_score*100:.1f}% | Zone: {detection_info['zone'].split(':')[0]}"
        cv2.putText(
            annotated,
            telemetry,
            (14, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        # 6. Live Anomaly Gauge Bar
        gauge_width = 140
        gauge_height = 14
        gx = w - gauge_width - 16
        gy = h - 26
        cv2.rectangle(annotated, (gx, gy), (gx + gauge_width, gy + gauge_height), (60, 60, 60), 1)
        fill_w = int(np.clip(anomaly_score, 0.0, 1.0) * gauge_width)
        bar_color = (0, 0, 220) if is_anomaly else (0, 200, 100)
        cv2.rectangle(annotated, (gx + 1, gy + 1), (gx + fill_w, gy + gauge_height - 1), bar_color, -1)

        # Threshold marker
        tx = int(gx + threshold * gauge_width)
        cv2.line(annotated, (tx, gy - 2), (tx, gy + gauge_height + 2), (255, 255, 0), 2)

        return annotated, detection_info
