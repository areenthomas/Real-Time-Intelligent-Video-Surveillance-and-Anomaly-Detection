import os
import time
import cv2
import numpy as np


class AlertDispatcher:
    """
    Manages security incident alerts and snapshot logging with cooldown throttling.
    """

    def __init__(
        self,
        alert_threshold: float = 0.55,
        cooldown_seconds: float = 5.0,
        snapshot_dir: str = "logs/incident_snapshots/",
    ):
        self.alert_threshold = alert_threshold
        self.cooldown_seconds = cooldown_seconds
        self.snapshot_dir = snapshot_dir
        self.last_alert_time = 0.0

        os.makedirs(self.snapshot_dir, exist_ok=True)

    def process_score(self, score: float, current_frame: np.ndarray) -> bool:
        """
        Evaluates current anomaly score. If above threshold and cooldown elapsed,
        triggers alert and saves visual snapshot.
        """
        current_time = time.time()
        if score >= self.alert_threshold:
            if current_time - self.last_alert_time >= self.cooldown_seconds:
                self.last_alert_time = current_time
                self._dispatch_alert(score, current_frame)
                return True
        return False

    def _dispatch_alert(self, score: float, frame: np.ndarray):
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"incident_{timestamp_str}_score_{score:.2f}.jpg"
        filepath = os.path.join(self.snapshot_dir, filename)

        # Annotate image with alert banner before saving
        annotated = frame.copy()
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 50), (0, 0, 220), -1)
        text = f"CRITICAL ANOMALY ALERT! Score: {score:.2f} | Time: {time.strftime('%H:%M:%S')}"
        cv2.putText(
            annotated, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        cv2.imwrite(filepath, annotated)
        print(f"\n🚨 [SECURITY ALERT] Incident Detected (Score: {score:.4f})! Snapshot saved: {filepath}")
