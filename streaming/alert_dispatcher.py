import os
import time
import json
import cv2
import numpy as np
from typing import Optional, Dict, Any, List


class AlertDispatcher:
    """
    Production-grade security incident alert dispatcher.
    Features:
      - Cooldown throttling to prevent alarm flooding.
      - High-resolution visual snapshot capture with HUD watermarking.
      - Optional local audio siren buzzer.
      - Optional Telegram Bot instant photo & telemetry alert notification.
      - Structured JSON security audit logging for forensic analysis.
    """

    def __init__(
        self,
        alert_threshold: float = 0.55,
        cooldown_seconds: float = 5.0,
        snapshot_dir: str = "logs/incident_snapshots/",
        audit_log_path: str = "logs/security_audit_log.json",
        enable_audio_alarm: bool = False,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ):
        self.alert_threshold = alert_threshold
        self.cooldown_seconds = cooldown_seconds
        self.snapshot_dir = snapshot_dir
        self.audit_log_path = audit_log_path
        self.enable_audio_alarm = enable_audio_alarm
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id

        self.last_alert_time = 0.0
        self.incident_count = 0
        self.incident_history: List[Dict[str, Any]] = []

        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)

    def process_score(
        self,
        score: float,
        current_frame: np.ndarray,
        camera_id: str = "CAM_01",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Evaluates current anomaly score against threshold.
        If threshold breached and cooldown elapsed, dispatches multi-channel alerts.
        """
        current_time = time.time()
        if score >= self.alert_threshold:
            if current_time - self.last_alert_time >= self.cooldown_seconds:
                self.last_alert_time = current_time
                self.incident_count += 1
                self._dispatch_alert(score, current_frame, camera_id, metadata)
                return True
        return False

    def _dispatch_alert(
        self,
        score: float,
        frame: np.ndarray,
        camera_id: str,
        metadata: Optional[Dict[str, Any]],
    ):
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        readable_time = time.strftime("%Y-%m-%d %H:%M:%S")
        filename = f"incident_{camera_id}_{timestamp_str}_score_{score:.2f}.jpg"
        filepath = os.path.join(self.snapshot_dir, filename)

        # 1. Annotate frame with high-contrast alert banner
        annotated = frame.copy()
        h, w, _ = annotated.shape
        cv2.rectangle(annotated, (0, 0), (w, 55), (0, 0, 220), -1)
        alert_text = f"CRITICAL ANOMALY ALERT! Camera: {camera_id} | Score: {score:.2f} ({score*100:.1f}%)"
        cv2.putText(
            annotated,
            alert_text,
            (15, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Save snapshot
        cv2.imwrite(filepath, annotated)

        # 2. Build incident record
        record = {
            "incident_id": self.incident_count,
            "camera_id": camera_id,
            "timestamp": readable_time,
            "anomaly_score": round(float(score), 4),
            "threshold": self.alert_threshold,
            "snapshot_path": filepath,
            "metadata": metadata or {},
        }
        self.incident_history.append(record)
        self._append_to_audit_log(record)

        print(
            f"\n🚨 [SECURITY ALERT #{self.incident_count}] Camera: {camera_id} | "
            f"Score: {score:.4f} > {self.alert_threshold:.2f} | Snapshot: {filepath}"
        )

        # 3. Audio alarm (optional)
        if self.enable_audio_alarm:
            self._trigger_audio_siren()

        # 4. Telegram alert (optional)
        if self.telegram_bot_token and self.telegram_chat_id:
            self._send_telegram_alert(record, filepath)

    def _trigger_audio_siren(self):
        """Emits an audible security warning sound."""
        try:
            import winsound
            winsound.Beep(1200, 350)  # 1200 Hz tone for 350 ms
        except Exception:
            # Fallback for non-Windows platforms
            print("\a", end="", flush=True)

    def _send_telegram_alert(self, record: Dict[str, Any], photo_path: str):
        """Sends instant security incident photo with telemetry to Telegram Bot."""
        try:
            import requests

            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendPhoto"
            caption = (
                f"🚨 <b>CRITICAL SURVEILLANCE ALERT</b>\n"
                f"📷 <b>Camera:</b> {record['camera_id']}\n"
                f"⏱️ <b>Time:</b> {record['timestamp']}\n"
                f"📊 <b>Anomaly Score:</b> {record['anomaly_score']:.4f}\n"
                f"⚠️ <b>Action Required:</b> Immediate Security Verification."
            )
            with open(photo_path, "rb") as photo:
                payload = {"chat_id": self.telegram_chat_id, "caption": caption, "parse_mode": "HTML"}
                files = {"photo": photo}
                requests.post(url, data=payload, files=files, timeout=5.0)
                print(f"[✓] Telegram alert successfully sent to chat {self.telegram_chat_id}")
        except Exception as e:
            print(f"[!] Warning: Failed to send Telegram alert: {e}")

    def _append_to_audit_log(self, record: Dict[str, Any]):
        """Appends incident record to JSON audit log."""
        try:
            logs = []
            if os.path.exists(self.audit_log_path):
                with open(self.audit_log_path, "r", encoding="utf-8") as f:
                    try:
                        logs = json.load(f)
                    except json.JSONDecodeError:
                        logs = []
            logs.append(record)
            with open(self.audit_log_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"[!] Warning: Could not write to audit log: {e}")
