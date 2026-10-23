import threading
import time
import cv2
import numpy as np
from typing import Optional, Tuple


class ThreadedStreamCapture:
    """
    High-performance asynchronous threaded video capture worker.
    Continuously drains the underlying camera buffer in a background thread
    to prevent OpenCV frame lag and latency accumulation in RTSP/Webcam streams.
    """

    def __init__(self, source: str = "0", reconnect_interval: float = 3.0):
        # If integer string like "0", parse as device index
        self.source = int(source) if str(source).isdigit() else source
        self.reconnect_interval = reconnect_interval

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.ret: bool = False
        self.is_running: bool = False
        self.lock = threading.Lock()
        self.fps: float = 30.0
        self.width: int = 640
        self.height: int = 480
        self.frame_count: int = 0

        self._connect()

    def _connect(self) -> bool:
        """Initializes or reconnects to the video stream."""
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"[!] Warning: Unable to open video source: {self.source}")
            return False

        # Query hardware stream properties
        reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = reported_fps if reported_fps > 0 else 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        print(f"[✓] StreamCapture opened: {self.source} ({self.width}x{self.height} @ {self.fps:.1f} FPS)")
        return True

    def start(self) -> "ThreadedStreamCapture":
        """Starts the background capture thread."""
        if self.is_running:
            return self

        self.is_running = True
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()
        return self

    def _capture_worker(self):
        """Worker loop that continuously reads latest frame."""
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(self.reconnect_interval)
                self._connect()
                continue

            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = True
                    self.frame = frame
                    self.frame_count += 1
            else:
                # Video file reached end or stream dropped
                with self.lock:
                    self.ret = False
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the most recent frame captured by the background thread."""
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        """Stops the capture worker and releases hardware resources."""
        self.is_running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("[✓] StreamCapture stopped successfully.")

