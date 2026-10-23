import collections
import threading
import torch
import numpy as np
from typing import Optional


class CircularSlidingWindowBuffer:
    """
    Thread-safe circular sliding-window buffer for continuous real-time video surveillance.
    Accumulates video frames and provides formatted 16-frame tensors for 3D feature inference.
    """

    def __init__(self, window_size: int = 16, stride: int = 8, max_capacity: int = 120):
        self.window_size = window_size
        self.stride = stride
        self.buffer = collections.deque(maxlen=max_capacity)
        self.lock = threading.Lock()
        self.frames_since_last_inference = 0

    def add_frame(self, frame: np.ndarray):
        """Adds a newly captured BGR/RGB frame to the circular buffer."""
        with self.lock:
            self.buffer.append(frame)
            self.frames_since_last_inference += 1

    def is_ready_for_inference(self) -> bool:
        """Returns True if the buffer has enough frames and reached the stride interval."""
        with self.lock:
            return (
                len(self.buffer) >= self.window_size
                and self.frames_since_last_inference >= self.stride
            )

    def get_window_tensor(self, target_size=(224, 224)) -> Optional[torch.Tensor]:
        """
        Extracts the latest window_size frames and formats into PyTorch tensor (1, 3, T, H, W).
        """
        import cv2

        with self.lock:
            if len(self.buffer) < self.window_size:
                return None

            # Extract the last window_size frames
            frames = list(self.buffer)[-self.window_size:]
            self.frames_since_last_inference = 0

        # Resize and normalize frames
        processed = []
        for img in frames:
            resized = cv2.resize(img, target_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            # ImageNet mean & std normalization
            normalized = (normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            processed.append(normalized)

        # Shape: (T, H, W, C) -> (C, T, H, W) -> (1, C, T, H, W)
        tensor = np.stack(processed, axis=0)
        tensor = np.transpose(tensor, (3, 0, 1, 2))
        return torch.from_numpy(tensor).unsqueeze(0).float()

