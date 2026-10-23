import time
import cv2
import numpy as np
from typing import List, Dict, Any, Optional


class MultiCameraManager:
    """
    Multi-Camera CCTV Control Room Coordinator.
    Manages concurrent surveillance streams, synchronizes frame ingestion,
    evaluates multi-stream anomaly scores, and renders a unified 2x2 grid
    with automated focus-highlighting for anomalous feeds.
    """

    def __init__(
        self,
        camera_configs: Optional[List[Dict[str, Any]]] = None,
        grid_size: tuple = (2, 2),
        cell_resolution: tuple = (480, 270),
    ):
        self.grid_size = grid_size  # (rows, cols)
        self.cell_resolution = cell_resolution  # (width, height)

        if camera_configs is None:
            self.camera_configs = [
                {"id": "CAM_01", "name": "Main Entrance", "source": "0"},
                {"id": "CAM_02", "name": "Parking Lot", "source": "synthetic_normal"},
                {"id": "CAM_03", "name": "ATM Kiosk", "source": "synthetic_abnormal"},
                {"id": "CAM_04", "name": "Corridor B", "source": "synthetic_normal"},
            ]
        else:
            self.camera_configs = camera_configs

        self.num_cameras = len(self.camera_configs)
        self.scores: Dict[str, float] = {cfg["id"]: 0.05 for cfg in self.camera_configs}
        self.last_incident_cam: Optional[str] = None

    def update_score(self, camera_id: str, score: float):
        """Updates the most recent anomaly score for a specific camera feed."""
        if camera_id in self.scores:
            self.scores[camera_id] = float(score)

    def render_grid(
        self,
        frames_dict: Dict[str, np.ndarray],
        threshold: float = 0.55,
    ) -> np.ndarray:
        """
        Combines individual camera frames into a unified 2x2 multi-viewport CCTV matrix.
        Highlights anomalous cameras with red borders and incident warnings.
        """
        cell_w, cell_h = self.cell_resolution
        rows, cols = self.grid_size
        canvas_w = cell_w * cols
        canvas_h = cell_h * rows
        grid_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for idx, cfg in enumerate(self.camera_configs):
            cam_id = cfg["id"]
            cam_name = cfg["name"]
            score = self.scores.get(cam_id, 0.0)
            is_anomaly = score >= threshold

            # Get frame or create synthetic stand-in
            if cam_id in frames_dict and frames_dict[cam_id] is not None:
                cell_frame = cv2.resize(frames_dict[cam_id], (cell_w, cell_h))
            else:
                cell_frame = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                cv2.putText(
                    cell_frame,
                    f"NO SIGNAL: {cam_name}",
                    (30, cell_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (100, 100, 100),
                    2,
                )

            # Draw camera-specific overlay
            # Header tag
            header_color = (0, 0, 200) if is_anomaly else (30, 30, 30)
            cv2.rectangle(cell_frame, (0, 0), (cell_w, 28), header_color, -1)
            status_text = f"🚨 ALERT ({score:.2f})" if is_anomaly else f"p={score:.2f}"
            cv2.putText(
                cell_frame,
                f"[{cam_id}] {cam_name} | {status_text}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Border
            border_color = (0, 0, 255) if is_anomaly else (60, 60, 60)
            border_thickness = 4 if is_anomaly else 1
            cv2.rectangle(cell_frame, (0, 0), (cell_w - 1, cell_h - 1), border_color, border_thickness)

            # Place into canvas
            r = idx // cols
            c = idx % cols
            y_start = r * cell_h
            x_start = c * cell_w
            grid_canvas[y_start : y_start + cell_h, x_start : x_start + cell_w] = cell_frame

        return grid_canvas
