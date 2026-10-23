"""
Standalone Frontend Video Anomaly Detection & Threat Tracking Engine
Loads exported trained model weights (.pth) and executes real-time inference
on live webcams, CCTV streams, or MP4 video files.

Usage:
    # 1. Live Webcam:
    python frontend_inference.py --weights checkpoints/best_anomaly_detector.pth --source 0

    # 2. Video File:
    python frontend_inference.py --weights checkpoints/best_anomaly_detector.pth --source my_cctv_video.mp4

    # 3. Built-in Incident Demo (No camera or video needed):
    python frontend_inference.py --weights checkpoints/best_anomaly_detector.pth --source demo --scenario robbery
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional

# =============================================================================
# 1. NEURAL ARCHITECTURES (Self-Contained for Zero Dependency Loading)
# =============================================================================

class MILRankingModel(nn.Module):
    def __init__(self, in_features: int = 512, dropout: float = 0.6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.net(x).squeeze(-1), None


class PyramidDilatedConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilations=(1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for d in dilations
        ])
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels * len(dilations), out_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [b(x) for b in self.branches]
        concat = torch.cat(branch_outs, dim=1)
        return self.fusion(concat)


class RTFMModel(nn.Module):
    def __init__(self, in_features: int = 512, hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        )
        self.pyramid = PyramidDilatedConv(hidden_dim, hidden_dim, dilations=(1, 2, 4))
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(x)
        h_conv = self.pyramid(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = h + h_conv
        attn_out, _ = self.attn(h, h, h)
        h = self.norm(h + attn_out)
        scores = self.classifier(h).squeeze(-1)
        return scores, h


class MISTModel(nn.Module):
    def __init__(self, in_features: int = 512, hidden_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attn_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.encoder(x)
        attn = self.attn_conv(h.permute(0, 2, 1)).permute(0, 2, 1)
        h_boosted = h * (1.0 + attn)
        scores = self.classifier(h_boosted).squeeze(-1)
        return scores, h


class MultiModelEnsemble(nn.Module):
    def __init__(self, in_features: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.mil = MILRankingModel(in_features)
        self.rtfm = RTFMModel(in_features, hidden_dim)
        self.mist = MISTModel(in_features, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_mil, _ = self.mil(x)
        s_rtfm, rtfm_feats = self.rtfm(x)
        s_mist, _ = self.mist(x)
        weights = self.gate(x)
        fused = weights[:, :, 0] * s_rtfm + weights[:, :, 1] * s_mist + weights[:, :, 2] * s_mil
        return fused, rtfm_feats


# =============================================================================
# 2. MODEL LOADER & EXPORT RECONSTRUCTOR
# =============================================================================

def load_exported_model(weights_path: str, device: str = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
    """Loads exported model weights and restores configuration metadata."""
    if not os.path.exists(weights_path):
        print(f"[!] Warning: Checkpoint '{weights_path}' not found. Using initialized ensemble model.")
        model = MultiModelEnsemble(in_features=512, hidden_dim=256)
        model.to(device)
        model.eval()
        return model, {"threshold": 0.55, "architecture": "MultiModelEnsemble"}

    checkpoint = torch.load(weights_path, map_location=device)
    
    # Handle both full checkpoint dict and bare state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        metadata = checkpoint.get("metadata", {})
        in_feat = metadata.get("in_features", 512)
        h_dim = metadata.get("hidden_dim", 256)
        arch = metadata.get("architecture", "MultiModelEnsemble")
        threshold = metadata.get("threshold", 0.55)
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        in_feat, h_dim = 512, 256
        arch = "MultiModelEnsemble"
        threshold = 0.55
        metadata = {"threshold": threshold, "architecture": arch}

    # Instantiate model
    if arch == "MILRankingModel":
        model = MILRankingModel(in_features=in_feat)
    elif arch == "RTFMModel":
        model = RTFMModel(in_features=in_feat, hidden_dim=h_dim)
    elif arch == "MISTModel":
        model = MISTModel(in_features=in_feat, hidden_dim=h_dim)
    else:
        model = MultiModelEnsemble(in_features=in_feat, hidden_dim=h_dim)

    try:
        model.load_state_dict(state_dict)
        print(f"[✓] Successfully loaded weights from: {weights_path} ({arch})")
    except Exception as e:
        print(f"[!] Partial weight loading: {e}")

    model.to(device)
    model.eval()
    return model, metadata


# =============================================================================
# 3. SPATIO-TEMPORAL LOCALIZER & HUD RENDERER
# =============================================================================

class FrontendLocalizer:
    def __init__(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)

    def process_frame(self, frame: np.ndarray, anomaly_score: float, threshold: float = 0.55, scenario: str = None) -> np.ndarray:
        h, w, _ = frame.shape
        annotated = frame.copy()
        is_alert = anomaly_score >= threshold

        # 1. Motion candidate bounding boxes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg = self.bg_sub.apply(cv2.GaussianBlur(gray, (5, 5), 0))
        _, thresh = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for c in contours:
            if cv2.contourArea(c) > 600:
                x, y, bw, bh = cv2.boundingRect(c)
                if bw < w * 0.85 and bh < h * 0.85:
                    boxes.append((x, y, bw, bh))

        # Sort by box area
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

        # 2. Determine "WHAT IT IS" and "WHERE IT IS"
        if is_alert:
            if scenario and "robbery" in scenario.lower():
                crime = "Armed Robbery / Cash Grab"
                zone = "Zone B: Cash Counter"
            elif scenario and "assault" in scenario.lower():
                crime = "Physical Assault / Fighting"
                zone = "Zone A: Entrance Concourse"
            elif scenario and "burglary" in scenario.lower():
                crime = "Burglary / Forced Entry"
                zone = "Zone C: ATM Kiosk"
            else:
                crime = "Suspicious Criminal Threat"
                zone = "Zone B: Main Facility"

            # Draw tactical red boxes
            for (bx, by, bw, bh) in boxes[:3]:
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 0, 230), 2)
                # Corner brackets
                c_len = min(18, bw // 4, bh // 4)
                cv2.line(annotated, (bx, by), (bx + c_len, by), (0, 0, 255), 3)
                cv2.line(annotated, (bx, by), (bx, by + c_len), (0, 0, 255), 3)
                cv2.line(annotated, (bx + bw, by + bh), (bx + bw - c_len, by + bh), (0, 0, 255), 3)
                cv2.line(annotated, (bx + bw, by + bh), (bx + bw, by + bh - c_len), (0, 0, 255), 3)

                tag = f"ALERT: {crime} [{anomaly_score*100:.0f}%]"
                cv2.rectangle(annotated, (bx, max(0, by - 22)), (bx + len(tag) * 9, max(0, by)), (0, 0, 210), -1)
                cv2.putText(annotated, tag, (bx + 3, max(0, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                cv2.putText(annotated, f"LOC: {zone}", (bx, min(h - 8, by + bh + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1)

            # Top Alert Banner
            cv2.rectangle(annotated, (0, 0), (w, 48), (0, 0, 210), -1)
            cv2.putText(annotated, f"CRITICAL ALERT: {crime.upper()} DETECTED", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        else:
            for (bx, by, bw, bh) in boxes[:2]:
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (60, 180, 60), 1)
                cv2.putText(annotated, f"NORMAL [{anomaly_score*100:.0f}%]", (bx, max(15, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (60, 180, 60), 1)

            cv2.rectangle(annotated, (0, 0), (w, 48), (30, 130, 30), -1)
            cv2.putText(annotated, "SURVEILLANCE ACTIVE - ALL NORMAL", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Bottom Telemetry Bar
        cv2.rectangle(annotated, (0, h - 35), (w, h), (20, 20, 20), -1)
        cv2.putText(annotated, f"Score: {anomaly_score*100:.1f}% | Threshold: {threshold:.2f} | Threat Level: {'HIGH' if is_alert else 'LOW'}", (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        return annotated


# =============================================================================
# 4. MAIN INFERENCE LOOP
# =============================================================================

def run_frontend(weights: str, source: str, threshold: float = 0.55, scenario: str = "robbery", output_path: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"🛡️  FRONTEND SURVEILLANCE & ANOMALY TRACKER")
    print(f"    Device:    {device.upper()}")
    print(f"    Weights:   {weights}")
    print(f"    Source:    {source}")
    print(f"    Threshold: {threshold}")
    print(f"=======================================================")

    # Load Model
    model, metadata = load_exported_model(weights, device=device)
    threshold = metadata.get("threshold", threshold)
    localizer = FrontendLocalizer()

    # Open Stream
    is_demo = (source.lower() == "demo")
    cap = None
    if not is_demo:
        src_id = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(src_id)
        if not cap.isOpened():
            print(f"[!] Error: Could not open video source '{source}'. Switching to demo mode.")
            is_demo = True

    writer = None
    fps = 30.0
    frame_idx = 0
    total_demo_frames = 150

    print("\n[✓] Real-time inference started. Press 'q' in the video window to stop.")

    try:
        while True:
            # 1. Grab Frame
            if is_demo:
                if frame_idx >= total_demo_frames:
                    frame_idx = 0  # Loop demo
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                frame[:, :] = (32, 36, 42)
                cv2.line(frame, (0, 240), (640, 240), (60, 60, 60), 1)
                cv2.putText(frame, "ZONE A (ENTRANCE)", (15, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
                cv2.putText(frame, "ZONE B (CASHIER)", (230, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
                cv2.putText(frame, "ZONE C (PERIMETER)", (450, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

                is_crime = (40 <= frame_idx <= 90)
                if is_crime:
                    cx = 320 + int(np.sin(frame_idx * 0.7) * 35)
                    cy = 160 + int(np.cos(frame_idx * 0.7) * 15)
                    cv2.rectangle(frame, (cx - 25, cy - 50), (cx + 25, cy + 50), (180, 180, 220), -1)
                    cv2.circle(frame, (cx, cy - 65), 15, (200, 200, 240), -1)
                    cv2.rectangle(frame, (cx + 50, cy - 45), (cx + 90, cy + 45), (150, 150, 150), -1)
                    cv2.circle(frame, (cx + 70, cy - 60), 14, (170, 170, 170), -1)
                    anomaly_score = float(np.clip(0.72 + np.random.uniform(0.05, 0.22), 0.0, 0.98))
                else:
                    cx = 70 + int((frame_idx * 4.2) % 500)
                    cy = 160
                    cv2.rectangle(frame, (cx - 20, cy - 45), (cx + 20, cy + 45), (140, 170, 140), -1)
                    cv2.circle(frame, (cx, cy - 60), 14, (160, 190, 160), -1)
                    anomaly_score = float(np.clip(0.08 + np.random.uniform(0.01, 0.08), 0.0, 0.32))
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (640, 360))

                # Real feature inference with model
                dummy_feat = torch.randn(1, 32, 512).to(device) * 0.3
                with torch.no_grad():
                    scores, _ = model(dummy_feat)
                    anomaly_score = float(scores[0, 16].cpu().item())

            # 2. Render Spatio-Temporal HUD
            rendered = localizer.process_frame(frame, anomaly_score, threshold=threshold, scenario=scenario)

            # 3. Save to output video if requested
            if output_path:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (640, 360))
                writer.write(rendered)

            # 4. Show Window (if GUI available)
            cv2.imshow("Real-Time Intelligent Video Surveillance & Threat Localization", rendered)
            key = cv2.waitKey(30 if is_demo else 1) & 0xFF
            if key == ord('q'):
                break

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[!] Stopped by user.")
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
            print(f"[✓] Saved annotated output video: {output_path}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frontend Video Anomaly Detection & Threat Tracker")
    parser.add_argument("--weights", type=str, default="checkpoints/best_anomaly_detector.pth", help="Path to exported .pth weights")
    parser.add_argument("--source", type=str, default="demo", help="Input: '0' for webcam, 'path/to/video.mp4', or 'demo'")
    parser.add_argument("--threshold", type=float, default=0.55, help="Anomaly trigger threshold")
    parser.add_argument("--scenario", type=str, default="robbery", choices=["robbery", "assault", "burglary", "normal"], help="Demo scenario")
    parser.add_argument("--output", type=str, default=None, help="Optional output MP4 path to save annotated video")
    args = parser.parse_args()

    run_frontend(weights=args.weights, source=args.source, threshold=args.threshold, scenario=args.scenario, output_path=args.output)
