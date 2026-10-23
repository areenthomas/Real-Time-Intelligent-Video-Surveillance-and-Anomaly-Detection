import argparse
import time
import os
import cv2
import numpy as np
import torch
import torch.nn as nn

from models import (
    MILRankingHead,
    RTFMMagnitudeHead,
    MISTAttentionHead,
    MultiModelEnsembleHead,
    LightweightEdgeBackbone,
)
from streaming.sliding_window_buffer import CircularSlidingWindowBuffer
from streaming.alert_dispatcher import AlertDispatcher


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Video Surveillance & Anomaly Detection")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: '0' for webcam, or path to MP4 video, or RTSP URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["mil", "rtfm", "mist", "ensemble"],
        help="Anomaly detection model to use for inference",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained .pth model weights checkpoint",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Anomaly detection alarm threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=16,
        help="Number of frames per 3D clip window",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Frame stride between inference passes",
    )
    return parser.parse_args()


def load_model(model_name: str, weights_path: str = None, device: str = "cpu") -> nn.Module:
    if model_name == "mil":
        model = MILRankingHead(in_features=512)
    elif model_name == "rtfm":
        model = RTFMMagnitudeHead(in_features=512, hidden_dim=256)
    elif model_name == "mist":
        model = MISTAttentionHead(in_features=512, hidden_dim=256)
    elif model_name == "ensemble":
        model = MultiModelEnsembleHead(in_features=512, hidden_dim=256)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[✓] Loaded trained weights from: {weights_path}")
    else:
        print(f"[!] Using initialized {model_name.upper()} model weights for real-time demo.")

    model.to(device)
    model.eval()
    return model


def draw_hud(
    frame: np.ndarray,
    score: float,
    threshold: float,
    fps: float,
    waveform_history: list,
) -> np.ndarray:
    """Draws real-time heads-up display overlay onto video frame."""
    h, w, _ = frame.shape
    is_anomaly = score >= threshold

    # 1. Top Status Banner
    banner_color = (0, 0, 220) if is_anomaly else (40, 160, 40)
    cv2.rectangle(frame, (0, 0), (w, 55), banner_color, -1)

    status_text = "🚨 CRITICAL ANOMALY ALERT" if is_anomaly else "NORMAL SURVEILLANCE"
    cv2.putText(
        frame,
        f"STATUS: {status_text}",
        (15, 38),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    # 2. Score Bar & Metrics
    bar_width = 200
    bar_x = w - bar_width - 20
    cv2.rectangle(frame, (bar_x, 15), (bar_x + bar_width, 40), (70, 70, 70), -1)
    fill_width = int(bar_width * min(1.0, score))
    cv2.rectangle(frame, (bar_x, 15), (bar_x + fill_width, 40), (0, 140, 255), -1)
    cv2.rectangle(frame, (bar_x, 15), (bar_x + bar_width, 40), (220, 220, 220), 2)

    score_text = f"Anomaly: {score*100:.1f}%"
    cv2.putText(
        frame,
        score_text,
        (bar_x - 140, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    # 3. Live FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (15, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    # 4. Real-Time Anomaly Waveform Sparkline (bottom right)
    plot_w, plot_h = 180, 60
    plot_x, plot_y = w - plot_w - 15, h - plot_h - 15
    cv2.rectangle(
        frame,
        (plot_x, plot_y),
        (plot_x + plot_w, plot_y + plot_h),
        (20, 20, 20),
        -1,
    )
    cv2.rectangle(
        frame,
        (plot_x, plot_y),
        (plot_x + plot_w, plot_y + plot_h),
        (100, 100, 100),
        1,
    )

    if len(waveform_history) > 1:
        step = plot_w / float(max(1, len(waveform_history) - 1))
        pts = []
        for i, val in enumerate(waveform_history):
            pt_x = int(plot_x + i * step)
            pt_y = int(plot_y + plot_h - (val * (plot_h - 4)))
            pts.append((pt_x, pt_y))
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], (0, 165, 255), 2)

    return frame


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=====================================================================")
    print("🎥 Starting Real-Time Video Surveillance & Anomaly Detection")
    print(f"   Model: {args.model.upper()} | Device: {device} | Threshold: {args.threshold}")
    print("=====================================================================")

    # Initialize video capture source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video source: {args.source}")
        return

    # Initialize lightweight feature backbone & anomaly head
    backbone = LightweightEdgeBackbone(pretrained=False, output_dim=512).to(device)
    backbone.eval()
    head = load_model(args.model, weights_path=args.weights, device=device)

    # Initialize buffer and alert dispatcher
    buffer = CircularSlidingWindowBuffer(window_size=args.window_size, stride=args.stride)
    dispatcher = AlertDispatcher(alert_threshold=args.threshold, cooldown_seconds=5.0)

    current_score = 0.0
    waveform_history = [0.0] * 30
    frame_count = 0
    prev_time = time.time()
    fps = 30.0

    print("🟢 Live stream active! Press 'q' on the video window to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] End of video stream reached.")
                break

            frame_count += 1
            buffer.add_frame(frame)

            # Perform inference whenever sliding buffer has reached stride
            if buffer.is_ready_for_inference():
                clip_tensor = buffer.get_window_tensor()
                if clip_tensor is not None:
                    clip_tensor = clip_tensor.to(device)
                    with torch.no_grad():
                        # Extract 3D spatial-temporal embedding
                        feat = backbone(clip_tensor)  # (1, 512)
                        # Expand to 32 segments for head
                        bag_feat = feat.repeat(1, 32, 1)  # (1, 32, 512)

                        if args.model in ["rtfm", "mist", "ensemble"]:
                            scores, _ = head(bag_feat)
                        else:
                            scores = head(bag_feat)

                        current_score = float(scores[0, 16].cpu().item())
                        waveform_history.pop(0)
                        waveform_history.append(current_score)
                        dispatcher.process_score(current_score, frame)

            # Compute FPS
            now = time.time()
            if now - prev_time >= 0.5:
                fps = frame_count / (now - prev_time)
                frame_count = 0
                prev_time = now

            # Render HUD overlay onto frame
            display_frame = draw_hud(frame, current_score, args.threshold, fps, waveform_history)
            cv2.imshow("Real-Time Intelligent Video Surveillance", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[✓] Video stream closed successfully.")


if __name__ == "__main__":
    main()
