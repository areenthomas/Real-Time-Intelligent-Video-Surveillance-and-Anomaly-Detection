"""
I3D / 3D CNN Feature Extraction Pipeline for Video Surveillance & Anomaly Detection
Extracts 2048-dimensional spatio-temporal snippet embeddings from raw MP4/AVI videos
using a 3D Convolutional Neural Network (I3D / S3D architecture pretrained on Kinetics-400).
"""

import os
import sys
import argparse
import time
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# Force UTF-8 encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.backbones.i3d import I3DFeatureExtractor
from data.temporal_sampler import sample_32_segments


class VideoFeatureExtractor:
    """
    Production-grade Feature Extraction Engine.
    Converts raw video footage into 2048-D temporal snippet features and 32-segment MIL bags.
    """

    def __init__(
        self,
        clip_len: int = 16,
        target_dim: int = 2048,
        device: Optional[str] = None,
        use_pretrained: bool = True,
    ):
        self.clip_len = clip_len
        self.target_dim = target_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[+] Initializing I3D Feature Extractor on Device: {self.device}")
        self.model = I3DFeatureExtractor(pretrained=use_pretrained, output_dim=target_dim)
        self.model.to(self.device)
        self.model.eval()

        # Kinetics-400 Standard Normalization
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def extract_from_video(self, video_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads an untrimmed video, slices it into 16-frame non-overlapping clips,
        and extracts 2048-D feature embeddings.

        Returns:
            raw_features: np.ndarray of shape (num_clips, 2048)
            bag_32_features: np.ndarray of shape (32, 2048) sampled for MIL
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR (OpenCV) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_frame = self.transform(rgb_frame)  # (3, 224, 224)
            frames.append(tensor_frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames could be read from {video_path}")

        num_frames = len(frames)
        num_clips = max(1, num_frames // self.clip_len)

        clip_tensors = []
        for i in range(num_clips):
            start_idx = i * self.clip_len
            end_idx = min(start_idx + self.clip_len, num_frames)
            clip = frames[start_idx:end_idx]

            # Pad short clips if necessary
            while len(clip) < self.clip_len:
                clip.append(clip[-1])

            # Stack into (3, T=16, H=224, W=224)
            clip_tensor = torch.stack(clip, dim=1)
            clip_tensors.append(clip_tensor)

        # Batch inference through I3D
        # (N, 3, 16, 224, 224)
        batch = torch.stack(clip_tensors, dim=0).to(self.device)

        with torch.no_grad():
            # Process in sub-batches if large to prevent memory overflow
            sub_batch_size = 16
            feat_list = []
            for b_idx in range(0, batch.size(0), sub_batch_size):
                sub_b = batch[b_idx : b_idx + sub_batch_size]
                feats = self.model(sub_b)  # (sub_B, 2048)
                feat_list.append(feats.cpu())
            all_features = torch.cat(feat_list, dim=0)  # (num_clips, 2048)

        # Resample to canonical 32-bag MIL tensor
        bag_32 = sample_32_segments(all_features, target_segments=32)

        raw_np = all_features.numpy().astype(np.float32)
        bag_np = bag_32.numpy().astype(np.float32)

        return raw_np, bag_np


def main():
    parser = argparse.ArgumentParser(description="Extract I3D 2048-D features from surveillance videos")
    parser.add_argument("--video", type=str, default="data/test_videos/Robbery050_x264.mp4", help="Path to raw video")
    parser.add_argument("--output_dir", type=str, default="data/features/i3d_ucf_crime", help="Output directory for .npy features")
    parser.add_argument("--use_pretrained", action="store_true", default=False, help="Use pretrained torchvision weights (requires internet)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("UCF-CRIME I3D FEATURE EXTRACTION PIPELINE")
    print(f"Target Video : {args.video}")
    print(f"Output Dir   : {args.output_dir}")
    print("=" * 80)

    extractor = VideoFeatureExtractor(clip_len=16, target_dim=2048, use_pretrained=args.use_pretrained)

    t0 = time.time()
    raw_feats, bag_32_feats = extractor.extract_from_video(args.video)
    elapsed = time.time() - t0

    video_basename = os.path.splitext(os.path.basename(args.video))[0]
    raw_out_path = os.path.join(args.output_dir, f"{video_basename}_raw.npy")
    bag_out_path = os.path.join(args.output_dir, f"{video_basename}.npy")

    np.save(raw_out_path, raw_feats)
    np.save(bag_out_path, bag_32_feats)

    print("-" * 80)
    print(f"[✓] Feature Extraction Completed in {elapsed:.2f} seconds!")
    print(f"[✓] Raw Snippet Features Shape : {raw_feats.shape} -> Saved to {raw_out_path}")
    print(f"[✓] Canonical 32-Bag MIL Shape : {bag_32_feats.shape} -> Saved to {bag_out_path}")
    print(f"[✓] Feature Range: Min={bag_32_feats.min():.4f}, Max={bag_32_feats.max():.4f}, Mean={bag_32_feats.mean():.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()