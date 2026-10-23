import torch
import numpy as np


def sample_32_segments(features: torch.Tensor, target_segments: int = 32) -> torch.Tensor:
    """
    Interpolates or uniformly partitions an arbitrary-length feature sequence (T, D)
    into exactly target_segments (default: 32) temporal bags for Multiple Instance Learning.

    Args:
        features: Input feature tensor of shape (T, D) or numpy array
        target_segments: Target number of segments (standard: 32)

    Returns:
        Sampled feature tensor of shape (32, D)
    """
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()

    T, D = features.shape

    if T == target_segments:
        return features

    if T < target_segments:
        # Dynamic replication for extremely short videos
        indices = np.linspace(0, T - 1, target_segments)
        indices = np.round(indices).astype(int)
        return features[indices]

    # Uniform segment splitting with mean pooling per interval
    seg_len = T / float(target_segments)
    sampled = []
    for i in range(target_segments):
        start_idx = int(np.floor(i * seg_len))
        end_idx = int(np.ceil((i + 1) * seg_len))
        if start_idx >= end_idx:
            end_idx = start_idx + 1
        end_idx = min(end_idx, T)
        segment_mean = torch.mean(features[start_idx:end_idx], dim=0)
        sampled.append(segment_mean)

    return torch.stack(sampled, dim=0)  # (32, D)
