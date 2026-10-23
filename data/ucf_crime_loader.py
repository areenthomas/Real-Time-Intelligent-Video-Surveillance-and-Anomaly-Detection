import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Optional
from .temporal_sampler import sample_32_segments


class UCFCrimeDataset(Dataset):
    """
    PyTorch Dataset for Weakly Supervised Video Anomaly Detection on UCF-Crime.

    Pairs one abnormal video bag with one normal video bag per index to facilitate
    Multiple Instance Learning (MIL) ranking loss computation.
    """

    def __init__(
        self,
        features_dir: str = "data/features/i3d_ucf_crime/",
        split_file: Optional[str] = "data/splits/Anomaly_Train.txt",
        segments_per_bag: int = 32,
        feature_dim: int = 2048,
        mock_mode: bool = False,
        num_mock_samples: int = 200,
    ):
        self.features_dir = features_dir
        self.segments_per_bag = segments_per_bag
        self.feature_dim = feature_dim
        self.mock_mode = mock_mode

        self.abnormal_files: List[str] = []
        self.normal_files: List[str] = []

        if not mock_mode and split_file and os.path.exists(split_file) and os.path.exists(features_dir):
            self._load_split(split_file)
        else:
            # Automatic mock mode for dry-runs / Colab environment testing
            self.mock_mode = True
            self._generate_mock_list(num_mock_samples)

    def _load_split(self, split_file: str):
        with open(split_file, "r") as f:
            for line in f:
                path = line.strip().split()[0]
                base_name = os.path.splitext(os.path.basename(path))[0]
                feat_path = os.path.join(self.features_dir, f"{base_name}.npy")
                if "Normal" in base_name:
                    self.normal_files.append(feat_path)
                else:
                    self.abnormal_files.append(feat_path)

        if len(self.abnormal_files) == 0 or len(self.normal_files) == 0:
            self.mock_mode = True
            self._generate_mock_list(100)

    def _generate_mock_list(self, n: int):
        self.abnormal_files = [f"mock_abnormal_{i}" for i in range(n)]
        self.normal_files = [f"mock_normal_{i}" for i in range(n)]

    def __len__(self) -> int:
        return min(len(self.abnormal_files), len(self.normal_files))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            feat_abnormal: Tensor of shape (32, feature_dim)
            feat_normal:   Tensor of shape (32, feature_dim)
        """
        if self.mock_mode:
            # Synthetic feature simulation:
            # Abnormal has large feature magnitude and localized activation spike
            feat_abnormal = torch.randn(self.segments_per_bag, self.feature_dim) * 0.5
            feat_abnormal[10:16] += torch.randn(6, self.feature_dim) * 3.5  # High energy anomaly

            # Normal has ambient baseline magnitude
            feat_normal = torch.randn(self.segments_per_bag, self.feature_dim) * 0.5
            return feat_abnormal, feat_normal

        # Real feature loading
        abn_path = self.abnormal_files[idx]
        norm_path = self.normal_files[idx]

        feat_abn = np.load(abn_path)
        feat_norm = np.load(norm_path)

        feat_abn_32 = sample_32_segments(feat_abn, self.segments_per_bag)
        feat_norm_32 = sample_32_segments(feat_norm, self.segments_per_bag)

        return feat_abn_32, feat_norm_32


def get_ucf_crime_loaders(
    features_dir: str = "data/features/i3d_ucf_crime/",
    batch_size: int = 32,
    num_workers: int = 2,
    mock_mode: bool = False,
) -> DataLoader:
    """Creates a PyTorch DataLoader yielding paired normal and abnormal video bags."""
    dataset = UCFCrimeDataset(features_dir=features_dir, mock_mode=mock_mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
