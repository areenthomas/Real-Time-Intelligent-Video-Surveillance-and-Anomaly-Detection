import torch
import torch.nn as nn
from torchvision.models.video import s3d, S3D_Weights


class I3DFeatureExtractor(nn.Module):
    """
    Inflated 3D (I3D) / Separable 3D (S3D) Video Feature Extractor.
    Processes video clips of shape (B, 3, T, H, W) and extracts 2048-dim (or 1024-dim)
    spatial-temporal embeddings. Pretrained on Kinetics-400.
    """

    def __init__(self, pretrained: bool = True, output_dim: int = 2048):
        super().__init__()
        weights = S3D_Weights.DEFAULT if pretrained else None
        base_model = s3d(weights=weights)

        # Retain convolutional layers and global pooling
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Project to target feature dimension (standard 2048-dim for UCF-Crime)
        in_dim = 1024  # S3D penultimate feature dimension
        if output_dim != in_dim:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, output_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video clip tensor of shape (B, 3, T, H, W) with T >= 16 frames.
        Returns:
            Embedding vector of shape (B, output_dim).
        """
        feats = self.features(x)
        pooled = self.avgpool(feats)      # (B, 1024, 1, 1, 1)
        flat = torch.flatten(pooled, 1)   # (B, 1024)
        return self.proj(flat)            # (B, output_dim)
