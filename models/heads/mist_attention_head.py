import torch
import torch.nn as nn
from typing import Tuple


class MISTAttentionHead(nn.Module):
    """
    Self-Guided Attention Boosted Feature Encoder for Weakly-Supervised
    Video Anomaly Detection (MIST, Feng et al. CVPR 2021).

    Uses LayerNorm and temporal 1D attention convolution to boost localized anomalies.
    """

    def __init__(
        self,
        in_features: int = 2048,
        hidden_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.self_guided_attn = nn.Sequential(
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input feature tensor of shape (B, T, in_features)

        Returns:
            scores: Anomaly probability scores of shape (B, T)
            attn_weights: Attention weights of shape (B, T)
        """
        # Encode features
        h = self.encoder(x)  # (B, T, hidden_dim)

        # Self-guided attention along temporal dimension T
        h_t = h.permute(0, 2, 1)           # (B, hidden_dim, T)
        attn = self.self_guided_attn(h_t)  # (B, 1, T)
        attn_weights = attn.squeeze(1)     # (B, T)

        # Attention-boosted feature representation
        h_boosted = h * (1.0 + attn.permute(0, 2, 1))

        # Predict anomaly probability
        scores = self.classifier(h_boosted).squeeze(-1)  # (B, T)

        return scores, attn_weights
