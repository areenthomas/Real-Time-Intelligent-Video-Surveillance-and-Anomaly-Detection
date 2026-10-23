import torch
import torch.nn as nn
from typing import Tuple


class SelfGuidedAttention(nn.Module):
    """
    Self-Guided Attention Module (MIST - Feng et al. CVPR 2021).
    Computes temporal importance weights to isolate subtle anomalous actions.
    """

    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T)
        h = self.relu(self.conv1(x))
        attn_weights = self.sigmoid(self.conv2(h))  # (B, 1, T)
        return attn_weights


class MISTAttentionHead(nn.Module):
    """
    Self-Guided Attention Boosted Feature Encoder for Weakly-Supervised
    Video Anomaly Detection (MIST, Feng et al. CVPR 2021).

    Applies self-guided temporal attention weights to boost subtle anomalies
    (such as Shoplifting, Stealing, and Burglary).
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
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.self_guided_attn = SelfGuidedAttention(in_features=hidden_dim, hidden_dim=128)

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
            x: Input video snippet features (B, T, in_features)

        Returns:
            scores: Anomaly probability scores of shape (B, T)
            attn_weights: Attention importance weights of shape (B, T)
        """
        B, T, D = x.shape

        # Encode snippet features
        x_flat = x.view(B * T, D)
        h = self.encoder(x_flat).view(B, T, -1)  # (B, T, hidden_dim)

        # Compute self-guided attention over sequence
        h_trans = h.permute(0, 2, 1)             # (B, hidden_dim, T)
        attn = self.self_guided_attn(h_trans)    # (B, 1, T)
        attn_weights = attn.squeeze(1)           # (B, T)

        # Attention-boosted feature representation
        h_boosted = h * (1.0 + attn.permute(0, 2, 1))

        # Predict anomaly probability
        scores = self.classifier(h_boosted).squeeze(-1)  # (B, T)

        return scores, attn_weights
