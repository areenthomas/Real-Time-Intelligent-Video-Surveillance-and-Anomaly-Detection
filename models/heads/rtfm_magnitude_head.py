import torch
import torch.nn as nn
from typing import Tuple


class PyramidDilatedConv(nn.Module):
    """
    Multi-Scale Temporal Dilated Convolution Module.
    Captures short, medium, and long-range temporal contexts.
    """

    def __init__(self, in_channels: int, out_channels: int, dilations=(1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels // len(dilations),
                    kernel_size=3,
                    padding=d,
                    dilation=d,
                ),
                nn.BatchNorm1d(out_channels // len(dilations)),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for d in dilations
        ])
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T)
        branch_outs = [branch(x) for branch in self.branches]
        concat = torch.cat(branch_outs, dim=1)
        return self.fusion(concat)


class RTFMMagnitudeHead(nn.Module):
    """
    Robust Temporal Feature Magnitude (RTFM) Head (Tian et al. ICCV 2021).

    Combines Pyramid Dilated Convolutions and Temporal Self-Attention
    to enforce feature magnitude margins on abnormal clips.
    """

    def __init__(
        self,
        in_features: int = 2048,
        hidden_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        # 1. Feature Projection
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        )

        # 2. Pyramid Temporal Dilated Convolution (B, C, T)
        self.temporal_pyramid = PyramidDilatedConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            dilations=(1, 2, 4),
        )

        # 3. Multi-Head Temporal Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(hidden_dim)

        # 4. Classification & Anomaly Scoring Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input video snippet features of shape (B, T, in_features)

        Returns:
            scores: Anomaly probability scores of shape (B, T)
            refined_feats: Refined temporal features of shape (B, T, hidden_dim) for magnitude loss
        """
        B, T, D = x.shape

        # Step 1: Linear Projection per snippet
        x_flat = x.view(B * T, D)
        h = self.proj(x_flat).view(B, T, -1)  # (B, T, hidden_dim)

        # Step 2: Temporal Dilated Convolutions over sequence T
        h_conv = h.permute(0, 2, 1)           # (B, hidden_dim, T)
        h_conv = self.temporal_pyramid(h_conv).permute(0, 2, 1)  # (B, T, hidden_dim)

        # Residual connection
        h = h + h_conv

        # Step 3: Self-Attention across temporal segments
        attn_out, _ = self.self_attn(h, h, h)
        h = self.norm_attn(h + attn_out)       # (B, T, hidden_dim)

        # Step 4: Scoring
        scores = self.classifier(h).squeeze(-1)  # (B, T)

        return scores, h
