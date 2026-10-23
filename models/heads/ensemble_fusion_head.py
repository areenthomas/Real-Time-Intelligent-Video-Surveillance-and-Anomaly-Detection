import torch
import torch.nn as nn
from typing import Tuple, Dict

from .mil_ranking_head import MILRankingHead
from .rtfm_magnitude_head import RTFMMagnitudeHead
from .mist_attention_head import MISTAttentionHead


class MultiModelEnsembleHead(nn.Module):
    """
    Multi-Model Ensemble Fusion Architecture for Video Anomaly Detection.

    Dynamically fuses:
      1. RTFM Magnitude Head (captures high-energy kinetic anomalies: Accidents, Fighting)
      2. MIST Self-Guided Attention Head (isolates subtle localized anomalies: Shoplifting, Stealing)
      3. MIL Ranking Head (provides robust global video-bag separation)

    Uses a learned confidence gate to weight predictions per segment:
      Scores_ensemble = alpha * Scores_rtfm + beta * Scores_mist + gamma * Scores_mil
    """

    def __init__(self, in_features: int = 2048, hidden_dim: int = 512):
        super().__init__()
        self.rtfm_branch = RTFMMagnitudeHead(in_features=in_features, hidden_dim=hidden_dim)
        self.mist_branch = MISTAttentionHead(in_features=in_features, hidden_dim=hidden_dim)
        self.mil_branch = MILRankingHead(in_features=in_features)

        # Dynamic Gating Network to learn model weights dynamically
        self.gate = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: Input video snippet features of shape (B, T, in_features)

        Returns:
            fused_scores: Segment anomaly probability scores (B, T)
            diagnostics: Dictionary containing individual model predictions & gating weights
        """
        B, T, D = x.shape

        # 1. Forward through individual sub-models
        scores_rtfm, rtfm_feats = self.rtfm_branch(x)   # (B, T)
        scores_mist, attn_weights = self.mist_branch(x) # (B, T)
        scores_mil = self.mil_branch(x)                 # (B, T)

        # 2. Compute dynamic gating weights (B, T, 3)
        weights = self.gate(x)  # [alpha, beta, gamma] per snippet
        alpha = weights[:, :, 0]
        beta = weights[:, :, 1]
        gamma = weights[:, :, 2]

        # 3. Dynamic weighted fusion
        fused_scores = (
            alpha * scores_rtfm +
            beta * scores_mist +
            gamma * scores_mil
        )

        diagnostics = {
            "scores_rtfm": scores_rtfm,
            "scores_mist": scores_mist,
            "scores_mil": scores_mil,
            "rtfm_features": rtfm_feats,
            "attn_weights": attn_weights,
            "gating_weights": weights,
        }

        return fused_scores, diagnostics
