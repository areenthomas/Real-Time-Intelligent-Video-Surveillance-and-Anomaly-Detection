import torch
import torch.nn as nn


class MILRankingLoss(nn.Module):
    """
    Multiple Instance Learning (MIL) Ranking Loss with Temporal Smoothness
    and Sparsity Regularization (Sultani et al. CVPR 2018).

    L = max(0, 1 - max(S_a) + max(S_n))
        + lambda_smooth * sum((S_a[t+1] - S_a[t])^2)
        + lambda_sparse * sum(S_a)
    """

    def __init__(self, lambda_smooth: float = 8e-5, lambda_sparse: float = 8e-5):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse

    def forward(
        self,
        scores_abnormal: torch.Tensor,
        scores_normal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            scores_abnormal: Tensor of shape (B, T) with anomaly scores for abnormal bags.
            scores_normal:   Tensor of shape (B, T) with anomaly scores for normal bags.

        Returns:
            Scalar total ranking loss.
        """
        # Max score in each abnormal bag and normal bag
        max_abnormal, _ = torch.max(scores_abnormal, dim=1)  # (B,)
        max_normal, _ = torch.max(scores_normal, dim=1)      # (B,)

        # Hinge ranking margin loss
        ranking_loss = torch.mean(torch.clamp(1.0 - max_abnormal + max_normal, min=0.0))

        # Temporal smoothness constraint on abnormal bags (adjacent segments must transition smoothly)
        diff = scores_abnormal[:, 1:] - scores_abnormal[:, :-1]
        smoothness_loss = torch.mean(torch.sum(diff ** 2, dim=1))

        # Temporal sparsity constraint on abnormal bags (anomalies are rare and brief)
        sparsity_loss = torch.mean(torch.sum(scores_abnormal, dim=1))

        total_loss = (
            ranking_loss
            + self.lambda_smooth * smoothness_loss
            + self.lambda_sparse * sparsity_loss
        )
        return total_loss
