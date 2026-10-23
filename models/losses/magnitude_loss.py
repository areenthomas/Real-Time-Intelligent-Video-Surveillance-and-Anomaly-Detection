import torch
import torch.nn as nn


class FeatureMagnitudeLoss(nn.Module):
    """
    Robust Temporal Feature Magnitude (RTFM) Separation Loss (Tian et al. ICCV 2021).

    Enforces that the top-k largest feature vectors in abnormal bags have a distinct
    Euclidean norm margin (m) over the top-k feature vectors in normal bags.

    L_mag = max(0, m - (1/k)*sum(topk(||e_a||_2)) + (1/k)*sum(topk(||e_n||_2)))
    """

    def __init__(self, margin: float = 100.0, top_k: int = 3):
        super().__init__()
        self.margin = margin
        self.top_k = top_k

    def forward(
        self,
        features_abnormal: torch.Tensor,
        features_normal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features_abnormal: Tensor of shape (B, T, D) with feature embeddings for abnormal bags.
            features_normal:   Tensor of shape (B, T, D) with feature embeddings for normal bags.

        Returns:
            Scalar feature magnitude separation loss.
        """
        # Calculate L2 norms along feature dimension D
        norm_abnormal = torch.norm(features_abnormal, p=2, dim=2)  # (B, T)
        norm_normal = torch.norm(features_normal, p=2, dim=2)      # (B, T)

        # Select top-k largest feature magnitudes per video bag
        topk_abnormal, _ = torch.topk(norm_abnormal, k=self.top_k, dim=1)  # (B, k)
        topk_normal, _ = torch.topk(norm_normal, k=self.top_k, dim=1)      # (B, k)

        mean_topk_abnormal = torch.mean(topk_abnormal, dim=1)  # (B,)
        mean_topk_normal = torch.mean(topk_normal, dim=1)      # (B,)

        magnitude_loss = torch.mean(
            torch.clamp(self.margin - mean_topk_abnormal + mean_topk_normal, min=0.0)
        )
        return magnitude_loss
