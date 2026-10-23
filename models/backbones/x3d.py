import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class LightweightEdgeBackbone(nn.Module):
    """
    Lightweight 3D Spatial-Temporal Backbone for Low-Latency Real-Time
    Edge Surveillance Inference (>60 FPS).

    Processes video clips (B, 3, 16, 224, 224) and outputs compact embeddings
    (default 512-dim or 2048-dim) with ultra-low memory footprint.
    """

    def __init__(self, pretrained: bool = True, output_dim: int = 512):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        base_model = r3d_18(weights=weights)

        # Retain stem and layer 1-4
        self.stem = base_model.stem
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        in_dim = 512  # R3D-18 final channel count
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
            x: Video clip tensor of shape (B, 3, T, H, W).
        Returns:
            Feature embedding of shape (B, output_dim).
        """
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        pooled = self.avgpool(h)
        flat = torch.flatten(pooled, 1)
        return self.proj(flat)
