import torch
import torch.nn as nn


class MILRankingHead(nn.Module):
    """
    3-Layer Fully-Connected Ranking Network for Weakly-Supervised
    Video Anomaly Detection (Sultani et al. CVPR 2018).

    Input:  (B, T, D) snippet feature vectors (e.g. D=2048 for I3D or D=4096 for C3D)
    Output: (B, T) snippet anomaly probability scores in [0.0, 1.0]
    """

    def __init__(self, in_features: int = 2048, dropout_rate: float = 0.6):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(512, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, D)
        Returns:
            scores: Tensor of shape (B, T) in range [0, 1]
        """
        # x: (B, T, D)
        out = self.dropout1(self.relu1(self.fc1(x)))
        out = self.dropout2(self.relu2(self.fc2(out)))
        scores = self.sigmoid(self.fc3(out))  # (B, T, 1)
        return scores.squeeze(-1)              # (B, T)
