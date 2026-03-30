import torch
import torch.nn as nn

from models.spatial.efficientnet import EfficientNetSpatial
from models.frequency.fft_model import FrequencyBranch


class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial = EfficientNetSpatial()
        self.frequency = FrequencyBranch()

        self.classifier = nn.Sequential(
            nn.Linear(self.spatial.feature_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        spatial_feat = self.spatial(x)
        freq_feat = self.frequency(x)

        combined = torch.cat([spatial_feat, freq_feat], dim=1)

        out = self.classifier(combined)
        return torch.sigmoid(out)