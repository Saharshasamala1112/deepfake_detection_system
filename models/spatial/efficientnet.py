import torch
import torch.nn as nn
from torchvision import models

class EfficientNetSpatial(nn.Module):
    def __init__(self):
        super().__init__()

        model = models.efficientnet_b0(pretrained=True)
        self.features = model.features

        # ✅ IMPORTANT FIX
        self.feature_dim = 1280

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)
