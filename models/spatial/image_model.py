import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        return self.backbone(x)  # feature vector