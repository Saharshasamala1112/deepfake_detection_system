import torchvision.models as models
import torch.nn as nn

class SpatialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b4(pretrained=True)
        self.model.classifier = nn.Identity()

    def forward(self, x):
        return self.model(x)