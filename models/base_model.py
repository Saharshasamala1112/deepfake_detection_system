import torch
from torchvision.models import resnet18
import torch.nn as nn

from models.frequency.fft_model import FrequencyBranch


class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial = resnet18(pretrained=True)
        self.spatial.fc = nn.Linear(512,1)

    def forward(self, x):
        return self.spatial
