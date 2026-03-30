import torch
import torch.nn as nn
from models.base_model import DeepfakeDetector

class VideoModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial = DeepfakeDetector()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)

    def forward(self, frames):
        # frames: (B, T, C, H, W)
        B, T, C, H, W = frames.shape

        outputs = []
        for t in range(T):
            out = self.spatial(frames[:, t])
            outputs.append(out)

        seq = torch.stack(outputs, dim=1)  # (B, T, 1)

        _, (h, _) = self.lstm(seq)
        return h[-1]  # (B, 64)