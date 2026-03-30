import torch
import torch.nn as nn


class VideoModel(nn.Module):
    def __init__(self, feature_dim=1280):
        super().__init__()

        self.lstm = nn.LSTM(feature_dim, 256, batch_first=True)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        return out[:, -1, :]  # last frame feature