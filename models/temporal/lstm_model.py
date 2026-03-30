import torch
import torch.nn as nn

class TemporalModel(nn.Module):
    def __init__(self, feature_dim=1280, hidden_dim=256):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return torch.sigmoid(out)