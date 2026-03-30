import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, dim=1792):
        super().__init__()
        encoder = nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=2)

    def forward(self, x):
        return self.transformer(x).mean(dim=1)