import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, img_feat, vid_feat, aud_feat):
        combined = torch.cat([img_feat, vid_feat, aud_feat], dim=1)

        weights = self.attn(combined)

        w1, w2, w3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]

        fused = w1 * img_feat + w2 * vid_feat + w3 * aud_feat

        return fused