import torch.nn as nn

from models.spatial.image_model import ImageModel
from models.video.video_model import VideoModel
from models.audio.audio_cnn import AudioCNN
from fusion.attention_fusion import AttentionFusion


class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_model = ImageModel()
        self.video_model = VideoModel()
        self.audio_model = AudioCNN()

        self.fusion = AttentionFusion(dim=128)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, image, video, audio):
        img_feat = self.image_model(image)
        vid_feat = self.video_model(video)
        aud_feat = self.audio_model(audio)

        fused = self.fusion(img_feat, vid_feat, aud_feat)

        return self.classifier(fused)