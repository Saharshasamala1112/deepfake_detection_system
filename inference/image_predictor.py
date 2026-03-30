import torch
import librosa
import numpy as np

from models.multimodal_model import MultiModalModel
from utils.device import get_device


class AudioPredictor:
    def __init__(self):
        self.device = get_device()

        self.model = MultiModalModel().to(self.device)
        self.model.load_state_dict(torch.load("model.pth", map_location=self.device))
        self.model.eval()

    def audio_to_spec(self, path):
        y, sr = librosa.load(path, sr=16000)
        spec = librosa.feature.melspectrogram(y=y, sr=sr)
        spec = librosa.power_to_db(spec)

        spec = np.resize(spec, (224, 224))
        spec = np.stack([spec, spec, spec])

        return torch.tensor(spec).float().unsqueeze(0)

    def predict(self, audio_path):
        spec = self.audio_to_spec(audio_path).to(self.device)

        with torch.no_grad():
            output = self.model(spec).item()

        if output > 0.6:
            label = "FAKE"
        elif output < 0.4:
            label = "REAL"
        else:
            label = "UNCERTAIN"

        return label, output