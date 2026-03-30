import torch
import librosa
import numpy as np
import pandas as pd
import os

from models.multimodal_model import MultiModalModel
from utils.device import get_device


class AudioPredictor:
    def __init__(self):
        self.device = get_device()

        self.model = MultiModalModel().to(self.device)
        self.model.load_state_dict(torch.load("model.pth", map_location=self.device))
        self.model.eval()

    def load_audio(self, path):
        ext = os.path.splitext(path)[1].lower()

        # 🔊 WAV / FLAC
        if ext in [".wav", ".flac"]:
            y, sr = librosa.load(path, sr=16000)
            return y, sr

        # 📊 PARQUET
        elif ext == ".parquet":
            df = pd.read_parquet(path)

            # assume audio column exists
            if "audio" in df.columns:
                y = np.array(df["audio"].iloc[0])
            else:
                y = df.values.flatten()

            sr = 16000
            return y, sr

        else:
            raise ValueError("Unsupported audio format")

    def predict(self, audio_path):
        try:
            y, sr = self.load_audio(audio_path)

            spec = librosa.feature.melspectrogram(y=y, sr=sr)
            spec = librosa.power_to_db(spec)

            spec = np.resize(spec, (224, 224))
            spec = np.stack([spec, spec, spec])

            spec = torch.tensor(spec).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(spec).item()

            if output > 0.6:
                return "FAKE", output
            elif output < 0.4:
                return "REAL", output
            else:
                return "UNCERTAIN", output

        except Exception as e:
            print("Audio error:", e)
            return "ERROR", 0.0