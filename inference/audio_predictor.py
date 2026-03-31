import torch
import numpy as np
import librosa
import pandas as pd
import tempfile

class AudioPredictor:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def load_audio(self, file):
        if file.name.endswith(".parquet"):
            df = pd.read_parquet(file)
            audio = df.values.flatten()
            sr = 16000
        else:
            audio, sr = librosa.load(file, sr=16000)

        return audio[:16000]  # 🔥 only 1 sec

    def predict(self, file):
        audio = self.load_audio(file)

        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
        mfcc = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            output = self.model(mfcc)
            prob = torch.sigmoid(output).item()

        return {
            "prediction": "FAKE" if prob > 0.5 else "REAL",
            "confidence": float(prob)
        }
