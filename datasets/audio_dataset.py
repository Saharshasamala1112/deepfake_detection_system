import os
import torch
import librosa
import numpy as np
from .base_dataset import BaseDataset


class AudioDataset(BaseDataset):
    def __init__(self, audio_dir, sample_rate=16000):
        super().__init__()
        self.audio_paths = [
            os.path.join(audio_dir, f)
            for f in os.listdir(audio_dir)
            if f.endswith(".wav")
        ]
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audio_paths)

    def extract_spectrogram(self, path):
        y, sr = librosa.load(path, sr=self.sample_rate)

        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrogram = librosa.power_to_db(spectrogram)

        spectrogram = np.expand_dims(spectrogram, axis=0)
        return torch.tensor(spectrogram, dtype=torch.float32)

    def __getitem__(self, idx):
        spec = self.extract_spectrogram(self.audio_paths[idx])
        label = 0  # placeholder

        return spec, torch.tensor(label, dtype=torch.float32)