import librosa
import numpy as np

def compute_spectrogram(y, sr):
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    return librosa.power_to_db(spec)