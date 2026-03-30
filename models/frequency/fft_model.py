import torch.fft as fft

def extract_fft(x):
    return fft.fft(x).real