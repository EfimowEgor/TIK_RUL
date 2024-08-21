import numpy as np
from init import signal

def spectral_pow(y: np.array, sr: float = 30000, frame_size: float = 2048, hop_size: float = 512):
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_size)[::hop_size]

    spectrum = np.abs(np.fft.fft(frames, n=frame_size, axis=-1)[:, :frame_size // 2])

    pows = np.sum(spectrum ** 2, axis=1)

    return pows

pow_descriptor = spectral_pow(signal)