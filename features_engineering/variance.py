import numpy as np
from init import signal

def spectral_variance(y: np.array, sr: float = 30000, frame_size: int = 2048, hop_size: int = 512):
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_size)[::hop_size]

    spectrum = np.abs(np.fft.fft(frames, n=frame_size, axis=-1)[:, :frame_size // 2])

    variances = np.var(spectrum, axis=1)

    return variances

variance_descriptor = spectral_variance(signal)