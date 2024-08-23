import numpy as np
from init import signal

def spectral_percentiles(y: np.array, sr: float = 30000, frame_size: int = 2048, hop_size: int = 512):
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_size)[::hop_size]

    spectrum = np.abs(np.fft.fft(frames, n=frame_size, axis=-1)[:, :frame_size // 2])

    percentiles = np.percentile(spectrum, np.arange(1, 101), axis=1)

    return percentiles

percentile_descriptor = spectral_percentiles(signal)