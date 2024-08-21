import numpy as np
from init import signal

def spectral_centroid(y: np.array, sr: float = 30000, frame_size: int = 2048, hop_size: int = 512):
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_size)[::hop_size]

    spectrum = np.abs(np.fft.fft(frames, n=frame_size, axis=-1)[:, :frame_size // 2])
    frequencies = np.fft.fftfreq(frame_size, d=1 / sr)[:frame_size // 2]

    centroids = np.sum(spectrum * frequencies, axis=1) / np.sum(spectrum, axis=1)

    return centroids

centroid_descriptor = spectral_centroid(signal)