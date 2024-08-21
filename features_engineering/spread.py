from centroid import spectral_centroid
import numpy as np
from init import signal

def spectral_spread(y: np.array, sr: float = 30000, frame_size: float = 2048, hop_size: float = 512):
    centroids = spectral_centroid(y, sr, frame_size, hop_size)

    frames = np.lib.stride_tricks.sliding_window_view(y, frame_size)[::hop_size]

    spectrum = np.abs(np.fft.fft(frames, n=frame_size, axis=-1)[:, :frame_size // 2])
    frequencies = np.fft.fftfreq(frame_size, d=1 / sr)[:frame_size // 2]

    spreads = np.sqrt(np.sum(((frequencies - centroids[:, None]) ** 2) * spectrum, axis=1) / np.sum(spectrum, axis=1))

    return spreads

spread_descriptor = spectral_spread(signal)