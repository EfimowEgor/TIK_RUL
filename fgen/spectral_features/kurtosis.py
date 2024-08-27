from fgen.spectral_features.centroid import centroid
from fgen.spectral_features.spread import spread
import numpy as np

def kurtosis(y: np.array, sr: float = 30000, frame_size: int = 2048, hop_size: int = 512) -> np.array:
    centroids = centroid(y, sr, frame_size, hop_size)
    spreads = spread(y, sr, frame_size, hop_size)

    frames = np.lib.stride_tricks.sliding_window_view(y, frame_size)[::hop_size]

    spectrum = np.abs(np.fft.fft(frames, n=frame_size, axis=-1)[:, :frame_size // 2])
    frequencies = np.fft.fftfreq(frame_size, d=1 / sr)[:frame_size // 2]

    kurtosises = np.sum((frequencies - centroids[:, None]) ** 4 * spectrum) / (spreads ** 4 * np.sum(spectrum))

    return kurtosises