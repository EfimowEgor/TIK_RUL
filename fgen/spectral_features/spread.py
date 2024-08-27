from fgen.spectral_features.centroid import centroid
import numpy as np

def spread(y: np.array, sr: float = 30000, frame_size: int = 2048, hop_size: int = 512) -> np.array:
    centroids = centroid(y, sr, frame_size, hop_size)

    frames = np.lib.stride_tricks.sliding_window_view(y, frame_size)[::hop_size]

    spectrum = np.abs(np.fft.fft(frames, n=frame_size, axis=-1)[:, :frame_size // 2])
    frequencies = np.fft.fftfreq(frame_size, d=1 / sr)[:frame_size // 2]

    spreads = np.sqrt(np.sum(((frequencies - centroids[:, None]) ** 2) * spectrum, axis=1) / np.sum(spectrum, axis=1))

    return spreads