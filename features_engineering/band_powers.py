import numpy as np
from init import signal

def band_powers(y: np.array, sr: float = 30000, frame_size: float = 2048, hop_size: float = 512):
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_size)[::hop_size]

    spectrum = np.abs(np.fft.fft(frames, n=frame_size, axis=-1)[:, :frame_size // 2])
    frequencies = np.fft.fftfreq(frame_size, d=1 / sr)[:frame_size // 2]

    band_edges = np.arange(0, 30000 + 1000, 1000)
    band_indices = [np.logical_and(frequencies >= band_edges[i], frequencies < band_edges[i + 1]) for i in range(len(band_edges) - 1)]

    band_pows = np.zeros((len(band_edges) - 1, frames.shape[0]))

    for i, indices in enumerate(band_indices):
        band_pows[i, :] = np.sum(spectrum[:, indices] ** 2, axis=1)

    return band_pows

band_pows = band_powers(signal)