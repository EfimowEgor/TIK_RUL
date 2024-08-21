import numpy as np
from init import signal

def freq_peaks(y: np.array, sr: float = 30000, frame_size: float = 2048, hop_size: float = 512):
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_size)[::hop_size]

    spectrum = np.abs(np.fft.fft(frames, n=frame_size, axis=-1)[:, :frame_size // 2])
    frequencies = np.fft.fftfreq(frame_size, d=1 / sr)[:frame_size // 2]

    min_indices = np.argmin(spectrum, axis=1)
    peaks = frequencies[min_indices]

    return peaks

peak_freqs = freq_peaks(signal)