import numpy as np
from init import signal

def spectral_slope(y: np.array, sr: float = 30000, frame_size: float = 2048, hop_size: float = 512):
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_size)[::hop_size]

    spectrum = np.abs(np.fft.fft(frames, n=frame_size, axis=-1)[:, :frame_size // 2])
    frequencies = np.fft.fftfreq(frame_size, d=1 / sr)[:frame_size // 2]

    slopes = np.sum((frequencies - np.mean(frequencies)) * (spectrum - np.mean(spectrum)), axis=1) / np.sum(
        (frequencies - np.mean(frequencies)) ** 2)

    return slopes

slope_descriptor = spectral_slope(signal)