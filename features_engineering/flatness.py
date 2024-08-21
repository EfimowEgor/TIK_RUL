import numpy as np
from init import signal

def spectral_flatness(y: np.array, sr: float = 30000, frame_size: float = 2048, hop_size: float = 512):
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_size)[::hop_size]

    spectrum = np.abs(np.fft.fft(frames, n=frame_size, axis=-1)[:, :frame_size // 2])

    geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-10), axis=1))
    arithmetic_mean = np.mean(spectrum, axis=1)
    flatnesses = geometric_mean / arithmetic_mean

    return flatnesses

flatness_descriptor = spectral_flatness(signal)