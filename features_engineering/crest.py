import numpy as np
from init import signal

def spectral_crest(y: np.array, sr: float = 30000, frame_size: float = 2048, hop_size: float = 512):
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_size)[::hop_size]

    spectrum = np.abs(np.fft.fft(frames, n=frame_size, axis=-1)[:, :frame_size // 2])

    crests = np.max(spectrum, axis=1) / (np.sum(spectrum, axis=1) / spectrum.shape[1])

    return crests

crest_descriptor = spectral_crest(signal)