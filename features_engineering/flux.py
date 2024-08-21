import numpy as np
from init import signal

def spectral_flux(y: np.array, sr: float = 30000, frame_size: float = 2048, hop_size: float = 512, p: float = 2):
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_size)[::hop_size]

    spectrum = np.abs(np.fft.fft(frames, n=frame_size, axis=-1)[:, :frame_size // 2])

    diff_spectrum = np.diff(spectrum, axis=0)

    fluxes = np.sum(np.abs(diff_spectrum) ** p, axis=1) ** (1 / p)

    return np.concatenate(([0], fluxes))

flux_descriptor = spectral_flux(signal)