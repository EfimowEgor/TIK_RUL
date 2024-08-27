import numpy as np


def band_power(y: np.array, lower_bound: int, upper_bound: int,
                sr: float = 30000, frame_size: int = 2048, hop_size: int = 512) -> np.array:
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_size)[::hop_size]

    spectrum = np.abs(np.fft.fft(frames, n=frame_size, axis=-1)[:, :frame_size // 2]) ** 2
    frequencies = np.fft.fftfreq(frame_size, d=1 / sr)[:frame_size // 2]

    valid_indices = np.logical_and(frequencies >= lower_bound, frequencies <= upper_bound)

    band_powers = np.sum(spectrum[:, valid_indices]**2, axis=1)

    return band_powers