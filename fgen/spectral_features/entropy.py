import numpy as np

def entropy(y: np.array, sr: float = 30000, frame_size: int = 2048, hop_size: int = 512) -> np.array:
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_size)[::hop_size]

    spectrum = np.abs(np.fft.fft(frames, n=frame_size, axis=-1)[:, :frame_size // 2])

    spectrum_sum = np.sum(spectrum, axis=1, keepdims=True)
    spectrum_norm = spectrum / spectrum_sum
    entropies = -np.sum(spectrum_norm * np.log(spectrum_norm), axis=1) / np.log(spectrum.shape[1])

    return entropies