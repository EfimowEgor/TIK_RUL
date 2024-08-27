import numpy as np

def std(y: np.array, sr: float = 30000, frame_size: int = 2048, hop_size: int = 512) -> np.array:
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_size)[::hop_size]

    spectrum = np.abs(np.fft.fft(frames, n=frame_size, axis=-1)[:, :frame_size // 2])

    stds = np.std(spectrum, axis=1)

    return stds