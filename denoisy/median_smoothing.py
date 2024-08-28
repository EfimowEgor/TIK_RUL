import numpy as np

def median_smoothing(signal: np.array):
    median_smoothing_signal = np.zeros_like(signal)

    window_size = 10
    half_window = window_size // 2
    for i in range(len(signal)):
        start = max(0, i - half_window)
        end = min(len(signal), i + half_window + 1)

        median_smoothing_signal[i] = np.median(signal[start:end])

    return median_smoothing_signal
