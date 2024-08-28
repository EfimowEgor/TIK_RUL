import numpy as np

def exponential_smoothing(signal: np.array):
    exponential_smoothing_signal = np.zeros_like(signal)

    exponential_smoothing_signal[0] = signal[0]
    alpha = 0.3
    for i in range(1, len(signal)):
        exponential_smoothing_signal[i] = alpha * signal[i] + (1 - alpha) * exponential_smoothing_signal[i - 1]

    return exponential_smoothing_signal
