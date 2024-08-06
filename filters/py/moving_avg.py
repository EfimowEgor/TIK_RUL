import numpy as np

def moving_avg(signal: np.array):
    window_size = 10

    moving_avg_signal = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
    return moving_avg_signal
