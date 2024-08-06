from scipy.signal import savgol_filter
import numpy as np

def savgol(signal: np.array):
    window_size = 11
    poly_order = 2
    savgol_signal = savgol_filter(signal, window_size, poly_order)

    return savgol_signal
