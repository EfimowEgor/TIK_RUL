from scipy.signal import savgol_filter
import numpy as np

def savgol(signal: np.array, window_size: int, p: int = 2):
    savgol_signal = savgol_filter(signal, window_size, p)

    return savgol_signal
