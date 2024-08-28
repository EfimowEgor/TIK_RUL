import scipy
import numpy as np

def butter(signal: np.array, low_cut: int, sr: int = 3e4):
    b, a = scipy.signal.butter(2, low_cut / (sr / 2), btype='high', analog=False)
    return scipy.signal.filtfilt(b, a, signal)