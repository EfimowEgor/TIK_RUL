import scipy
import numpy as np
from timeline_params import sampling_rate

def butter(signal: np.array, low_cut: int):
    b, a = scipy.signal.butter(2, low_cut / (sampling_rate / 2), btype='high', analog=False)
    return scipy.signal.filtfilt(b, a, signal)