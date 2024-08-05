import scipy
from timeline_params import sampling_rate
import numpy as np

def fir(signal: np.array, low_pass: int):
    fir_coefficients = scipy.signal.firwin(101, 3000 / (sampling_rate / 2), pass_zero=False)
    return scipy.signal.lfilter(fir_coefficients, 1.0, signal)
