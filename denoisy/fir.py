import scipy
import numpy as np

def fir(signal: np.array, low_pass: int, sr: int = 3e4):
    fir_coefficients = scipy.signal.firwin(101, low_pass / (sr / 2), pass_zero=False)
    return scipy.signal.lfilter(fir_coefficients, 1.0, signal)
