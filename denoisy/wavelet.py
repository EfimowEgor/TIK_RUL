import pywt
import numpy as np

def wavelet(signal: np.array, wav_type: str = 'db4'):
    db = pywt.Wavelet(wav_type)
    max_level = pywt.dwt_max_level(len(signal), db)

    coeffs = pywt.wavedec(signal, db, level=max_level)
    coeffs[1:] = [np.zeros_like(coeff) for coeff in coeffs[1:]]

    return pywt.waverec(coeffs, db)
