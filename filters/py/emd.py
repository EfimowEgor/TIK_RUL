from PyEMD import EMD
import numpy as np

def empirical_mode_decomposition(signal: np.array):
    emd = EMD()
    imfs = emd(signal)

    num_noise_imfs = 2
    emd_signal = np.sum(imfs[num_noise_imfs:], axis=0)

    return emd_signal
