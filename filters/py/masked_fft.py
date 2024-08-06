import numpy as np
from timeline_params import sampling_rate


def masked_fft(signal: np.array, low_cut: int):
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(spectrum), 1 / sampling_rate)

    mask = np.abs(freqs) > low_cut

    masked_spectrum = spectrum * mask
    return np.fft.ifft(masked_spectrum).real