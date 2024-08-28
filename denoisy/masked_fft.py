import numpy as np

def masked_fft(signal: np.array, low_cut: int, sr: int = 3e4):
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(spectrum), 1 / sr)

    mask = np.abs(freqs) > low_cut

    masked_spectrum = spectrum * mask
    return np.fft.ifft(masked_spectrum).real