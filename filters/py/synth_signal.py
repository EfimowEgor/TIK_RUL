from timeline_params import t
import numpy as np

np.random.seed(100)

def signal_func(t: np.array, num: int = 13) -> np.array:
    freqs: np.array = np.random.normal(4000, 1000, num)
    ampls: np.array = np.random.normal(5.0, 2.5, num)
    phases: np.array = np.random.uniform(0, 2*np.pi, num)

    clear_signal = np.zeros_like(t)

    for ampl, freq, phase in zip(ampls, freqs, phases):
        clear_signal += ampl * np.sin(2 * np.pi * freq * t + phase)

    additional_freqs: np.array = np.concatenate((np.random.uniform(10, 3000, 50),
                                                np.random.uniform(5000, 15000, 50)))
    additional_ampls: np.array = np.random.uniform(0.01, 0.5, 100)

    for freq, ampl in zip(additional_freqs, additional_ampls):
        clear_signal += ampl * np.sin(2 * np.pi * freq * t)

    noise = np.random.normal(0, np.random.uniform(0.1, 0.4), len(t))
    noisy_signal = clear_signal + noise

    return noisy_signal

signal = signal_func(t)
