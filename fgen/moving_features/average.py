import numpy as np

def average(y: np.array, window_size: int = 256) -> np.array:
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')