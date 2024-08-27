import numpy as np

def std(y: np.array, window_size: int = 256) -> np.array:
    return np.array([np.std(y[i:i+window_size]) for i in range(len(y) - window_size + 1)])