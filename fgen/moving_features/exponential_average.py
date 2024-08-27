import numpy as np

def exponential_average(y: np.array, window_size: int = 256) -> np.array:
    alpha = 2 / (window_size + 1)

    ema = np.zeros(len(y))
    ema[0] = y[0]

    ema[1:] = alpha * y[1:] + (1 - alpha) * np.cumsum(alpha * y[:-1])

    return ema