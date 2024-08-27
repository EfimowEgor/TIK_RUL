import numpy as np

def lag(y: np.array, lag_value: int) -> np.array:
    result = np.empty_like(y)
    result[:lag_value] = np.nan
    result[lag_value:] = y[:-lag_value]
    return result