import numpy as np
from fgen.lags import lag

def lag_diff(y: np.array, lag_value: int) -> np.array:
    return y - lag(y, lag_value)
