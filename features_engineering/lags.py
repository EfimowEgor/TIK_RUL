import numpy as np
from init import signal
from moving_features import new_size, block_size

def lag(arr, lag_value):
    result = np.empty_like(arr)
    result[:lag_value] = np.nan
    result[lag_value:] = arr[:-lag_value]
    return result

max_dif = 10

compressed_signal = np.array([signal[i*block_size:(i+1)*block_size].mean() for i in range(new_size)])
lags = np.zeros((max_dif, len(compressed_signal)))

for i in range(max_dif):
    lags[i] = lag(compressed_signal, i+1)

lag_diffs = np.zeros((max_dif, len(compressed_signal)))

for i in range(max_dif):
    lag_diffs[i] = compressed_signal - lags[i]