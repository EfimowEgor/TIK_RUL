import numpy as np
from init import signal

window_size = 256
new_size = 125

#Moving avg
moving_avg = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

block_size = moving_avg.shape[0] // new_size
moving_avg = np.array([moving_avg[i*block_size:(i+1)*block_size].mean() for i in range(new_size)])

#EMA
alpha = 2 / (window_size + 1)

ema = np.zeros(len(signal))
ema[0] = signal[0]
ema[1:] = alpha * signal[1:] + (1 - alpha) * np.cumsum(alpha * signal[:-1])

ema = np.array([ema[i*block_size:(i+1)*block_size].mean() for i in range(new_size)])

#Moving std
moving_std = np.array([np.std(signal[i:i+window_size]) for i in range(len(signal) - window_size + 1)])

block_size = moving_std.shape[0] // new_size
moving_std = np.array([moving_std[i*block_size:(i+1)*block_size].mean() for i in range(new_size)])