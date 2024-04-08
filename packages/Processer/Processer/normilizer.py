from copy import deepcopy

import numpy as np
from sklearn.preprocessing import MinMaxScaler

def tkeo_operator(data, k = 1):
    """
    Teager-Kaiser Energy operator
    """
    npnts = len(data[0])
    nsignals = len(data)
    filt_data = deepcopy(data)
    for i in range(nsignals):
        for n in range(k, npnts-k):
            filt_data[i][n] = data[i][n]**2-data[i][n-1]*data[i][n+1]
    return filt_data

def normilize(signal: np.ndarray):
    """
    MinMaxScaler + Teager-Kaiser Operator + MinMaxScaler
    """
    # scalers = [MinMaxScaler, StandardScaler]
    scaler = MinMaxScaler(feature_range=(0, 1))
    signal = scaler.fit_transform(signal)
    print(f'norm1 max: {signal.max()}, min: {signal.min()}')
    signal = tkeo_operator(signal)
    print(f'tkeo max: {signal.max()}, min: {signal.min()}')
    signal = scaler.fit_transform(signal)
    print(f'norm2 max: {signal.max()}, min: {signal.min()}')
    return signal
