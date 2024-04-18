import logging
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Log everything
# ----------
logger = logging.getLogger(__name__)

# NORMALIZATION
# ----------
def tkeo_operator(data, k = 1):
    """
    Remove noise using Teager-Kaiser energy operator
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
    signal = tkeo_operator(signal)
    signal = scaler.fit_transform(signal)

    return signal

# SQUASH DATA
# ----------
def compress(signals: pd.DataFrame, floor: str='30min', method='max'):
    """
    floor: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    method: Определяет каким образом сжимается ряд. Принимает значения 'max' - максимум интервала, 'mean' - среднее значение интервала
    """
    match method:
        case 'max':
            return signals.groupby(signals.date.dt.floor(floor)).max().drop('date', axis=1)
        case 'mean':
            return signals.groupby(signals.date.dt.floor(floor)).mean().drop('date', axis=1)
        case 'mixed':
            pass
            # do i even need this case
        case _:
            raise ValueError(f'Unknown method: {method}')
