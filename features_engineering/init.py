import pandas as pd
import numpy as np

SAMPLE_PATH: str = "../Выборка_Н22_5_1.csv"

df = pd.read_csv(SAMPLE_PATH, sep=';', usecols=[1], header=0)
df['Signals'] = df['Signals'].str.replace(',', '.').astype('float32')

signal: np.array = df['Signals'].values

mean_value = np.nanmean(signal)
signal = np.nan_to_num(signal, nan=mean_value)