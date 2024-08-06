import numpy as np

def rmse(y_true: np.array, y_pred: np.array) -> float:
    return np.sqrt(np.square(np.subtract(y_true, y_pred)).mean())