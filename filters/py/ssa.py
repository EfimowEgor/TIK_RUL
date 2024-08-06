from timeline_params import n_samples
import numpy as np

def conv_signal(signal: np.array, conv_length: int = 100):
    window_size = n_samples // conv_length
    return np.array([np.mean(signal[i * window_size:(i + 1) * window_size]) for i in range(conv_length)])

def ssa(conv_signal: np.array):
    N = len(conv_signal)
    L = 20
    K = N - L + 1

    X = np.column_stack([conv_signal[i:i + L] for i in range(0, K)])

    d = np.linalg.matrix_rank(X)
    U, Sigma, VT = np.linalg.svd(X)

    main_comps = np.arange(0, 5)

    k = U.shape[1]
    n = U.shape[0] + VT.shape[1] - 1
    recon = np.zeros(n)
    counts = np.zeros(n)

    for i in main_comps:
        component_matrix = Sigma[i] * np.outer(U[:, i], VT[i, :])
        for j in range(component_matrix.shape[1]):
            recon[j:j + component_matrix.shape[0]] += component_matrix[:, j]
            counts[j:j + component_matrix.shape[0]] += 1

    return recon / counts