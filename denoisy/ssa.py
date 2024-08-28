import numpy as np

def ssa(conv_signal: np.array, comps_amount: int = 5) -> np.array:
    n = len(conv_signal)
    l = 20
    k = n - l + 1

    x = np.column_stack([conv_signal[i:i + l] for i in range(0, k)])

    d = np.linalg.matrix_rank(x)
    u, sigma, vt = np.linalg.svd(x)

    main_comps = np.arange(0, comps_amount)

    k = u.shape[1]
    n = u.shape[0] + vt.shape[1] - 1
    recon = np.zeros(n)
    counts = np.zeros(n)

    for i in main_comps:
        component_matrix = sigma[i] * np.outer(u[:, i], vt[i, :])
        for j in range(component_matrix.shape[1]):
            recon[j:j + component_matrix.shape[0]] += component_matrix[:, j]
            counts[j:j + component_matrix.shape[0]] += 1

    return recon / counts