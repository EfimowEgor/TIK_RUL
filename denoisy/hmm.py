from hmmlearn import hmm
import numpy as np

def best_hmm_models(signal: np.array, max_states: int = 10):
    aic_values = []
    bic_values = []
    models = []

    for n_components in range(1, max_states + 1):
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100, random_state=42)
        model.fit(signal.reshape(-1, 1))
        log_likelihood = model.score(signal.reshape(-1, 1))

        num_params = n_components ** 2 + 2 * n_components - 1
        aic = 2 * num_params - 2 * log_likelihood
        bic = num_params * np.log(len(signal)) - 2 * log_likelihood

        aic_values.append(aic)
        bic_values.append(bic)
        models.append(model)

    return models[np.argmin(aic_values)], models[np.argmin(bic_values)]
