from hmmlearn import hmm
from timeline_params import n_samples
import numpy as np
from synth_signal import signal

MAX_STATES = 10

def best_hmm_models(signal: np.array):
    aic_values = []
    bic_values = []
    models = []

    for n_components in range(1, MAX_STATES + 1):
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100, random_state=42)
        model.fit(signal.reshape(-1, 1))
        log_likelihood = model.score(signal.reshape(-1, 1))

        # Вычисление AIC и BIC
        num_params = n_components ** 2 + 2 * n_components - 1  # число параметров модели
        aic = 2 * num_params - 2 * log_likelihood
        bic = num_params * np.log(n_samples) - 2 * log_likelihood

        aic_values.append(aic)
        bic_values.append(bic)
        models.append(model)

    return models[np.argmin(aic_values)], models[np.argmin(bic_values)]

best_aic_model, best_bic_model = best_hmm_models(signal)
aic_signal = best_aic_model.predict(signal.reshape(-1, 1))
bic_signal = best_bic_model.predict(signal.reshape(-1, 1))
