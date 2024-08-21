import pandas as pd
from centroid import centroid_descriptor
from spread import spread_descriptor
from skewness import skewness_descriptor
from kurtosis import kurtosis_descriptor
from entropy import entropy_descriptor
from flatness import flatness_descriptor
from crest import crest_descriptor
from flux import flux_descriptor
from slope import slope_descriptor
from mean import mean_descriptor
from std import std_descriptor
from power import pow_descriptor
from moving_features import ema, moving_std, moving_avg
from lags import lags, lag_diffs
from peaks import peak_freqs
from band_powers import band_pows

def make_dataset(make_csv: bool = False):
    pipeline = {
        'centoid': centroid_descriptor,
        'spread': spread_descriptor,
        'skewness': skewness_descriptor,
        'kurtosis': kurtosis_descriptor,
        'entropy': entropy_descriptor,
        'flatness': flatness_descriptor,
        'crest': crest_descriptor,
        'flux': flux_descriptor,
        'slope': slope_descriptor,
        'mean': mean_descriptor,
        'std': std_descriptor,
        'moving_avg': moving_avg,
        'moving_std': moving_std,
        'ema': ema,
        **{f'lag_{i+1}': lags[i] for i in range(10)},
        **{f'lag_diff_{i+1}': lag_diffs[i] for i in range(10)},
        'powers': pow_descriptor,
        **{f'band_pow_{i*1000}_{(i+1)*1000}': band_pows[i] for i in range(30)},
        'peak_freqs': peak_freqs
    }

    normalize = lambda arr: arr if arr.max() == arr.min() == 0 else (arr - arr.min()) / (arr.max() - arr.min())
    pipeline = {key: normalize(value) for key, value in pipeline.items()}

    if make_csv:
        dataset = pd.DataFrame(pipeline)
        dataset.to_csv('dataset.csv', index=False)
        return dataset

    return pipeline