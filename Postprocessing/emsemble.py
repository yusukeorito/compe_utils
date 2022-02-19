"""
median ensembleとmean ensembleのいいとこどりのアルゴリズム
meanは外れ値の影響を受けやすく
medianは一つの予測値しか使えていない
"""

import numpy as np

def median_mean_ensemble(inputs:np.ndarray, axis:int):
    """
    inputs = ndarray of shape (n_samples, nfolds)
    """

    spread = inputs.max(axis=axis) - inputs.min(axis=axis)
    spread_threshold = 0.45
    print(f'Inliers: {(spread < spread_threshold).sum() :7} -> compute mean')
    print(f'Inliers: {(spread >= spread_threshold).sum() :7} -> compute median')
    return np.where(spread < spread_threshold, np.mean(inputs, axis=axis), np.median(input, axis=axis))


