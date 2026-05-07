"""
splitting.py — Train / validation / test split utilities.

Returns a Repeated Stratified K-Fold: `N_REPEATS` independent K-fold
passes each with a different random seed, giving `N_SPLITS * N_REPEATS`
total folds. Every fold uses the full dataset - there is no global held-out
test set. Each out-of-fold slice acts as the test split for that fold, so
every sample appears in exactly ``N_REPEATS`` test evaluations.

With `idx_val = None` the final probe in `solution.py` trains on all
`N` samples (every sample is in training for at least one fold, so the
`np.unique` union covers the full index range).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

N_SPLITS  = 6
N_REPEATS = 3


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    random_state: int = 67,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Build a repeated stratified K-fold split over the full dataset.

    Args:
        y:            Label array of shape `(N,)` with values in `{0, 1}`.
        df:           Unused; kept for API compatibility.
        random_state: Base seed; each repeat uses `random_state + repeat`.

    Returns:
        A list of `N_SPLITS * N_REPEATS` `(idx_train, None, idx_test)`
        tuples.  `idx_val` is always `None`.
    """
    idx_all = np.arange(len(y))
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for repeat in range(N_REPEATS):
        kf = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=random_state + repeat,
        )
        for idx_train, idx_test in kf.split(idx_all, y):
            splits.append((idx_train, None, idx_test))

    return splits
