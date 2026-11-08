import numpy as np
import pandas as pd
from src.cv_utils import make_cv_folds


def test_folds_cover_all_rows_and_are_stratified():
    # rows ordered by group, like the real training set: 120 zeros first, then 80 spenders
    y = pd.Series(np.r_[np.zeros(120), np.linspace(1, 6, 80)])
    X = pd.DataFrame({'a': np.arange(200)})

    folds = make_cv_folds(X, y)

    assert len(folds) == 5
    all_val = np.concatenate([val for _, val in folds])
    assert sorted(all_val) == list(range(200))            # every row validated exactly once
    for _, val in folds:
        zero_share = (y.iloc[val] == 0).mean()
        assert abs(zero_share - 0.6) < 0.05               # every fold mirrors the 60% zeros