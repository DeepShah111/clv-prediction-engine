"""Cross-validation helpers shared by benchmarking and tuning."""
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.config import RANDOM_SEED


def make_cv_folds(X_train, y_train):
    """
    One set of shuffled, stratified (zero vs non-zero spend) folds, used for EVERY
    model, so scores are comparable and not biased by the row order of X_train.
    Returns a list of (train_idx, validation_idx) pairs.
    """
    n_splits = min(5, max(3, len(X_train) // 10))
    y_binary = (np.asarray(y_train) > 0).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    return list(skf.split(X_train, y_binary))