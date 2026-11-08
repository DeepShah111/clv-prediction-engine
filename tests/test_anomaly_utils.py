import numpy as np
import pandas as pd
import pytest
from src.anomaly_utils import apply_minmax, add_business_flags


def test_whale_flag_uses_0_to_1_percentile():
    df = pd.DataFrame({'Return_Rate': [0.0], 'Monetary_Percentile': [0.95], 'Days_Since_Purchase': [10.0]})
    out = add_business_flags(df, [0.9])
    assert bool(out['is_whale_anomaly'].iloc[0])


def test_suspicious_needs_high_return_anomaly_and_recent_purchase():
    df = pd.DataFrame({'Return_Rate': [0.5, 0.5], 'Monetary_Percentile': [0.5, 0.5],
                       'Days_Since_Purchase': [10.0, 100.0]})
    out = add_business_flags(df, [0.9, 0.9])
    assert out['is_suspicious'].tolist() == [True, False]


def test_minmax_uses_training_range_not_the_batch():
    assert apply_minmax([0.73], 0.0, 1.0)[0] == pytest.approx(0.73)   # a lone customer is not forced to 0
    assert apply_minmax([2.0, -1.0], 0.0, 1.0).tolist() == [1.0, 0.0]


def test_minmax_rejects_missing_training_values():
    with pytest.raises(ValueError):
        apply_minmax([0.5], None, None)