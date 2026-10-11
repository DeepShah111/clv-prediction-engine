import pandas as pd
import pytest
from src.feature_utils import _compute_return_rate


def _raw():
    return pd.DataFrame({
        'Invoice':     ['1', '2', 'C3', '4', '5', 'C6'],
        'Customer ID': [100.0, 100.0, 100.0, 200.0, 200.0, 200.0],   # float IDs, as in the raw file
        'InvoiceDate': ['2010-01-01', '2010-01-05', '2010-01-06',
                        '2010-01-02', '2010-01-03', '2010-03-01'],
        'Quantity':    [2, 1, -1, 3, 1, -2],
    })


def test_return_rate_matches_ids_and_uses_only_training_window():
    idx = pd.Index(['100', '200'])                       # text IDs, as in rfm.index
    rr = _compute_return_rate(_raw(), idx, pd.Timestamp('2010-02-01'))
    assert rr['100'] == pytest.approx(1 / 3)             # 1 return invoice out of 3
    assert rr['200'] == 0.0                               # its only return is after the split date