"""Pure helpers for anomaly scoring and flags (no heavy imports, easy to test)."""
import numpy as np
import pandas as pd


def apply_minmax(values, vmin, vmax):
    """Rescale with the TRAINING min/max (never the batch's own), clipped to [0, 1]."""
    if vmin is None or vmax is None or vmax <= vmin:
        raise ValueError(
            "Missing or invalid normalisation values in the anomaly bundle; "
            "re-run run_anomaly_pipeline."
        )
    values = np.asarray(values, dtype=float)
    return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)


def add_business_flags(df, combined_score,
                       anomaly_threshold=0.50,
                       return_rate_threshold=0.30,
                       high_return_score_min=0.40,
                       recent_days=30):
    """Adds is_anomaly, is_high_return, is_whale_anomaly and is_suspicious."""
    df = df.copy()
    score = np.asarray(combined_score, dtype=float)

    df['is_anomaly'] = score > anomaly_threshold

    if 'Return_Rate' in df.columns:
        df['is_high_return'] = (df['Return_Rate'] > return_rate_threshold) & (score > high_return_score_min)
    else:
        df['is_high_return'] = False

    # Monetary_Percentile is a fraction between 0 and 1, so "top decile" means > 0.90
    if 'Monetary_Percentile' in df.columns:
        df['is_whale_anomaly'] = (df['Monetary_Percentile'] > 0.90) & df['is_anomaly']
    else:
        df['is_whale_anomaly'] = False

    # "Recently active" means few days since the LAST purchase
    if 'Days_Since_Purchase' in df.columns:
        df['is_suspicious'] = df['is_high_return'] & df['is_anomaly'] & (df['Days_Since_Purchase'] < recent_days)
    else:
        df['is_suspicious'] = False

    return df