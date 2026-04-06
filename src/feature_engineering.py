"""
Feature engineering module.
Combines probabilistic BTYD models with RFM metrics to produce
a rich customer-level feature set with a rigorous temporal train/test split.

UPGRADED v2.4.0 (Round 6):
  LEAKAGE FIX — Monetary_Percentile:
    Previously computed via rank(pct=True) on ALL rfm customers before
    the 80/20 stratified split. This means test customers influenced each
    other's percentile ranks — a form of target leakage.

    Fix: Monetary_Percentile is now computed AFTER the 80/20 split:
      - Train: rank(pct=True) within train_idx customers only.
      - Test:  searchsorted against sorted train distribution → divide by n_train.
    Test customers' percentiles are now derived purely from the training
    distribution, matching real deployment where only historical customers
    define the reference distribution.

  All other logic unchanged from v2.3.0 (Round 5):
    - Single split_date anchor (FIX F-1)
    - BTYD fitted before capping (FIX F-2, F-3)
    - Stratification on log1p(y_raw) (FIX F-4)
    - Max_Single_Order computed on train_txns only
"""
import logging
import pandas as pd
import numpy as np
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from src.config import FEATURE_COLS, RANDOM_SEED

logger = logging.getLogger(__name__)

TRAIN_RATIO = 0.75


def build_hybrid_features(
    df: pd.DataFrame,
    raw_df: pd.DataFrame = None,
    split_days: int = 90,
):
    """
    Builds the full feature matrix using temporal split + log1p target transform.

    Returns
    -------
    X_train, X_test  : feature DataFrames (16 features each in v2.4.0)
    y_train          : log1p-transformed train targets (used for modeling)
    y_test           : log1p-transformed test targets  (used for modeling)
    y_test_raw       : original dollar-scale test targets (for business metrics)

    IMPORTANT: Returns 5 values. Cell 3 unpacks as:
        X_train, X_test, y_train, y_test, y_test_raw = build_hybrid_features(...)
    """
    logger.info("[3/8] Engineering Features — Temporal Split + Log1p Transform (v2.4.0)")

    # -----------------------------------------------------------------------
    # Stage 1: Temporal split — single anchor date
    # train_txns = everything before split_date
    # test_txns  = everything from split_date onward
    # Both windows anchor on the same date so T, Recency, and BTYD features
    # are all computed relative to the exact point where prediction begins.
    # -----------------------------------------------------------------------
    max_date   = df['InvoiceDate'].max()
    split_date = max_date - pd.Timedelta(days=split_days)

    logger.info(
        f"Temporal split — "
        f"Train window: {df['InvoiceDate'].min().date()} → {split_date.date()} | "
        f"Prediction window: {split_date.date()} → {max_date.date()} ({split_days} days)"
    )

    train_txns = df[df['InvoiceDate'] <  split_date].copy()
    test_txns  = df[df['InvoiceDate'] >= split_date].copy()

    train_customers = set(train_txns['Customer ID'].unique())
    logger.info(f"Customers with train history: {len(train_customers):,}")

    # -----------------------------------------------------------------------
    # Stage 2: RFM Summary (train window only)
    # observation_period_end = split_date so T and Recency are computed
    # relative to the same date the prediction window starts.
    # -----------------------------------------------------------------------
    rfm = summary_data_from_transaction_data(
        train_txns,
        customer_id_col='Customer ID',
        datetime_col='InvoiceDate',
        monetary_value_col='TotalAmount',
        observation_period_end=split_date,
    )

    rfm = rfm.rename(columns={
        'recency':        'Recency',
        'frequency':      'Frequency',
        'monetary_value': 'Monetary',
        'T':              'T',
    })

    # -----------------------------------------------------------------------
    # Stage 3: Behavioral Feature Engineering
    # NOTE: Monetary_Percentile is intentionally NOT computed here.
    # It is computed after the 80/20 split (Stage 5b) using only train_idx
    # customers to prevent leakage into test_idx percentile ranks.
    # -----------------------------------------------------------------------

    # 3a. Interpurchase time std
    def inter_std(dates: pd.Series) -> float:
        sorted_dates = dates.sort_values()
        return sorted_dates.diff().dt.days.std() if len(dates) > 1 else 0.0

    rfm['Interpurchase_Std'] = (
        train_txns.groupby('Customer ID')['InvoiceDate']
        .agg(inter_std)
        .fillna(0)
    )

    # 3b. Unique products
    rfm['Unique_Products'] = (
        train_txns.groupby('Customer ID')['StockCode']
        .nunique()
        .reindex(rfm.index, fill_value=0)
    )

    # 3c. Visit diversity
    rfm['Visit_Diversity'] = (
        train_txns.groupby('Customer ID')['InvoiceDate']
        .apply(lambda x: x.dt.date.nunique())
        .reindex(rfm.index, fill_value=0)
    )

    # 3d. Average basket size
    invoice_size = (
        train_txns.groupby(['Customer ID', 'Invoice'])['Quantity']
        .sum()
        .reset_index()
        .groupby('Customer ID')['Quantity']
        .mean()
    )
    rfm['Avg_Basket_Size'] = (
        invoice_size
        .reindex(rfm.index, fill_value=0)
        .clip(lower=0)
    )

    # 3e. Return rate
    if raw_df is not None:
        raw_filtered  = raw_df[raw_df['Customer ID'].isin(rfm.index)].copy()
        total_orders  = raw_filtered.groupby('Customer ID')['Invoice'].nunique()
        return_orders = (
            raw_filtered[raw_filtered['Quantity'] < 0]
            .groupby('Customer ID')['Invoice']
            .nunique()
        )
        return_rate = (return_orders / total_orders).fillna(0).clip(0, 1)
        rfm['Return_Rate'] = return_rate.reindex(rfm.index, fill_value=0)
    else:
        logger.warning("raw_df not provided — Return_Rate set to 0 for all customers.")
        rfm['Return_Rate'] = 0.0

    # 3f. Max_Single_Order — computed on train_txns only, no leakage
    max_order = (
        train_txns.groupby(['Customer ID', 'Invoice'])['TotalAmount']
        .sum()
        .reset_index()
        .groupby('Customer ID')['TotalAmount']
        .max()
    )
    rfm['Max_Single_Order'] = (
        max_order
        .reindex(rfm.index, fill_value=0)
        .clip(lower=0)
    )

    logger.info(
        f"Max_Single_Order — median=${rfm['Max_Single_Order'].median():.2f}, "
        f"max=${rfm['Max_Single_Order'].max():,.2f}"
    )

    # -----------------------------------------------------------------------
    # Stage 4: Target Variable (raw dollar scale)
    # -----------------------------------------------------------------------
    y = (
        test_txns.groupby('Customer ID')[['TotalAmount']]
        .sum()
        .rename(columns={'TotalAmount': 'Next_3M_Spend'})
    )

    full_data = rfm.join(y, how='left').fillna({'Next_3M_Spend': 0})
    full_data = full_data[full_data.index.isin(train_customers)]

    # Monetary_Percentile excluded here — added post-split in Stage 5b
    feature_base_cols = [
        'Recency', 'Frequency', 'Monetary', 'Interpurchase_Std', 'T',
        'Unique_Products', 'Visit_Diversity', 'Avg_Basket_Size', 'Return_Rate',
        'Max_Single_Order',
    ]
    X_raw = full_data[feature_base_cols]
    y_raw = full_data['Next_3M_Spend']

    # -----------------------------------------------------------------------
    # Stage 5: Stratified 80/20 Customer Split
    # Stratification on log1p(y_raw) prevents quintile collapse on the
    # zero-heavy raw distribution and ensures whale representation in both sets.
    # -----------------------------------------------------------------------
    try:
        spend_quintile = pd.qcut(
            np.log1p(y_raw), q=5, labels=False, duplicates='drop'
        ).fillna(0).astype(int)
    except Exception:
        spend_quintile = pd.cut(
            y_raw,
            bins=[-1, 0, y_raw[y_raw > 0].median(), y_raw.max() + 1],
            labels=[0, 1, 2]
        ).fillna(0).astype(int)

    rng       = np.random.default_rng(RANDOM_SEED)
    train_idx = []
    test_idx  = []

    for quintile_val in spend_quintile.unique():
        bucket  = spend_quintile[spend_quintile == quintile_val].index.tolist()
        rng.shuffle(bucket)
        n_train = int(len(bucket) * 0.80)
        train_idx.extend(bucket[:n_train])
        test_idx.extend(bucket[n_train:])

    X_train_raw = X_raw.loc[train_idx].copy()
    X_test_raw  = X_raw.loc[test_idx].copy()
    y_train_raw = y_raw.loc[train_idx]
    y_test_raw  = y_raw.loc[test_idx]

    logger.info(
        f"Stratified split — Train: {len(train_idx):,} | Test: {len(test_idx):,}"
    )

    train_whale_pct = (y_train_raw > y_raw.quantile(0.80)).mean()
    test_whale_pct  = (y_test_raw  > y_raw.quantile(0.80)).mean()
    zero_pct        = (y_test_raw == 0).mean()
    logger.info(
        f"Whale distribution check — "
        f"Train top-20%: {train_whale_pct:.1%} | "
        f"Test top-20%: {test_whale_pct:.1%} (should be ~equal)"
    )
    logger.info(
        f"y_test_raw — Mean: ${y_test_raw.mean():,.2f} | "
        f"Max: ${y_test_raw.max():,.2f} | Zero-spend: {zero_pct:.1%}"
    )

    # -----------------------------------------------------------------------
    # Stage 5b: Leakage-Free Monetary_Percentile
    # Computed AFTER the 80/20 split so test customers do not influence
    # train customers' percentile ranks.
    #
    # Train: standard rank(pct=True) within train_idx customers only.
    # Test:  searchsorted against sorted train distribution.
    #        np.searchsorted returns the number of train customers each test
    #        customer's Monetary EXCEEDS, divided by n_train → a percentile
    #        in [0.0, 1.0] anchored on the training distribution.
    #
    # In real deployment you only know the historical customer distribution
    # at scoring time — this matches that constraint exactly.
    # -----------------------------------------------------------------------
    train_monetary_vals = X_train_raw['Monetary'].values
    train_sorted        = np.sort(train_monetary_vals)
    n_train             = len(train_sorted)

    # Train percentile: rank within train set
    X_train_raw['Monetary_Percentile'] = (
        pd.Series(train_monetary_vals, index=X_train_raw.index)
        .rank(pct=True, method='average')
        .values
    )

    # Test percentile: position in train distribution
    X_test_raw['Monetary_Percentile'] = (
        np.searchsorted(train_sorted, X_test_raw['Monetary'].values, side='right')
        / n_train
    )

    logger.info(
        f"Leakage-free Monetary_Percentile — "
        f"Train mean: {X_train_raw['Monetary_Percentile'].mean():.3f} | "
        f"Test mean: {X_test_raw['Monetary_Percentile'].mean():.3f} | "
        f"Test max: {X_test_raw['Monetary_Percentile'].max():.3f}"
    )

    # -----------------------------------------------------------------------
    # Stage 6: Log1p Target Transform
    # -----------------------------------------------------------------------
    y_train = np.log1p(y_train_raw)
    y_test  = np.log1p(y_test_raw)

    logger.info(
        f"Log1p transform — "
        f"y_train: mean={y_train.mean():.3f}, max={y_train.max():.3f} | "
        f"y_test:  mean={y_test.mean():.3f},  max={y_test.max():.3f}"
    )

    # -----------------------------------------------------------------------
    # Stage 7: BTYD Feature Extraction (Train Set Only)
    # BGF and GGF are fitted on X_train_raw BEFORE any outlier capping
    # so whale customers receive correct probabilistic estimates.
    # -----------------------------------------------------------------------
    bgf = None

    try:
        bgf = BetaGeoFitter(penalizer_coef=0.01)
        bgf.fit(
            X_train_raw['Frequency'],
            X_train_raw['Recency'],
            X_train_raw['T'],
            tol=1e-5,
        )
        logger.info("BGF converged with penalizer_coef=0.01 (tol=1e-5)")
    except Exception as e:
        if 'converge' not in str(e).lower():
            raise
        logger.warning("BGF failed at penalizer=0.01. Escalating...")
        bgf = None

    if bgf is None:
        for penalizer in [0.05, 0.1, 0.5, 1.0, 5.0]:
            try:
                bgf = BetaGeoFitter(penalizer_coef=penalizer)
                bgf.fit(
                    X_train_raw['Frequency'],
                    X_train_raw['Recency'],
                    X_train_raw['T'],
                )
                logger.info(f"BGF converged with penalizer_coef={penalizer}")
                break
            except Exception as e:
                if 'converge' not in str(e).lower():
                    raise
                logger.warning(f"BGF failed with penalizer_coef={penalizer}. Retrying...")

    if bgf is None:
        raise RuntimeError(
            "BetaGeoFitter failed to converge across all penalizer values. "
            "Check training data for extreme outliers or insufficient customers."
        )

    train_gg = X_train_raw[
        (X_train_raw['Frequency'] > 0) & (X_train_raw['Monetary'] > 0)
    ].copy()
    train_gg['Frequency'] = train_gg['Frequency'].round().astype(int)

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(train_gg['Frequency'], train_gg['Monetary'])

    global_avg_monetary = train_gg['Monetary'].mean()

    def add_prob_features(data_split: pd.DataFrame) -> pd.DataFrame:
        """
        Applies fitted BTYD models and computes all derived features.
        Pure transform — BGF/GGF fitted on train only.
        Input data_split already contains Monetary_Percentile from Stage 5b.
        """
        data = data_split.copy()

        data['Prob_Pred_Txn'] = bgf.predict(
            split_days, data['Frequency'], data['Recency'], data['T']
        )
        data['Prob_Alive'] = bgf.conditional_probability_alive(
            data['Frequency'], data['Recency'], data['T']
        )

        data['Prob_Pred_Val'] = global_avg_monetary
        repeat_mask = data['Frequency'] > 0
        repeat_freq = data.loc[repeat_mask, 'Frequency'].round().astype(int)
        data.loc[repeat_mask, 'Prob_Pred_Val'] = (
            ggf.conditional_expected_average_profit(
                repeat_freq,
                data.loc[repeat_mask, 'Monetary'],
            )
        )

        data['Purchase_Rate'] = (
            data['Frequency'] / data['T'].clip(lower=1)
        ).fillna(0)

        data['Days_Since_Purchase'] = (data['T'] - data['Recency']).clip(lower=0)

        data['Revenue_Per_Day'] = (
            data['Monetary'] / data['T'].clip(lower=1)
        ).fillna(0)

        return data

    # BTYD features computed on uncapped data (includes Monetary_Percentile already)
    X_train_featured = add_prob_features(X_train_raw)
    X_test_featured  = add_prob_features(X_test_raw)

    # -----------------------------------------------------------------------
    # Stage 8: Outlier Capping — train-derived caps applied after BTYD
    # Monetary_Percentile is already in [0.0, 1.0] — no capping needed.
    # -----------------------------------------------------------------------
    freq_cap       = X_train_raw['Frequency'].quantile(0.99)
    monetary_cap   = X_train_raw['Monetary'].quantile(0.99)
    basket_cap     = X_train_raw['Avg_Basket_Size'].quantile(0.99)
    max_order_cap  = X_train_raw['Max_Single_Order'].quantile(0.99)

    X_train_featured = X_train_featured.copy()
    X_test_featured  = X_test_featured.copy()

    for X in [X_train_featured, X_test_featured]:
        X['Frequency']        = X['Frequency'].clip(upper=freq_cap).round().astype(int)
        X['Monetary']         = X['Monetary'].clip(upper=monetary_cap)
        X['Avg_Basket_Size']  = X['Avg_Basket_Size'].clip(upper=basket_cap)
        X['Max_Single_Order'] = X['Max_Single_Order'].clip(upper=max_order_cap)

    X_train = X_train_featured[FEATURE_COLS]
    X_test  = X_test_featured[FEATURE_COLS]

    logger.info(
        f"Feature engineering complete — "
        f"Train: {X_train.shape} | Test: {X_test.shape}"
    )

    return X_train, X_test, y_train, y_test, y_test_raw