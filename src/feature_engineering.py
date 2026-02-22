"""
Feature engineering module combining Probabilistic Models (BTYD) + RFM.
"""
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.model_selection import train_test_split
from src.config import logger, FEATURE_COLS

def build_hybrid_features(df: pd.DataFrame, split_days: int = 90):
    logger.info("[3/8] Engineering Features")
    
    # Temporal Split
    max_date = df['InvoiceDate'].max()
    split_date = max_date - pd.Timedelta(days=split_days)
    past = df[df['InvoiceDate'] < split_date]
    future = df[df['InvoiceDate'] >= split_date]
    
    # RFM Calculation
    rfm = summary_data_from_transaction_data(
        past, 'Customer ID', 'InvoiceDate', monetary_value_col='TotalAmount', observation_period_end=split_date
    )
    rfm = rfm[rfm['monetary_value'] > 0]
    rfm['frequency'] = rfm['frequency'].clip(upper=rfm['frequency'].quantile(0.99))
    
    # Interpurchase Time Std Dev
    def inter_std(x): return x.diff().dt.days.std() if len(x) > 1 else 0
    rfm['Interpurchase_Std'] = past.groupby('Customer ID')['InvoiceDate'].agg(inter_std).fillna(0)

    # Target Generation
    y = future.groupby('Customer ID')[['TotalAmount']].sum().rename(columns={'TotalAmount': 'Next_3M_Spend'})
    
    rfm.rename(columns={'recency': 'Recency', 'frequency': 'Frequency', 'monetary_value': 'Monetary'}, inplace=True)
    full_data = rfm.join(y, how='left').fillna(0)
    
    target_cap = full_data['Next_3M_Spend'].quantile(0.99)
    full_data['Next_3M_Spend'] = full_data['Next_3M_Spend'].clip(lower=0, upper=target_cap)
    
    X_cols = ['Recency', 'Frequency', 'Monetary', 'Interpurchase_Std', 'T']
    X_raw = full_data[X_cols]
    y_raw = full_data['Next_3M_Spend']
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )
    
    # Fitting Probabilistic Models (Train Set Only)
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(X_train_raw['Frequency'], X_train_raw['Recency'], X_train_raw['T'])
    
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    train_gg = X_train_raw[X_train_raw['Frequency'] > 0]
    ggf.fit(train_gg['Frequency'], train_gg['Monetary'])
    
    def add_prob_features(data_split):
        data = data_split.copy()
        data['Prob_Pred_Txn'] = bgf.predict(split_days, data['Frequency'], data['Recency'], data['T'])
        data['Prob_Pred_Val'] = ggf.conditional_expected_average_profit(data['Frequency'], data['Monetary'])
        return data

    X_train = add_prob_features(X_train_raw)[FEATURE_COLS]
    X_test = add_prob_features(X_test_raw)[FEATURE_COLS]
    
    logger.info(f"Feature Engineering Complete. Train Size: {len(X_train)}")
    
    return X_train, X_test, y_train, y_test