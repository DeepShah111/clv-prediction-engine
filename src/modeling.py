import pandas as pd
import numpy as np
import warnings
from typing import Optional

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, TweedieRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.config import logger

try:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    ADVANCED_BOOSTING_AVAILABLE = True
except ImportError:
    ADVANCED_BOOSTING_AVAILABLE = False
    warnings.warn("Advanced Boosting libraries not found. Using standard sklearn.")

class NaiveBaseline(BaseEstimator, RegressorMixin):
    """
    Baseline Heuristic Model.
    Predicts future spend will be identical to the customer's historical average.
    """
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'NaiveBaseline':
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X['Monetary'].values

def train_and_benchmark(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    #Trains models and ranks by RMSE.
    logger.info("[6/8] Benchmarking Model Zoo...")
    models = {
        "Naive Baseline": NaiveBaseline(),
        "Linear Reg": Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
        "Ridge (L2)": Pipeline([('scaler', StandardScaler()), ('model', Ridge())]),
        "ElasticNet": Pipeline([('scaler', StandardScaler()), ('model', ElasticNet())]),
        "Tweedie Reg": Pipeline([('scaler', StandardScaler()), ('model', TweedieRegressor(power=1.5, link='log'))]),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    }
    
    if ADVANCED_BOOSTING_AVAILABLE:
        models["XGBoost (Tweedie)"] = XGBRegressor(objective='reg:tweedie', tweedie_variance_power=1.5, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        models["LightGBM"] = LGBMRegressor(objective='tweedie', tweedie_variance_power=1.5, n_estimators=100, verbose=-1, random_state=42)
        models["CatBoost"] = CatBoostRegressor(loss_function='Tweedie:variance_power=1.5', n_estimators=100, verbose=0, random_state=42)

    results = []
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            results.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Object': model})
        except Exception as e:
            logger.warning(f"Failed to train {name}: {e}")

    leaderboard_df = pd.DataFrame(results).sort_values(by='RMSE')
    print("\n" + "="*60 + "\n   MASTER MODEL LEADERBOARD (Test Set)\n" + "="*60)
    print(leaderboard_df[['Model', 'RMSE', 'MAE', 'R2']].to_string(index=False))
    
    best_ml_model = leaderboard_df.iloc[0]['Object']
    champion_name = leaderboard_df.iloc[0]['Model']
    
    return best_ml_model, champion_name, leaderboard_df

def tune_champion_model(best_ml_model, champion_name: str, X_train: pd.DataFrame, y_train: pd.Series):
    logger.info(f"[7/8] Tuning Champion: {champion_name}...")
    
    param_grids = {
        "Random Forest": {'n_estimators': [50, 100], 'max_depth': [10, None]},
        "XGBoost": {'n_estimators': [100], 'learning_rate': [0.05, 0.1]}
    }
    grid_params = None
    for key in param_grids:
        if key in champion_name:
            grid_params = param_grids[key]
            break
    
    if grid_params:
        grid = GridSearchCV(best_ml_model, grid_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        logger.info("   ✓ Tuning Complete.")
        return grid.best_estimator_
    
    logger.info(" No tuning parameters defined for this model. Skipping.")
    return best_ml_model