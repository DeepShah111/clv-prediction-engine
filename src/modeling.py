"""
Model training, benchmarking, and hyperparameter tuning module.

UPGRADED v2.5.0 (Round 6):
  1. UPPER CLIP — log predictions clipped at a_max=12.0 (≈ $162K) in both
     train_and_benchmark and tune_champion_model. Prevents linear models
     from producing astronomic Dollar_MAE/R² values (Ridge was −13,812).

  2. CALIBRATED STAGE 1 CLASSIFIER — TwoStageRegressor.fit() now applies
     isotonic calibration (CalibratedClassifierCV, cv='prefit') to the
     Stage 1 classifier after fitting. Corrects RF/XGB probability
     overconfidence that caused zero-spend avg prediction of $103.

  3. STRATIFIEDKFOLD IN TUNING — tune_champion_model() now uses the same
     StratifiedKFold approach as train_and_benchmark() for TwoStage models.
     Pre-tuning CV and GridSearchCV both use stratified folds.

  4. CATBOOST — CatBoostRegressor and Two-Stage (CatBoost) added to model
     zoo. Graceful fallback if catboost not installed.

  5. MONOTONE CONSTRAINTS — XGBoost and LightGBM regressors now enforce
     domain-consistent monotone relationships (e.g., higher Monetary must
     not decrease predicted spend). Improves generalization on whale segment.

  6. WEIGHTED ENSEMBLE — WeightedEnsemble class added. After main benchmark
     loop, top-3 models by Dollar_R² are combined with inverse-MAE weights.
     Ensemble appears in leaderboard but is excluded from champion selection
     (no independent CV score).

  NEW v2.6.0 (Upgrade Round 7):
  7. MLFLOW TRACKING — All 14 model CV scores, params, Dollar R², MAE
     logged as nested MLflow runs under experiment 'CLV_Pipeline_v2.5.0'.
     Champion model logged as MLflow artifact via log_champion_to_mlflow().
     Graceful no-op if mlflow not installed.

  ALL v2.4.0 LOGIC RETAINED:
    - Dollar R² eligibility gate (MIN_ELIGIBLE_DOLLAR_R2 = 0.10)
    - CHURN_THRESHOLD = 0.50 with $10 floor
    - StratifiedKFold pre-computed folds for TwoStage CV
    - SELECTION_INELIGIBLE set
"""
import warnings
import logging
import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.utils.validation import check_is_fitted
from sklearn.frozen import FrozenEstimator

from src.config import RANDOM_SEED, FEATURE_COLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional boosting libraries
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBRegressor, XGBClassifier
    from lightgbm import LGBMRegressor, LGBMClassifier
    ADVANCED_BOOSTING_AVAILABLE = True
except ImportError:
    ADVANCED_BOOSTING_AVAILABLE = False
    warnings.warn("XGBoost / LightGBM not found. Install via requirements.txt.")

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.info("CatBoost not installed — Two-Stage (CatBoost) skipped.")

# ---------------------------------------------------------------------------
# Optional MLflow — graceful degradation if not installed
# ---------------------------------------------------------------------------
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
    logger.info("MLflow detected — experiment tracking enabled.")
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.info("mlflow not installed — tracking skipped. Run: pip install mlflow")

MLFLOW_EXPERIMENT = "CLV_Pipeline_v2.5.0"
if MLFLOW_AVAILABLE:
    mlflow.set_tracking_uri("file:///content/drive/MyDrive/clv-prediction-engine/mlruns")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_TRAIN_SAMPLES      = 50
CV_SCORING             = 'neg_mean_absolute_error'
MIN_ELIGIBLE_DOLLAR_R2 = 0.10
CHURN_THRESHOLD        = 0.50

# Models excluded from champion selection — leaderboard reference only
SELECTION_INELIGIBLE = {'Naive Baseline (Mean)', 'BTYD Statistical Baseline'}

# Weighted Ensemble has no independent CV — excluded from CV-based selection.
# It can still beat the tuned champion post-hoc (checked in main_execution.ipynb).
ENSEMBLE_NAME = 'Weighted Ensemble'

# ---------------------------------------------------------------------------
# LOG PREDICTION UPPER CLIP
# expm1(12.0) ≈ $162,754 — safely above the dataset max of ~$26K.
# Prevents a single rogue linear prediction from collapsing Dollar_R².
# Applied consistently in every predict → expm1 pipeline.
# ---------------------------------------------------------------------------
LOG_PRED_MAX = 12.0

# ---------------------------------------------------------------------------
# Monotone constraints for XGBoost / LightGBM regressors.
# Order MUST match FEATURE_COLS exactly.
# +1 = monotone positive, -1 = monotone negative, 0 = unconstrained.
# ---------------------------------------------------------------------------
_LGBM_MONOTONE = [0, 1, 1, 1, 1, 1, 0, 1, -1, 1, 1, 1, 0, -1, 1, 1]

_XGB_MONOTONE = {
    'Recency': 0, 'Frequency': 1, 'Monetary': 1,
    'Prob_Pred_Txn': 1, 'Prob_Pred_Val': 1, 'Prob_Alive': 1,
    'Interpurchase_Std': 0, 'Purchase_Rate': 1,
    'Days_Since_Purchase': -1, 'Revenue_Per_Day': 1,
    'Unique_Products': 1, 'Visit_Diversity': 1, 'Avg_Basket_Size': 0,
    'Return_Rate': -1, 'Monetary_Percentile': 1, 'Max_Single_Order': 1,
}


# ===========================================================================
# Custom Estimators
# ===========================================================================

class NaiveBaselineMean(BaseEstimator, RegressorMixin):
    """Naive baseline — predicts training mean. SELECTION INELIGIBLE."""
    def fit(self, X, y):
        self.mean_prediction_ = float(np.asarray(y).mean())
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_prediction_)


class BTYDBaseline(BaseEstimator, RegressorMixin):
    """
    Probabilistic BTYD baseline.
    Predicts: log1p(E[transactions] × E[order value]).
    SELECTION INELIGIBLE.
    """
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        raw_pred = (X['Prob_Pred_Txn'] * X['Prob_Pred_Val']).values
        return np.log1p(np.clip(raw_pred, 0, None))


class TwoStageRegressor(BaseEstimator, RegressorMixin):
    """
    Hurdle model for zero-inflated CLV distributions.

    Stage 1 — Classifier: P(spend > $0)
    Stage 2 — Regressor:  E[log1p(spend) | spend > $0]

    v2.4.0: Combines stages in DOLLAR-space (mathematically correct):
        E[spend] = P(spend>0) × E[spend | spend>0]

    v2.5.0 NEW: Isotonic calibration applied to Stage 1 classifier inside
    fit() using cv='prefit'. Corrects RF/XGB probability overconfidence
    so that CHURN_THRESHOLD=0.50 actually fires on genuinely uncertain
    customers rather than under-predicted churners.
    """
    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor  = regressor

    def fit(self, X, y):
        y_arr    = np.asarray(y)
        y_binary = (y_arr > 0).astype(int)

        # Stage 1: fit classifier then calibrate with isotonic regression
        self.classifier_ = clone(self.classifier)
        self.classifier_.fit(X, y_binary)

        try:
            calibrated = CalibratedClassifierCV(
                FrozenEstimator(self.classifier_), method='isotonic'
            )
            calibrated.fit(X, y_binary)
            self.classifier_ = calibrated
            self._calibrated = True
        except Exception as cal_err:
            logger.warning(
                f"TwoStageRegressor: classifier calibration failed — {cal_err}. "
                "Using uncalibrated probabilities."
            )
            self._calibrated = False

        # Stage 2: fit regressor on spenders only
        positive_mask = y_arr > 0
        n_positive    = int(positive_mask.sum())

        if n_positive >= 5:
            self.regressor_      = clone(self.regressor)
            self.regressor_.fit(X[positive_mask], y_arr[positive_mask])
            self.fallback_spend_ = None
        else:
            self.regressor_      = None
            self.fallback_spend_ = (
                float(y_arr[positive_mask].mean()) if n_positive > 0 else 0.0
            )

        return self

    def predict(self, X):
        prob_spend = self.classifier_.predict_proba(X)[:, 1]

        if self.regressor_ is None:
            expected_log_amount = np.full(len(prob_spend), self.fallback_spend_)
        else:
            expected_log_amount = np.clip(self.regressor_.predict(X), 0, LOG_PRED_MAX)

        dollar_amount = np.expm1(expected_log_amount)
        dollar_result = prob_spend * dollar_amount

        dollar_result[prob_spend < CHURN_THRESHOLD] = 0.0
        dollar_result[dollar_result < 10.0] = 0.0

        return np.log1p(dollar_result)


class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """
    Inverse-Dollar_MAE weighted average of top-k fitted models.

    Models are already fitted — ensemble only combines their predictions.
    Weights are inversely proportional to each model's Dollar_MAE so
    lower-error models contribute more to the combined prediction.

    NOTE: No independent CV score exists for the ensemble since it requires
    all base models to be fitted first. It is evaluated on the test set only
    and is EXCLUDED from champion selection (no CV_MAE_Mean to compare on).
    """
    def __init__(self, fitted_models: list, weights: np.ndarray, model_names: list):
        self.fitted_models = fitted_models
        self.weights       = weights
        self.model_names   = model_names

    def fit(self, X, y):
        return self

    def predict(self, X):
        preds = np.stack([m.predict(X) for m in self.fitted_models], axis=0)
        return np.average(preds, axis=0, weights=self.weights)

    def get_params(self, deep=True):
        return {
            'fitted_models': self.fitted_models,
            'weights':       self.weights,
            'model_names':   self.model_names,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ===========================================================================
# MLflow Helpers  [NEW v2.6.0]
# ===========================================================================

def _log_model_to_mlflow(
    name: str,
    params: dict,
    cv_mae_mean: float,
    cv_mae_std: float,
    log_mae: float,
    log_r2: float,
    dollar_mae: float,
    dollar_r2: float,
    smape: float,
    wape: float,
) -> None:
    """
    Logs a single model's metrics and params to MLflow as a nested run.
    Called once per model inside train_and_benchmark(). No-op if mlflow absent.
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        with mlflow.start_run(run_name=name, nested=True):
            # Flatten params to strings — nested dicts break mlflow.log_params
            safe_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(safe_params)

            mlflow.log_metrics({
                "cv_mae_mean": cv_mae_mean,
                "cv_mae_std":  cv_mae_std,
                "log_mae":     log_mae,
                "log_r2":      log_r2,
                "dollar_mae":  dollar_mae,
                "dollar_r2":   dollar_r2,
                "smape":       smape,
                "wape":        wape,
            })
    except Exception as mlf_err:
        logger.warning(f"MLflow logging failed for '{name}': {mlf_err}")


def log_champion_to_mlflow(
    tuned_model,
    champion_name: str,
    dollar_r2: float,
    dollar_mae: float,
    log_r2: float,
    feature_names: list,
) -> None:
    """
    Logs the final tuned champion model as an MLflow artifact run.
    Registers under 'CLV_Champion' in the MLflow Model Registry.

    Called from main_execution.ipynb after tune_champion_model() +
    evaluate_and_plot() so final test-set metrics are available.

    Parameters
    ----------
    tuned_model   : fitted champion (post GridSearchCV)
    champion_name : string model name
    dollar_r2     : final Dollar R² on test set (from evaluate_and_plot)
    dollar_mae    : final Dollar MAE on test set (from evaluate_and_plot)
    log_r2        : final Log R² on test set (from evaluate_and_plot)
    feature_names : list of feature column names (FEATURE_COLS)
    """
    if not MLFLOW_AVAILABLE:
        logger.info("MLflow not available — champion artifact logging skipped.")
        return

    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name=f"CHAMPION_{champion_name}"):
            mlflow.log_param("champion_name",  champion_name)
            mlflow.log_param("model_version",  "2.5.0")
            mlflow.log_param("n_features",     len(feature_names))
            mlflow.log_param("feature_names",  str(feature_names))

            mlflow.log_metrics({
                "champion_dollar_r2":  dollar_r2,
                "champion_dollar_mae": dollar_mae,
                "champion_log_r2":     log_r2,
            })

            mlflow.sklearn.log_model(
                sk_model              = tuned_model,
                artifact_path         = "champion_model",
                registered_model_name = "CLV_Champion",
                input_example         = {f: 0.0 for f in feature_names},
            )

            logger.info(
                f"Champion '{champion_name}' logged to MLflow "
                f"experiment '{MLFLOW_EXPERIMENT}' | Dollar R²: {dollar_r2:.4f}"
            )

    except Exception as mlf_err:
        logger.warning(f"Champion MLflow logging failed: {mlf_err}", exc_info=True)


# ===========================================================================
# Model Zoo
# ===========================================================================

def _build_model_zoo(y_train: pd.Series) -> dict:
    """
    Constructs the full model registry (v2.5.0).

    Changes from v2.4.0:
      - CatBoost added (if installed)
      - XGBoost/LightGBM regressors have monotone_constraints
      - TwoStage classifiers now get calibrated inside TwoStageRegressor.fit()
        (no change to zoo definitions — calibration is automatic in fit())
    """
    n_positive = int((y_train > 0).sum())
    n_negative = int((y_train == 0).sum())
    spw = max(1.0, n_negative / n_positive) if n_positive > 0 else 1.0
    logger.info(
        f"Class imbalance — Spenders: {n_positive:,} | "
        f"Non-spenders: {n_negative:,} | scale_pos_weight: {spw:.2f}"
    )

    models = {
        "Naive Baseline (Mean)":     NaiveBaselineMean(),
        "BTYD Statistical Baseline": BTYDBaseline(),

        "Linear Regression": Pipeline([
            ('scaler', StandardScaler()), ('model', LinearRegression())
        ]),
        "Ridge (L2)": Pipeline([
            ('scaler', StandardScaler()),
            ('model',  Ridge(alpha=10.0, random_state=RANDOM_SEED))
        ]),
        "ElasticNet": Pipeline([
            ('scaler', StandardScaler()),
            ('model',  ElasticNet(
                alpha=0.05, l1_ratio=0.5, max_iter=5000, random_state=RANDOM_SEED
            ))
        ]),

        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            max_features=0.7, random_state=RANDOM_SEED, n_jobs=-1
        ),

        "Two-Stage (Random Forest)": TwoStageRegressor(
            classifier=RandomForestClassifier(
                n_estimators=300, max_depth=6, class_weight='balanced',
                min_samples_leaf=5, random_state=RANDOM_SEED, n_jobs=-1
            ),
            regressor=RandomForestRegressor(
                n_estimators=300, max_depth=8, min_samples_leaf=8,
                max_features=0.7, random_state=RANDOM_SEED, n_jobs=-1
            ),
        ),
    }

    if ADVANCED_BOOSTING_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            objective='reg:squarederror', n_estimators=300, max_depth=4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=10, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            monotone_constraints=_XGB_MONOTONE,
            random_state=RANDOM_SEED, verbosity=0, n_jobs=-1
        )

        models["LightGBM"] = LGBMRegressor(
            objective='regression', n_estimators=300, num_leaves=31,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            monotone_constraints=_LGBM_MONOTONE,
            monotone_constraints_method='advanced',
            verbose=-1, random_state=RANDOM_SEED, n_jobs=-1
        )

        models["Two-Stage (XGBoost)"] = TwoStageRegressor(
            classifier=XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                scale_pos_weight=spw, min_child_weight=5,
                random_state=RANDOM_SEED, verbosity=0, n_jobs=-1
            ),
            regressor=XGBRegressor(
                objective='reg:squarederror', n_estimators=300, max_depth=4,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=8, reg_alpha=0.1, reg_lambda=1.0,
                monotone_constraints=_XGB_MONOTONE,
                random_state=RANDOM_SEED, verbosity=0, n_jobs=-1
            ),
        )

        models["Two-Stage (LightGBM)"] = TwoStageRegressor(
            classifier=LGBMClassifier(
                n_estimators=300, num_leaves=31, learning_rate=0.05,
                scale_pos_weight=spw, min_child_samples=15,
                verbose=-1, random_state=RANDOM_SEED, n_jobs=-1
            ),
            regressor=LGBMRegressor(
                objective='regression', n_estimators=300, num_leaves=31,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                monotone_constraints=_LGBM_MONOTONE,
                monotone_constraints_method='advanced',
                verbose=-1, random_state=RANDOM_SEED, n_jobs=-1
            ),
        )

    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6,
            l2_leaf_reg=3.0, loss_function='RMSE',
            random_seed=RANDOM_SEED, verbose=0,
            allow_writing_files=False,
        )

        models["Two-Stage (CatBoost)"] = TwoStageRegressor(
            classifier=CatBoostClassifier(
                iterations=300, learning_rate=0.05, depth=4,
                scale_pos_weight=spw, loss_function='Logloss',
                random_seed=RANDOM_SEED, verbose=0,
                allow_writing_files=False,
            ),
            regressor=CatBoostRegressor(
                iterations=500, learning_rate=0.05, depth=6,
                l2_leaf_reg=3.0, loss_function='RMSE',
                random_seed=RANDOM_SEED, verbose=0,
                allow_writing_files=False,
            ),
        )

    return models


# ===========================================================================
# Training & Benchmarking
# ===========================================================================

def train_and_benchmark(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[object, str, pd.DataFrame]:
    """
    Trains every model in the zoo, evaluates, and builds leaderboard.
    Champion selected from ELIGIBLE models only (baselines + ensemble excluded).

    v2.5.0 changes:
      - log_preds clipped at LOG_PRED_MAX=12.0 (kills linear model explosion)
      - WeightedEnsemble added post-loop using top-3 by Dollar_R²

    v2.6.0 changes:
      - Each model run logged to MLflow as nested run under MLFLOW_EXPERIMENT
    """
    logger.info(
        "[6/8] Benchmarking Model Zoo (CV=5, MAE, log-scale — v2.5.0)..."
    )

    if len(X_train) < MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"Training set has only {len(X_train)} customers — "
            f"minimum required is {MIN_TRAIN_SAMPLES}."
        )

    cv_folds = min(5, max(3, len(X_train) // 10))
    models   = _build_model_zoo(y_train)
    results  = []

    dollar_actual = np.expm1(y_test.values)

    y_binary_cv      = (y_train.values > 0).astype(int)
    skf              = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    stratified_folds = list(skf.split(X_train, y_binary_cv))

    # Start parent MLflow run — all model runs nested inside
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        parent_run = mlflow.start_run(run_name="benchmark_run")

    try:
        for name, model in models.items():
            try:
                cv_splitter = (
                    stratified_folds if isinstance(model, TwoStageRegressor)
                    else cv_folds
                )
                cv_scores   = cross_val_score(
                    model, X_train, y_train,
                    cv=cv_splitter, scoring=CV_SCORING, n_jobs=-1
                )
                cv_mae_mean = -cv_scores.mean()
                cv_mae_std  =  cv_scores.std()

                model.fit(X_train, y_train)

                log_preds = np.clip(model.predict(X_test), a_min=0, a_max=LOG_PRED_MAX)

                log_mae  = mean_absolute_error(y_test, log_preds)
                log_rmse = np.sqrt(mean_squared_error(y_test, log_preds))
                log_r2   = r2_score(y_test, log_preds)

                dollar_preds = np.expm1(log_preds)
                dollar_mae   = mean_absolute_error(dollar_actual, dollar_preds)
                dollar_rmse  = np.sqrt(mean_squared_error(dollar_actual, dollar_preds))
                dollar_r2    = r2_score(dollar_actual, dollar_preds)

                smape = np.mean(
                    2 * np.abs(dollar_preds - dollar_actual) /
                    (np.abs(dollar_preds) + np.abs(dollar_actual) + 1e-8)
                ) * 100

                total = np.sum(np.abs(dollar_actual))
                wape  = (
                    np.sum(np.abs(dollar_actual - dollar_preds)) / total * 100
                    if total > 0 else float('nan')
                )

                # --- MLflow: log this model as nested run ---
                try:
                    _model_params = model.get_params() if hasattr(model, "get_params") else {}
                except Exception:
                    _model_params = {}
                _log_model_to_mlflow(
                    name=name,
                    params=_model_params,
                    cv_mae_mean=cv_mae_mean,
                    cv_mae_std=cv_mae_std,
                    log_mae=log_mae,
                    log_r2=log_r2,
                    dollar_mae=dollar_mae,
                    dollar_r2=dollar_r2,
                    smape=smape,
                    wape=wape if not np.isnan(wape) else 0.0,
                )

                results.append({
                    'Model':       name,
                    'Log_MAE':     log_mae,
                    'Log_RMSE':    log_rmse,
                    'Log_R2':      log_r2,
                    'Dollar_RMSE': dollar_rmse,
                    'Dollar_MAE':  dollar_mae,
                    'Dollar_R2':   dollar_r2,
                    'SMAPE':       smape,
                    'WAPE':        wape,
                    'CV_MAE_Mean': cv_mae_mean,
                    'CV_MAE_Std':  cv_mae_std,
                    'Object':      model,
                })

                logger.info(
                    f"  {name:<35} | "
                    f"Log MAE: {log_mae:.4f} | Log R²: {log_r2:.4f} | "
                    f"$MAE: ${dollar_mae:,.0f} | $R²: {dollar_r2:.4f} | "
                    f"CV MAE: {cv_mae_mean:.4f} ± {cv_mae_std:.4f}"
                )

            except Exception as e:
                logger.warning(f"Model '{name}' failed: {e}", exc_info=True)

    finally:
        # Always close parent MLflow run, even if a model errors
        if MLFLOW_AVAILABLE:
            mlflow.end_run()

    if not results:
        raise RuntimeError("All models failed. Check data quality.")

    # -----------------------------------------------------------------------
    # Weighted Ensemble — top-3 by Dollar_R², inverse-MAE weights
    # -----------------------------------------------------------------------
    eligible_for_ensemble = sorted(
        [r for r in results
         if r['Model'] not in SELECTION_INELIGIBLE and r['Dollar_R2'] > 0.10],
        key=lambda r: r['Dollar_R2'],
        reverse=True,
    )

    if len(eligible_for_ensemble) >= 2:
        top_k           = eligible_for_ensemble[:3]
        ens_models      = [r['Object'] for r in top_k]
        ens_names       = [r['Model']  for r in top_k]
        raw_weights     = np.array([1.0 / (r['Dollar_MAE'] + 1e-8) for r in top_k])
        ens_weights     = raw_weights / raw_weights.sum()

        ensemble        = WeightedEnsemble(ens_models, ens_weights, ens_names)
        ens_log_preds   = np.clip(ensemble.predict(X_test), a_min=0, a_max=LOG_PRED_MAX)
        ens_dollar      = np.expm1(ens_log_preds)

        ens_log_mae  = mean_absolute_error(y_test, ens_log_preds)
        ens_log_r2   = r2_score(y_test, ens_log_preds)
        ens_dol_mae  = mean_absolute_error(dollar_actual, ens_dollar)
        ens_dol_rmse = np.sqrt(mean_squared_error(dollar_actual, ens_dollar))
        ens_dol_r2   = r2_score(dollar_actual, ens_dollar)
        ens_smape    = np.mean(
            2 * np.abs(ens_dollar - dollar_actual) /
            (np.abs(ens_dollar) + np.abs(dollar_actual) + 1e-8)
        ) * 100
        ens_wape     = (
            np.sum(np.abs(dollar_actual - ens_dollar)) / np.sum(np.abs(dollar_actual)) * 100
        )

        results.append({
            'Model':       ENSEMBLE_NAME,
            'Log_MAE':     ens_log_mae,
            'Log_RMSE':    np.sqrt(mean_squared_error(y_test, ens_log_preds)),
            'Log_R2':      ens_log_r2,
            'Dollar_RMSE': ens_dol_rmse,
            'Dollar_MAE':  ens_dol_mae,
            'Dollar_R2':   ens_dol_r2,
            'SMAPE':       ens_smape,
            'WAPE':        ens_wape,
            'CV_MAE_Mean': float(np.average(
                [r['CV_MAE_Mean'] for r in top_k], weights=ens_weights
            )),
            'CV_MAE_Std':  0.0,
            'Object':      ensemble,
        })

        logger.info(
            f"  {'Weighted Ensemble':<35} | "
            f"Log R²: {ens_log_r2:.4f} | $R²: {ens_dol_r2:.4f} | "
            f"$MAE: ${ens_dol_mae:,.0f} | "
            f"Components: {ens_names}"
        )
    else:
        logger.warning("Fewer than 2 eligible models — Weighted Ensemble skipped.")

    # -----------------------------------------------------------------------
    # Build & display leaderboard
    # -----------------------------------------------------------------------
    leaderboard_df = (
        pd.DataFrame(results)
        .sort_values(by='CV_MAE_Mean')
        .reset_index(drop=True)
    )

    print("\n" + "=" * 115)
    print("   MASTER MODEL LEADERBOARD v2.5.0 — Ranked by CV MAE (log-scale)")
    print("   Primary metrics: Log-scale | Business metrics: Dollar-scale")
    print("   * = Selection ineligible  † = No independent CV (test-set evaluation only)")
    print("=" * 115)

    display_df = leaderboard_df[[
        'Model', 'Log_MAE', 'Log_R2', 'Dollar_MAE',
        'Dollar_R2', 'WAPE', 'CV_MAE_Mean', 'CV_MAE_Std'
    ]].copy()

    def _label(m):
        if m in SELECTION_INELIGIBLE:
            return f"* {m}"
        if m == ENSEMBLE_NAME:
            return f"† {m}"
        return f"  {m}"

    display_df['Model'] = display_df['Model'].apply(_label)
    print(display_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # -----------------------------------------------------------------------
    # Champion selection
    # -----------------------------------------------------------------------
    eligible_df = leaderboard_df[
        (~leaderboard_df['Model'].isin(SELECTION_INELIGIBLE)) &
        (leaderboard_df['Model'] != ENSEMBLE_NAME) &
        (leaderboard_df['Dollar_R2'] > MIN_ELIGIBLE_DOLLAR_R2)
    ]

    if eligible_df.empty:
        raise RuntimeError("No eligible models remain after filtering.")

    best_model    = eligible_df.iloc[0]['Object']
    champion_name = eligible_df.iloc[0]['Model']

    logger.info(
        f"Champion: {champion_name} | "
        f"CV MAE: {eligible_df.iloc[0]['CV_MAE_Mean']:.4f} | "
        f"Log R²: {eligible_df.iloc[0]['Log_R2']:.4f} | "
        f"Dollar R²: {eligible_df.iloc[0]['Dollar_R2']:.4f}"
    )

    return best_model, champion_name, leaderboard_df


# ===========================================================================
# Hyperparameter Tuning
# ===========================================================================

def tune_champion_model(
    best_model: object,
    champion_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> object:
    """
    Tunes the champion model via GridSearchCV.

    v2.5.0 changes:
      - StratifiedKFold carried through from train_and_benchmark for TwoStage models
      - LOG_PRED_MAX clip applied consistently
      - CatBoost param grids added
      - WeightedEnsemble short-circuits immediately (nothing to tune)
    """
    logger.info(f"[7/8] Tuning Champion: {champion_name} (v2.5.0)...")

    if champion_name == ENSEMBLE_NAME:
        logger.info("Weighted Ensemble selected — no GridSearchCV tuning. Returning as-is.")
        return best_model

    param_grids = {
        "Random Forest": {
            'n_estimators':     [300, 500],
            'max_depth':        [6, 8, 10],
            'min_samples_leaf': [8, 12, 15],
            'max_features':     [0.6, 0.7, 0.8],
        },
        "XGBoost": {
            'n_estimators':     [300, 500],
            'learning_rate':    [0.03, 0.05, 0.08],
            'max_depth':        [3, 4, 5],
            'min_child_weight': [8, 12, 15],
            'subsample':        [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8],
        },
        "LightGBM": {
            'n_estimators':      [300, 500],
            'learning_rate':     [0.03, 0.05],
            'num_leaves':        [15, 31, 63],
            'min_child_samples': [15, 25, 40],
            'subsample':         [0.7, 0.8],
        },
        "CatBoost": {
            'iterations':    [300, 500],
            'learning_rate': [0.03, 0.05],
            'depth':         [4, 6, 8],
            'l2_leaf_reg':   [1.0, 3.0, 5.0],
        },
        "Two-Stage (Random Forest)": {
            'classifier__n_estimators':    [300, 500],
            'classifier__max_depth':       [4, 6],
            'regressor__n_estimators':     [300, 500],
            'regressor__max_depth':        [6, 8],
            'regressor__min_samples_leaf': [6, 10],
        },
        "Two-Stage (XGBoost)": {
            'classifier__n_estimators':    [300, 500],
            'classifier__max_depth':       [3, 4],
            'regressor__n_estimators':     [300, 500],
            'regressor__learning_rate':    [0.03, 0.05],
            'regressor__min_child_weight': [6, 10],
        },
        "Two-Stage (LightGBM)": {
            'classifier__n_estimators':     [300, 500],
            'classifier__num_leaves':       [15, 31],
            'regressor__n_estimators':      [300, 500],
            'regressor__min_child_samples': [15, 25],
        },
        "Two-Stage (CatBoost)": {
            'classifier__iterations':  [200, 300],
            'classifier__depth':       [3, 4],
            'regressor__iterations':   [300, 500],
            'regressor__depth':        [4, 6],
            'regressor__l2_leaf_reg':  [1.0, 3.0],
        },
        "ElasticNet": {
            'model__alpha':    [0.01, 0.05, 0.1, 0.5],
            'model__l1_ratio': [0.2, 0.5, 0.8],
        },
        "Ridge (L2)": {
            'model__alpha': [1.0, 5.0, 10.0, 50.0],
        },
    }

    cv_folds = min(5, max(3, len(X_train) // 10))

    y_binary_cv = (y_train.values > 0).astype(int)
    if isinstance(best_model, TwoStageRegressor):
        skf         = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
        cv_splitter = list(skf.split(X_train, y_binary_cv))
    else:
        cv_splitter = cv_folds

    pre_cv = -cross_val_score(
        best_model, X_train, y_train,
        cv=cv_splitter, scoring=CV_SCORING, n_jobs=-1
    ).mean()
    logger.info(f"Pre-tuning CV MAE (log): {pre_cv:.4f}")

    grid_params = param_grids.get(champion_name)

    if grid_params:
        grid = GridSearchCV(
            best_model, grid_params,
            cv=cv_splitter, scoring=CV_SCORING, n_jobs=-1, verbose=1,
        )
        grid.fit(X_train, y_train)
        tuned_model = grid.best_estimator_
        post_cv     = -grid.best_score_
        logger.info(f"Best params: {grid.best_params_}")
        logger.info(
            f"Tuning — Pre CV MAE: {pre_cv:.4f} → Post CV MAE: {post_cv:.4f}"
        )
    else:
        logger.info(f"No tuning grid for '{champion_name}'. Returning base model.")
        tuned_model = best_model

    log_preds     = np.clip(tuned_model.predict(X_test), 0, LOG_PRED_MAX)
    dollar_preds  = np.expm1(log_preds)
    dollar_actual = np.expm1(y_test.values)

    log_r2     = r2_score(y_test, log_preds)
    dollar_r2  = r2_score(dollar_actual, dollar_preds)
    dollar_mae = mean_absolute_error(dollar_actual, dollar_preds)

    logger.info(
        f"Final tuned model — "
        f"Log R²: {log_r2:.4f} | Dollar R²: {dollar_r2:.4f} | "
        f"Dollar MAE: ${dollar_mae:,.2f}"
    )

    return tuned_model