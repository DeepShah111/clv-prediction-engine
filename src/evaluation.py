"""
Model evaluation, business impact analysis, and artifact serialization.

UPGRADED v2.3.0 (Round 6):
  NEW PLOTS:
    Plot 3 (upgraded) — Dual feature importance: Stage 2 regressor (existing)
                        + Stage 1 classifier side-by-side for TwoStage models.
    Plot 5 (new)      — SHAP beeswarm summary (requires shap; skipped if absent).
    Plot 6 (new)      — SHAP waterfall plots for whale / mid-spender / low-spender.
    Plot 7 (new)      — Stage 1 calibration curve + probability distribution.

  NEW PARAMETER:
    evaluate_and_plot(..., churn_threshold=0.50)
    Passed from main_execution.ipynb for calibration plot annotation.

  RETAINED from v2.2.0:
    - Dual-scale metrics (log + dollar)
    - Segment-level evaluation with train-derived thresholds
    - Plots 1 (accuracy), 2 (lift), 4 (residuals)
    - Segment CSV export + model serialization
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.calibration import calibration_curve

from src.config import GRAPHS_DIR, MODELS_DIR, MODEL_VERSION

logger = logging.getLogger(__name__)

# Optional SHAP — graceful degradation if not installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.info("shap not installed — SHAP plots skipped. Run: pip install shap")


# ===========================================================================
# Metric Helpers
# ===========================================================================

def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return float(np.mean(2 * np.abs(y_pred - y_true) / denom) * 100)


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / total * 100) \
           if total > 0 else float('nan')


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'RMSE':  np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE':   mean_absolute_error(y_true, y_pred),
        'R2':    r2_score(y_true, y_pred),
        'WAPE':  _wape(y_true, y_pred),
        'SMAPE': _smape(y_true, y_pred),
    }


def _get_tree_importances(estimator) -> np.ndarray | None:
    """
    Safely extracts feature_importances_ from an estimator, unwrapping:
      - CalibratedClassifierCV  (accesses first calibrated sub-estimator)
      - sklearn Pipeline        (accesses 'model' named step)
    Returns None if the estimator has no feature_importances_.
    """
    # Unwrap CalibratedClassifierCV
    if hasattr(estimator, 'calibrated_classifiers_'):
        cal = estimator.calibrated_classifiers_[0]
        # sklearn >= 1.2 uses .estimator; older uses .base_estimator
        estimator = getattr(cal, 'estimator', getattr(cal, 'base_estimator', estimator))

    # Unwrap sklearn Pipeline
    if hasattr(estimator, 'named_steps'):
        estimator = estimator.named_steps.get('model', estimator)

    return getattr(estimator, 'feature_importances_', None)


# ===========================================================================
# Segment-Level Evaluation
# ===========================================================================

def _segment_evaluation(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    segment_thresholds: tuple = None,
) -> pd.DataFrame:
    """
    Evaluates model performance across customer value segments.
    segment_thresholds=(p20, p80) from y_train_raw ensures stable boundaries.
    """
    df = pd.DataFrame({'Actual': y_true_raw, 'Predicted': y_pred_raw})

    if segment_thresholds is not None:
        p20, p80 = segment_thresholds
    else:
        positive_actual = y_true_raw[y_true_raw > 0]
        p20 = np.percentile(positive_actual, 20) if len(positive_actual) > 0 else 0
        p80 = np.percentile(positive_actual, 80) if len(positive_actual) > 0 else 0

    segments = {
        'Top 20% Spenders':     df[df['Actual'] >= p80],
        'Mid Spenders':         df[(df['Actual'] > p20) & (df['Actual'] < p80)],
        'Low Spenders':         df[(df['Actual'] > 0) & (df['Actual'] <= p20)],
        'Zero-Spend (Churned)': df[df['Actual'] == 0],
        'All Customers':        df,
    }

    rows = []
    for seg_name, seg_df in segments.items():
        if len(seg_df) < 5:
            continue
        m = _compute_metrics(seg_df['Actual'].values, seg_df['Predicted'].values)
        rows.append({
            'Segment':    seg_name,
            'N':          len(seg_df),
            'RMSE':       m['RMSE'],
            'MAE':        m['MAE'],
            'R2':         m['R2'],
            'WAPE%':      m['WAPE'],
            'SMAPE%':     m['SMAPE'],
            'Avg_Actual': seg_df['Actual'].mean(),
            'Avg_Pred':   seg_df['Predicted'].mean(),
        })

    return pd.DataFrame(rows)


# ===========================================================================
# Main Evaluation Function
# ===========================================================================

def evaluate_and_plot(
    model,
    champion_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_test_raw: pd.Series,
    segment_thresholds: tuple = None,
    churn_threshold: float = 0.50,
) -> None:
    """
    Computes final metrics on log-scale and dollar-scale.
    Generates all diagnostic plots. Saves all artifacts to config paths.

    Parameters
    ----------
    model               : fitted champion model (predicts in log-scale)
    champion_name       : string name of champion model
    X_test              : test feature DataFrame
    y_test              : log1p-transformed test targets
    y_test_raw          : original dollar-scale test targets
    segment_thresholds  : (p20, p80) tuple from y_train_raw — stable boundaries
    churn_threshold     : CHURN_THRESHOLD from modeling.py (for calibration plot)
    """
    logger.info("[8/8] Evaluating Business Impact — Dual Scale (v2.3.0)...")

    from src.modeling import LOG_PRED_MAX

    # Predictions with consistent upper clip
    log_preds     = np.clip(model.predict(X_test), a_min=0, a_max=LOG_PRED_MAX)
    dollar_preds  = np.expm1(log_preds)
    dollar_actual = y_test_raw.values

    log_metrics    = _compute_metrics(y_test.values, log_preds)
    dollar_metrics = _compute_metrics(dollar_actual, dollar_preds)

    print(
        f"\n{'='*60}\n"
        f" FINAL CHAMPION METRICS: {champion_name}\n"
        f"{'='*60}\n"
        f" LOG-SCALE METRICS (model optimisation target):\n"
        f"   Log-RMSE  : {log_metrics['RMSE']:.4f}\n"
        f"   Log-MAE   : {log_metrics['MAE']:.4f}\n"
        f"   Log-R²    : {log_metrics['R2']:.4f}\n"
        f"\n"
        f" DOLLAR-SCALE METRICS (business reporting):\n"
        f"   RMSE  : ${dollar_metrics['RMSE']:,.2f}\n"
        f"   MAE   : ${dollar_metrics['MAE']:,.2f}\n"
        f"   R²    : {dollar_metrics['R2']:.4f}\n"
        f"   WAPE  : {dollar_metrics['WAPE']:.2f}%\n"
        f"   SMAPE : {dollar_metrics['SMAPE']:.2f}%\n"
        f"{'='*60}"
    )

    # Segment evaluation
    seg_df = _segment_evaluation(dollar_actual, dollar_preds, segment_thresholds)
    print(f"\n{'='*60}")
    print(" SEGMENT-LEVEL PERFORMANCE (dollar-scale)")
    print(f"{'='*60}")
    print(seg_df.to_string(index=False, float_format=lambda x: f'{x:,.2f}'))
    seg_df.to_csv(GRAPHS_DIR / 'segment_metrics.csv', index=False)
    logger.info("Segment metrics saved.")

    # -------------------------------------------------------------------
    # Plot 1: Actual vs Predicted (dollar + log scale)
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.scatter(dollar_actual, dollar_preds, alpha=0.4, color='purple', s=12)
    max_val = max(dollar_actual.max(), dollar_preds.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Spend ($)', fontsize=11)
    ax.set_ylabel('Predicted Spend ($)', fontsize=11)
    ax.set_title(
        f'Actual vs Predicted — Dollar Scale\n'
        f'R²={dollar_metrics["R2"]:.3f} | MAE=${dollar_metrics["MAE"]:,.0f}',
        fontsize=11
    )
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.scatter(np.log1p(dollar_actual), np.log1p(dollar_preds),
                alpha=0.4, color='steelblue', s=12)
    max_log = max(np.log1p(dollar_actual).max(), np.log1p(dollar_preds).max()) * 1.05
    ax2.plot([0, max_log], [0, max_log], 'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('log1p(Actual Spend)', fontsize=11)
    ax2.set_ylabel('log1p(Predicted Spend)', fontsize=11)
    ax2.set_title(
        f'Actual vs Predicted — Log Scale\n'
        f'Log R²={log_metrics["R2"]:.3f} | Log MAE={log_metrics["MAE"]:.4f}',
        fontsize=11
    )
    ax2.legend(fontsize=9)

    fig.suptitle(f'Champion: {champion_name}', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / 'accuracy_check.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info("Plot 1 saved: accuracy_check.png")

    # -------------------------------------------------------------------
    # Plot 2: Business Lift (Gain Chart)
    # -------------------------------------------------------------------
    eval_df = pd.DataFrame({
        'Actual': dollar_actual, 'Predicted': dollar_preds,
    }).sort_values(by='Predicted', ascending=False).reset_index(drop=True)

    total_actual = eval_df['Actual'].sum()

    if total_actual == 0:
        logger.warning("All y_test values are zero — lift chart skipped.")
    else:
        eval_df['Cum_Actual']     = eval_df['Actual'].cumsum() / total_actual
        eval_df['Cum_Population'] = np.linspace(0, 1, len(eval_df))
        
        idx_10  = int(len(eval_df) * 0.10)
        lift_10 = eval_df['Cum_Actual'].iloc[idx_10]
        idx_20  = int(len(eval_df) * 0.20)
        lift_20 = eval_df['Cum_Actual'].iloc[idx_20]
        idx_40  = int(len(eval_df) * 0.40)
        lift_40 = eval_df['Cum_Actual'].iloc[idx_40]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(eval_df['Cum_Population'], eval_df['Cum_Actual'],
                label='Champion Model', color='green', lw=3)
        ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Random Baseline (No Model)')
        ax.axvline(x=0.10, color='gray', lw=1, linestyle=':')
        ax.axvline(x=0.20, color='gray', lw=1, linestyle=':')
        ax.axvline(x=0.40, color='gray', lw=1, linestyle=':')

        ax.annotate(
            f'Top 10% customers\ncapture {lift_10:.0%} of revenue',
            xy=(0.10, lift_10), xytext=(0.18, max(lift_10 - 0.12, 0.05)),
            fontsize=10, color='darkgreen',
            arrowprops=dict(arrowstyle='->', color='darkgreen')
        )

        ax.annotate(
            f'Top 20% customers\ncapture {lift_20:.0%} of revenue',
            xy=(0.20, lift_20), xytext=(0.28, max(lift_20 - 0.1, 0.05)),
            fontsize=10, color='darkgreen',
            arrowprops=dict(arrowstyle='->', color='darkgreen')
        )
        ax.set_xlabel('Fraction of Customers Targeted', fontsize=12)
        ax.set_ylabel('Fraction of Total Revenue Captured', fontsize=12)
        ax.set_title('Business Lift Analysis — Gain Chart', fontsize=14)
        ax.legend(fontsize=11)
        fig.tight_layout()
        fig.savefig(GRAPHS_DIR / 'business_lift.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        logger.info(
            f"Plot 2 saved | Top 10%: {lift_10:.1%} | Top 20%: {lift_20:.1%} | Top 40%: {lift_40:.1%}"
        )

    # -------------------------------------------------------------------
    # Plot 3: Feature Importance — Stage 2 Regressor + Stage 1 Classifier
    # For TwoStage models: side-by-side importance plots for both stages.
    # For single-stage models: single importance plot (unchanged behaviour).
    # -------------------------------------------------------------------
    try:
        feature_names = list(X_test.columns) if hasattr(X_test, 'columns') else None

        # Regressor importance (Stage 2 or the full model)
        reg_estimator = model
        if hasattr(model, 'regressor_') and model.regressor_ is not None:
            reg_estimator = model.regressor_
        reg_importances = _get_tree_importances(reg_estimator)

        # Classifier importance (Stage 1, only for TwoStage models)
        clf_importances = None
        if hasattr(model, 'classifier_'):
            clf_importances = _get_tree_importances(model.classifier_)

        if reg_importances is not None and feature_names is not None:
            n_cols = 2 if clf_importances is not None else 1
            fig, axes = plt.subplots(1, n_cols, figsize=(10 * n_cols, 7))
            if n_cols == 1:
                axes = [axes]

            # Stage 2 / full model importance
            fi_reg = (
                pd.DataFrame({
                    'Feature':    feature_names[:len(reg_importances)],
                    'Importance': reg_importances,
                })
                .sort_values('Importance', ascending=False)
                .head(14)
            )
            colors_reg = ['#5C4DB1' if i < 3 else '#9B8FD9' for i in range(len(fi_reg))]
            sns.barplot(x='Importance', y='Feature', data=fi_reg,
                        palette=colors_reg, ax=axes[0])
            stage_label = "Stage 2 — Regressor" if clf_importances is not None else champion_name
            axes[0].set_title(f'Feature Importances — {stage_label}', fontsize=13)
            axes[0].set_xlabel('Relative Importance', fontsize=11)
            axes[0].set_ylabel('')

            # Stage 1 classifier importance (TwoStage only)
            if clf_importances is not None and len(axes) > 1:
                fi_clf = (
                    pd.DataFrame({
                        'Feature':    feature_names[:len(clf_importances)],
                        'Importance': clf_importances,
                    })
                    .sort_values('Importance', ascending=False)
                    .head(14)
                )
                colors_clf = ['#B84C4C' if i < 3 else '#D98F8F' for i in range(len(fi_clf))]
                sns.barplot(x='Importance', y='Feature', data=fi_clf,
                            palette=colors_clf, ax=axes[1])
                axes[1].set_title('Feature Importances — Stage 1 — Classifier (Churn)', fontsize=13)
                axes[1].set_xlabel('Relative Importance', fontsize=11)
                axes[1].set_ylabel('')

            fig.suptitle(f'Feature Importance Analysis — {champion_name}', fontsize=14)
            fig.tight_layout()
            fig.savefig(GRAPHS_DIR / 'feature_importance.png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            logger.info("Plot 3 saved: feature_importance.png (dual stage)")
        else:
            logger.warning(f"'{champion_name}' has no feature_importances_. Plot 3 skipped.")

    except Exception as e:
        logger.error(f"Feature importance plot failed: {e}", exc_info=True)

    # -------------------------------------------------------------------
    # Plot 4: Residual Analysis (log-scale)
    # -------------------------------------------------------------------
    try:
        log_residuals = y_test.values - log_preds
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(log_residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        axes[0].axvline(x=0, color='red', lw=2, linestyle='--', label='Zero Error')
        axes[0].axvline(
            x=log_residuals.mean(), color='orange', lw=2,
            label=f'Mean: {log_residuals.mean():.3f}'
        )
        axes[0].set_xlabel('Residual (log-scale)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Residual Distribution (log-scale)', fontsize=13)
        axes[0].legend(fontsize=10)

        axes[1].scatter(log_preds, log_residuals, alpha=0.3, color='steelblue', s=12)
        axes[1].axhline(y=0, color='red', lw=2, linestyle='--')
        axes[1].set_xlabel('Predicted (log-scale)', fontsize=12)
        axes[1].set_ylabel('Residual (log-scale)', fontsize=12)
        axes[1].set_title('Residuals vs Predicted (Heteroscedasticity Check)', fontsize=13)

        fig.suptitle(
            f'Residual Analysis — {champion_name} | '
            f'Log MAE: {log_metrics["MAE"]:.4f} | Log R²: {log_metrics["R2"]:.4f}',
            fontsize=12, y=1.01
        )
        fig.tight_layout()
        fig.savefig(GRAPHS_DIR / 'residual_analysis.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        logger.info("Plot 4 saved: residual_analysis.png")

    except Exception as e:
        logger.error(f"Residual plot failed: {e}", exc_info=True)

    # -------------------------------------------------------------------
    # Plot 5: SHAP Beeswarm Summary
    # Applied to Stage 2 regressor on spending customers only.
    # Skipped if shap not installed or model has no tree structure.
    # -------------------------------------------------------------------
    if SHAP_AVAILABLE:
        try:
            shap_estimator = model
            if hasattr(model, 'regressor_') and model.regressor_ is not None:
                shap_estimator = model.regressor_

            shap_importances = _get_tree_importances(shap_estimator)

            if shap_importances is not None:
                # Focus SHAP on customers the model predicts to spend (most informative)
                spender_mask  = dollar_preds > 0
                X_shap        = X_test[spender_mask] if spender_mask.sum() > 20 else X_test
                actual_shap   = dollar_actual[spender_mask] if spender_mask.sum() > 20 else dollar_actual

                explainer  = shap.TreeExplainer(shap_estimator)
                shap_vals  = explainer.shap_values(X_shap)   # (n_samples, n_features)

                # Beeswarm summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_vals, X_shap,
                    max_display=16, show=False, plot_type='dot'
                )
                plt.title(
                    f'SHAP Feature Impact — {champion_name} (Stage 2 Regressor)\n'
                    f'Computed on {len(X_shap)} predicted-spending customers',
                    fontsize=12, pad=20
                )
                plt.tight_layout()
                plt.savefig(GRAPHS_DIR / 'shap_summary.png', bbox_inches='tight', dpi=150)
                plt.close()
                logger.info(f"Plot 5 saved: shap_summary.png ({len(X_shap)} customers)")

                # -----------------------------------------------------------
                # Plot 6: SHAP Waterfall — Whale / Mid-Spender / Low Spender
                # Three representative customer profiles showing individual
                # feature contributions to each prediction.
                # -----------------------------------------------------------
                shap_exp = explainer(X_shap)   # Explanation object for waterfall

                # Identify representative customers within X_shap
                if len(actual_shap) >= 3 and actual_shap.max() > 0:
                    spender_actual = actual_shap[actual_shap > 0]
                    spender_idx    = np.where(actual_shap > 0)[0]

                    # Whale: highest actual spender in shap set
                    whale_local = np.argmax(spender_actual)
                    whale_idx   = spender_idx[whale_local]

                    # Mid: closest to median spend
                    median_val  = np.median(spender_actual)
                    mid_local   = np.argmin(np.abs(spender_actual - median_val))
                    mid_idx     = spender_idx[mid_local]

                    # Low: lowest positive spender
                    low_local   = np.argmin(spender_actual)
                    low_idx     = spender_idx[low_local]

                    profiles = [
                        (whale_idx, f'Whale Customer\nActual: ${actual_shap[whale_idx]:,.0f}'),
                        (mid_idx,   f'Mid-Spender\nActual: ${actual_shap[mid_idx]:,.0f}'),
                        (low_idx,   f'Low Spender\nActual: ${actual_shap[low_idx]:,.0f}'),
                    ]

                    for idx, title in profiles:
                        safe_title = title.split('\n')[0].replace(' ', '_').replace('/', '_')
                        plt.figure(figsize=(10, 6))
                        shap.plots.waterfall(shap_exp[idx], max_display=10, show=False)
                        plt.title(title, fontsize=12, pad=12)
                        plt.tight_layout()
                        fname = f'shap_waterfall_{safe_title.lower()}.png'
                        plt.savefig(GRAPHS_DIR / fname, bbox_inches='tight', dpi=150)
                        plt.close()
                        logger.info(f"Plot 6 saved: {fname}")

                else:
                    logger.warning("Insufficient spending customers for waterfall plots.")

            else:
                logger.warning(f"SHAP skipped — '{champion_name}' has no tree structure.")

        except Exception as shap_err:
            logger.warning(f"SHAP analysis failed: {shap_err}", exc_info=True)
    else:
        logger.info("SHAP plots skipped — shap not installed.")

    # -------------------------------------------------------------------
    # Plot 7: Stage 1 Calibration Curve (TwoStage models only)
    # Shows whether predicted churn probabilities are well-calibrated.
    # A perfectly calibrated model has points on the diagonal.
    # Also shows probability distribution with churn threshold marked.
    # -------------------------------------------------------------------
    if hasattr(model, 'classifier_'):
        try:
            y_binary_test = (dollar_actual > 0).astype(int)
            prob_spend    = model.classifier_.predict_proba(X_test)[:, 1]

            fraction_pos, mean_pred = calibration_curve(
                y_binary_test, prob_spend, n_bins=10, strategy='uniform'
            )

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Calibration curve
            axes[0].plot(mean_pred, fraction_pos, 's-', color='steelblue',
                         lw=2, label='Champion classifier', markersize=6)
            axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')
            axes[0].fill_between(mean_pred, fraction_pos, mean_pred,
                                 alpha=0.1, color='steelblue',
                                 label='Calibration gap')
            axes[0].set_xlabel('Mean predicted P(spend > $0)', fontsize=12)
            axes[0].set_ylabel('Fraction of actual spenders', fontsize=12)
            axes[0].set_title('Stage 1 Classifier — Calibration Curve', fontsize=13)
            axes[0].legend(fontsize=10)
            axes[0].set_xlim(0, 1)
            axes[0].set_ylim(0, 1)

            # Probability distribution with threshold
            n_spenders   = y_binary_test.sum()
            n_non        = len(y_binary_test) - n_spenders
            above_thresh = (prob_spend >= churn_threshold).sum()

            axes[1].hist(prob_spend[y_binary_test == 0], bins=40, alpha=0.6,
                         color='salmon', label=f'Actual non-spenders (n={n_non})',
                         density=True)
            axes[1].hist(prob_spend[y_binary_test == 1], bins=40, alpha=0.6,
                         color='steelblue', label=f'Actual spenders (n={n_spenders})',
                         density=True)
            axes[1].axvline(x=churn_threshold, color='red', lw=2, linestyle='--',
                            label=f'Churn threshold ({churn_threshold}) → '
                                  f'{above_thresh} customers predicted to spend')
            axes[1].set_xlabel('P(spend > $0)', fontsize=12)
            axes[1].set_ylabel('Density', fontsize=12)
            axes[1].set_title('Stage 1 Probability Distribution', fontsize=13)
            axes[1].legend(fontsize=9)

            fig.suptitle(
                f'Stage 1 Calibration Analysis — {champion_name}',
                fontsize=13, y=1.01
            )
            fig.tight_layout()
            fig.savefig(GRAPHS_DIR / 'calibration_curve.png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            logger.info("Plot 7 saved: calibration_curve.png")

        except Exception as cal_err:
            logger.error(f"Calibration curve failed: {cal_err}", exc_info=True)

    logger.info(f"All evaluation artifacts saved to: {GRAPHS_DIR}")


# ===========================================================================
# Model Serialization
# ===========================================================================

def save_model(model, feature_names: list) -> None:
    """Serializes the champion model bundle with metadata."""
    bundle = {
        'model':            model,
        'feature_names':    feature_names,
        'target_transform': 'log1p',
        'timestamp':        pd.Timestamp.now().isoformat(),
        'version':          MODEL_VERSION,
    }
    output_path = MODELS_DIR / 'clv_champion_bundle.pkl'
    joblib.dump(bundle, output_path)
    logger.info(f"Production model bundle saved to: {output_path}")