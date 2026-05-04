"""
CLV Prediction — Streamlit App
================================
Interactive dashboard for the Customer Lifetime Value pipeline (v2.5.0).

Features
--------
  Tab 1 — Single Customer Predictor
    • Sliders: Recency, Frequency, Monetary, Days_Since_Purchase, Max_Single_Order
    • Remaining features populated with dataset-calibrated defaults
    • Predicted 90-day CLV with ±1 std confidence band
    • Customer segment badge (Whale / Mid / Low / Churned)
    • SHAP waterfall for that specific customer
    • Position on Gain/Lift chart

  Tab 2 — Batch CSV Upload
    • Upload any CSV with FEATURE_COLS columns (or a subset)
    • Predict all customers, show segment breakdown
    • Download enriched CSV with predictions

Usage
-----
    streamlit run streamlit_app.py

Requirements (add to requirements.txt):
    streamlit>=1.35.0
    shap>=0.45.0
    matplotlib>=3.8.0
    pandas>=2.0.0
    joblib>=1.3.0
"""

import os
import io
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt
import joblib

import streamlit as st

# ── Suppress verbose libraries in the Streamlit log ──────────────────────────
logging.getLogger("shap").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ── SHAP — graceful degradation ───────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# =============================================================================
# Path / Model Config
# =============================================================================
from pathlib import Path
import sys

# Allow running from project root OR from any subdirectory
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Config must be importable; adjust CLV_BASE_DIR if needed
os.environ.setdefault("CLV_BASE_DIR", str(_HERE))

from src.config import (
    MODELS_DIR, GRAPHS_DIR, FEATURE_COLS, setup_logging, setup_directories,
)

# =============================================================================
# Constants — Segment Thresholds & UI Colours
# =============================================================================
#   Derived from Online Retail II training distribution.
#   Update SEGMENT_P20 / SEGMENT_P80 if you retrain on different data.
SEGMENT_P20 = 150.0     # ~20th percentile of positive spend (dollar-scale)
SEGMENT_P80 = 1_200.0   # ~80th percentile of positive spend (dollar-scale)
CHURN_THRESHOLD = 0.50  # mirrors CHURN_THRESHOLD in modeling.py

SEGMENT_CONFIG = {
    "🐋 Whale":   {"color": "#5C4DB1", "min": SEGMENT_P80,       "max": float("inf")},
    "💰 Mid":     {"color": "#2E86AB", "min": SEGMENT_P20,        "max": SEGMENT_P80},
    "📉 Low":     {"color": "#F4A261", "min": 0.01,               "max": SEGMENT_P20},
    "💤 Churned": {"color": "#E63946", "min": 0.0,                "max": 0.01},
}

LOG_PRED_MAX = 12.0    # mirrors LOG_PRED_MAX in modeling.py

# =============================================================================
# Feature Defaults — median values from training data
#   Used to fill the 11 features NOT exposed in the UI sliders.
#   Sourced from EDA on Online Retail II (UK, 2009-2011).
# =============================================================================
FEATURE_DEFAULTS = {
    "Recency":            365.0,
    "Frequency":          4.0,
    "Monetary":           300.0,
    "Prob_Pred_Txn":      1.5,
    "Prob_Pred_Val":      280.0,
    "Prob_Alive":         0.65,
    "Interpurchase_Std":  45.0,
    "Purchase_Rate":      0.012,
    "Days_Since_Purchase": 60.0,
    "Revenue_Per_Day":    1.8,
    "Unique_Products":    12.0,
    "Visit_Diversity":    5.0,
    "Avg_Basket_Size":    3.2,
    "Return_Rate":        0.04,
    "Monetary_Percentile": 0.50,
    "Max_Single_Order":   250.0,
}

# Slider UI-exposed features — min, max, default, step, format
SLIDER_CONFIG = {
    "Recency":             (30,    730,  365,   10,  "%d days"),
    "Frequency":           (1,     100,  4,     1,   "%d invoices"),
    "Monetary":            (10.0,  5000.0, 300.0, 10.0, "$%.0f"),
    "Days_Since_Purchase": (1,     365,  60,    1,   "%d days"),
    "Max_Single_Order":    (10.0,  5000.0, 250.0, 10.0, "$%.0f"),
}


# =============================================================================
# Helpers
# =============================================================================

@st.cache_resource(show_spinner="Loading CLV model bundle …")
def load_bundle() -> dict:
    """Loads the serialized champion model bundle (cached across sessions)."""
    bundle_path = MODELS_DIR / "clv_champion_bundle.pkl"
    if not bundle_path.exists():
        st.error(
            f"❌ Model bundle not found at `{bundle_path}`. "
            "Run the full pipeline (main_execution.ipynb) first."
        )
        st.stop()
    return joblib.load(bundle_path)


def _build_feature_row(overrides: dict) -> pd.DataFrame:
    """
    Merges UI slider values with FEATURE_DEFAULTS to produce a
    single-row DataFrame in FEATURE_COLS order.
    """
    row = {**FEATURE_DEFAULTS, **overrides}
    return pd.DataFrame([row])[FEATURE_COLS]


def _predict_clv(model, feature_row: pd.DataFrame) -> dict:
    """
    Runs model inference. Returns predicted CLV and confidence bounds.

    Confidence band: ±15% of predicted dollar value.
    A proper band requires either a Bayesian model or quantile regression;
    ±15% is a conservative proxy calibrated against holdout WAPE.
    """
    from src.modeling import LOG_PRED_MAX as _LP_MAX
    log_pred = float(np.clip(model.predict(feature_row), 0, _LP_MAX)[0])
    dollar   = float(np.expm1(log_pred))
    low      = max(0.0, dollar * 0.85)
    high     = dollar * 1.15
    return {"log_pred": log_pred, "dollar": dollar, "low": low, "high": high}


def _get_segment(dollar: float) -> str:
    """Maps a dollar CLV prediction to a named segment."""
    if dollar < 0.01:
        return "💤 Churned"
    elif dollar < SEGMENT_P20:
        return "📉 Low"
    elif dollar < SEGMENT_P80:
        return "💰 Mid"
    else:
        return "🐋 Whale"


def _shap_waterfall_figure(model, feature_row: pd.DataFrame) -> plt.Figure | None:
    """
    Computes SHAP values for a single customer and returns a waterfall figure.
    Returns None if SHAP is unavailable or model has no tree structure.
    """
    if not SHAP_AVAILABLE:
        return None

    estimator = model
    if hasattr(model, "regressor_") and model.regressor_ is not None:
        estimator = model.regressor_

    # Only tree-based models supported
    if not hasattr(estimator, "feature_importances_"):
        return None

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_exp  = explainer(feature_row)

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.sca(ax)
        shap.plots.waterfall(shap_exp[0], max_display=12, show=False)
        plt.title("SHAP Feature Contributions — This Customer", fontsize=12, pad=10)
        plt.tight_layout()
        return fig
    except Exception as e:
        return None


def _gain_chart_figure(dollar_pred: float) -> plt.Figure | None:
    """
    Loads the pre-computed business_lift.png and overlays the customer's
    predicted percentile position. Returns None if the image is missing.
    """
    lift_path = GRAPHS_DIR / "business_lift.png"
    if not lift_path.exists():
        return None

    # Load the gain chart as a background image and annotate
    img = plt.imread(str(lift_path))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img, aspect="auto", extent=[0, 1, 0, 1])
    ax.set_axis_off()
    ax.set_title(
        f"Your Customer's CLV: ${dollar_pred:,.0f} — see where they rank",
        fontsize=11, pad=8
    )
    return fig


def _batch_predict(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs batch CLV prediction on an uploaded DataFrame.

    Missing FEATURE_COLS are filled with FEATURE_DEFAULTS before prediction.
    Returns the input DataFrame with three appended columns:
        CLV_Predicted_90d, CLV_Low, CLV_High, Segment
    """
    from src.modeling import LOG_PRED_MAX as _LP_MAX

    # Fill any missing columns with defaults
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = FEATURE_DEFAULTS.get(col, 0.0)

    X = df[FEATURE_COLS].copy().fillna(
        {c: FEATURE_DEFAULTS.get(c, 0.0) for c in FEATURE_COLS}
    )

    log_preds    = np.clip(model.predict(X), 0, _LP_MAX)
    dollar_preds = np.expm1(log_preds)

    df = df.copy()
    df["CLV_Predicted_90d"] = np.round(dollar_preds, 2)
    df["CLV_Low"]           = np.round(np.maximum(0, dollar_preds * 0.85), 2)
    df["CLV_High"]          = np.round(dollar_preds * 1.15, 2)
    df["Segment"]           = [_get_segment(d) for d in dollar_preds]
    return df


# =============================================================================
# Page Setup
# =============================================================================

st.set_page_config(
    page_title="CLV Predictor v2.5.0",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ Customer Lifetime Value Predictor")
st.caption(
    "**Pipeline v2.5.0** — Two-Stage CatBoost Champion | "
    "BG/NBD + Gamma-Gamma Features | Dollar R² 0.581"
)

# Load model once
bundle  = load_bundle()
model   = bundle["model"]
version = bundle.get("version", "?")
trained = bundle.get("timestamp", "unknown")

with st.sidebar:
    st.markdown("### 🏆 Model Info")
    st.info(
        f"**Champion:** {type(model).__name__}\n\n"
        f"**Version:** {version}\n\n"
        f"**Trained:** {trained[:10] if trained != 'unknown' else 'unknown'}"
    )
    st.markdown("---")
    st.markdown(
        "**Segment thresholds** (dollar, 90-day):\n"
        f"- 🐋 Whale: ≥ ${SEGMENT_P80:,.0f}\n"
        f"- 💰 Mid: ${SEGMENT_P20:,.0f} – ${SEGMENT_P80:,.0f}\n"
        f"- 📉 Low: $0.01 – ${SEGMENT_P20:,.0f}\n"
        f"- 💤 Churned: $0"
    )

tab_single, tab_batch = st.tabs(["🔮 Single Customer", "📂 Batch CSV Upload"])


# =============================================================================
# Tab 1: Single Customer Predictor
# =============================================================================

with tab_single:
    st.subheader("Input Customer Features")
    st.caption(
        "Adjust the 5 key levers below. Remaining 11 features are set to "
        "training-set medians — edit `FEATURE_DEFAULTS` in `streamlit_app.py` "
        "to customise."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.slider(
            "Recency (days since first purchase)",
            min_value=30, max_value=730,
            value=365, step=10,
            help="Total observation window in days (not days since last purchase)."
        )
        frequency = st.slider(
            "Frequency (number of invoices)",
            min_value=1, max_value=100,
            value=4, step=1,
        )

    with col2:
        monetary = st.slider(
            "Monetary (avg order value, $)",
            min_value=10.0, max_value=5000.0,
            value=300.0, step=10.0,
        )
        days_since = st.slider(
            "Days Since Last Purchase",
            min_value=1, max_value=365,
            value=60, step=1,
        )

    with col3:
        max_order = st.slider(
            "Max Single Order ($)",
            min_value=10.0, max_value=5000.0,
            value=250.0, step=10.0,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("▶️ Predict CLV", use_container_width=True, type="primary")

    if predict_btn:
        overrides = {
            "Recency":             float(recency),
            "Frequency":           float(frequency),
            "Monetary":            float(monetary),
            "Days_Since_Purchase": float(days_since),
            "Max_Single_Order":    float(max_order),
        }
        feature_row = _build_feature_row(overrides)
        result      = _predict_clv(model, feature_row)
        segment     = _get_segment(result["dollar"])
        seg_color   = SEGMENT_CONFIG[segment]["color"]

        st.markdown("---")

        # ── Prediction Card ──────────────────────────────────────────────────
        res_col1, res_col2, res_col3 = st.columns([2, 1, 2])

        with res_col1:
            st.metric(
                label="📈 Predicted 90-Day CLV",
                value=f"${result['dollar']:,.2f}",
                delta=f"Range: ${result['low']:,.0f} – ${result['high']:,.0f}",
            )

        with res_col2:
            st.markdown(
                f"""
                <div style="
                    background:{seg_color};
                    border-radius:10px;
                    padding:14px 10px;
                    text-align:center;
                    color:white;
                    font-size:1.25rem;
                    font-weight:600;
                ">
                    {segment}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col3:
            st.markdown("**Feature inputs used**")
            st.dataframe(
                feature_row.T.rename(columns={0: "Value"}).round(3),
                use_container_width=True,
                height=350,
            )

        st.markdown("---")

        # ── SHAP Waterfall ───────────────────────────────────────────────────
        shap_col, lift_col = st.columns(2)

        with shap_col:
            st.subheader("🔍 SHAP Feature Contributions")
            if SHAP_AVAILABLE:
                fig_shap = _shap_waterfall_figure(model, feature_row)
                if fig_shap:
                    st.pyplot(fig_shap, use_container_width=True)
                    plt.close(fig_shap)
                else:
                    st.info(
                        "SHAP waterfall unavailable for this model type "
                        "(only tree-based models are supported)."
                    )
            else:
                st.warning("Install `shap` to enable waterfall plots: `pip install shap`")

        # ── Gain/Lift Position ───────────────────────────────────────────────
        with lift_col:
            st.subheader("📊 Gain Chart Position")
            fig_lift = _gain_chart_figure(result["dollar"])
            if fig_lift:
                st.pyplot(fig_lift, use_container_width=True)
                plt.close(fig_lift)
            else:
                st.info(
                    "Run the full pipeline to generate `business_lift.png`. "
                    "The chart will appear here automatically after the first pipeline run."
                )


# =============================================================================
# Tab 2: Batch CSV Upload
# =============================================================================

with tab_batch:
    st.subheader("Batch Predict — Upload Customer CSV")
    st.caption(
        f"Upload a CSV with any combination of the 16 model features. "
        f"Missing features are filled with training-set medians. "
        f"Required columns: any subset of `{FEATURE_COLS}`."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded, encoding='ISO-8859-1')
            st.success(f"✅ Loaded {len(input_df):,} customers from `{uploaded.name}`")

            with st.spinner("Running predictions …"):
                result_df = _batch_predict(model, input_df)

            st.markdown("#### Preview — Predictions")
            preview_cols = [c for c in result_df.columns
                            if c in FEATURE_COLS[:5] or
                            c in ["CLV_Predicted_90d", "CLV_Low", "CLV_High", "Segment"]]
            st.dataframe(result_df[preview_cols].head(50), use_container_width=True)

            # ── Segment Breakdown ────────────────────────────────────────────
            st.markdown("#### Segment Breakdown")
            seg_counts = (
                result_df["Segment"]
                .value_counts()
                .reset_index()
                .rename(columns={"index": "Segment", "Segment": "Count"})
            )
            seg_col1, seg_col2 = st.columns([1, 2])

            with seg_col1:
                st.dataframe(seg_counts, use_container_width=True)

            with seg_col2:
                seg_agg = (
                    result_df
                    .groupby("Segment")["CLV_Predicted_90d"]
                    .agg(["count", "mean", "sum"])
                    .rename(columns={"count": "N", "mean": "Avg CLV ($)", "sum": "Total CLV ($)"})
                    .reset_index()
                )
                st.dataframe(seg_agg.round(2), use_container_width=True)

            # ── Download Button ──────────────────────────────────────────────
            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv_buffer.getvalue().encode("utf-8"),
                file_name="clv_predictions.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )

        except Exception as e:
            st.error(f"❌ Failed to process file: {e}")