"""
CLV Prediction — Streamlit App
================================
Interactive dashboard for the Customer Lifetime Value pipeline (v2.5.0).

Features
--------
  Tab 1 — Single Customer Predictor
  Tab 2 — Batch CSV Upload
  Tab 3 — Customer Segments        ← NEW
  Tab 4 — Anomaly Detection        ← NEW
"""

import os
import io
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

import streamlit as st

logging.getLogger("shap").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

os.environ.setdefault("CLV_BASE_DIR", str(_HERE))

from src.config import (
    MODELS_DIR, GRAPHS_DIR, FEATURE_COLS, setup_logging, setup_directories,
)

# =============================================================================
# Constants
# =============================================================================
SEGMENT_P20     = 150.0
SEGMENT_P80     = 1_200.0
CHURN_THRESHOLD = 0.50
LOG_PRED_MAX    = 12.0

SEGMENT_CONFIG = {
    "🐋 Whale":   {"color": "#5C4DB1"},
    "💰 Mid":     {"color": "#2E86AB"},
    "📉 Low":     {"color": "#F4A261"},
    "💤 Churned": {"color": "#E63946"},
}

FEATURE_DEFAULTS = {
    "Recency": 365.0, "Frequency": 4.0, "Monetary": 300.0,
    "Prob_Pred_Txn": 1.5, "Prob_Pred_Val": 280.0, "Prob_Alive": 0.65,
    "Interpurchase_Std": 45.0, "Purchase_Rate": 0.012,
    "Days_Since_Purchase": 60.0, "Revenue_Per_Day": 1.8,
    "Unique_Products": 12.0, "Visit_Diversity": 5.0,
    "Avg_Basket_Size": 3.2, "Return_Rate": 0.04,
    "Monetary_Percentile": 0.50, "Max_Single_Order": 250.0,
}

# Segment colour map used in Tab 3
SEG_COLOURS = {
    "Loyal Customers": "#5C4DB1",
    "Champions":       "#2E86AB",
    "At Risk":         "#F4A261",
    "Lost":            "#E63946",
}

# Anomaly risk colours
RISK_COLOURS = {
    "🔴 High":   "#E63946",
    "🟡 Medium": "#F4A261",
    "🟢 Low":    "#2A9D8F",
}


# =============================================================================
# Loaders (cached)
# =============================================================================

@st.cache_resource(show_spinner="Loading CLV model …")
def load_clv_bundle() -> dict:
    p = MODELS_DIR / "clv_champion_bundle.pkl"
    if not p.exists():
        st.error(f"❌ CLV bundle not found: `{p}`")
        st.stop()
    return joblib.load(p)


@st.cache_resource(show_spinner="Loading segmentation bundle …")
def load_seg_bundle() -> dict | None:
    p = MODELS_DIR / "segmentation_bundle.pkl"
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_resource(show_spinner="Loading anomaly bundle …")
def load_anomaly_bundle() -> dict | None:
    p = MODELS_DIR / "anomaly_bundle.pkl"
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_data(show_spinner="Loading segment profiles …")
def load_segment_profiles() -> pd.DataFrame | None:
    p = GRAPHS_DIR / "segment_profiles.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(show_spinner="Loading customer segments …")
def load_customer_segments() -> pd.DataFrame | None:
    p = GRAPHS_DIR / "customer_segments.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(show_spinner="Loading anomaly scores …")
def load_anomaly_scores() -> pd.DataFrame | None:
    p = GRAPHS_DIR / "anomaly_scores.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def _load_graph(name: str) -> Path | None:
    p = GRAPHS_DIR / name
    return p if p.exists() else None


# =============================================================================
# CLV Helpers
# =============================================================================

def _build_feature_row(overrides: dict) -> pd.DataFrame:
    row = {**FEATURE_DEFAULTS, **overrides}
    return pd.DataFrame([row])[FEATURE_COLS]


def _predict_clv(model, feature_row: pd.DataFrame) -> dict:
    from src.modeling import LOG_PRED_MAX as _LP
    log_pred = float(np.clip(model.predict(feature_row), 0, _LP)[0])
    dollar   = float(np.expm1(log_pred))
    return {"log_pred": log_pred, "dollar": dollar,
            "low": max(0.0, dollar * 0.85), "high": dollar * 1.15}


def _get_segment(dollar: float) -> str:
    if dollar < 0.01:   return "💤 Churned"
    elif dollar < SEGMENT_P20: return "📉 Low"
    elif dollar < SEGMENT_P80: return "💰 Mid"
    else:               return "🐋 Whale"


def _shap_waterfall_figure(model, feature_row):
    if not SHAP_AVAILABLE:
        return None
    estimator = model
    if hasattr(model, "regressor_") and model.regressor_ is not None:
        estimator = model.regressor_
    if not hasattr(estimator, "feature_importances_"):
        return None
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_exp  = explainer(feature_row)
        fig, ax   = plt.subplots(figsize=(10, 5))
        plt.sca(ax)
        shap.plots.waterfall(shap_exp[0], max_display=12, show=False)
        plt.title("SHAP Feature Contributions — This Customer", fontsize=12, pad=10)
        plt.tight_layout()
        return fig
    except Exception:
        return None


def _gain_chart_figure(dollar_pred: float):
    lift_path = GRAPHS_DIR / "business_lift.png"
    if not lift_path.exists():
        return None
    img = plt.imread(str(lift_path))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img, aspect="auto", extent=[0, 1, 0, 1])
    ax.set_axis_off()
    ax.set_title(f"Customer CLV: ${dollar_pred:,.0f}", fontsize=11, pad=8)
    return fig


def _batch_predict(model, df: pd.DataFrame) -> pd.DataFrame:
    from src.modeling import LOG_PRED_MAX as _LP
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = FEATURE_DEFAULTS.get(col, 0.0)
    X = df[FEATURE_COLS].copy().fillna(
        {c: FEATURE_DEFAULTS.get(c, 0.0) for c in FEATURE_COLS}
    )
    log_preds    = np.clip(model.predict(X), 0, _LP)
    dollar_preds = np.expm1(log_preds)
    df = df.copy()
    df["CLV_Predicted_90d"] = np.round(dollar_preds, 2)
    df["CLV_Low"]           = np.round(np.maximum(0, dollar_preds * 0.85), 2)
    df["CLV_High"]          = np.round(dollar_preds * 1.15, 2)
    df["Segment"]           = [_get_segment(d) for d in dollar_preds]
    return df


# =============================================================================
# Anomaly Helpers
# =============================================================================

def _score_customer_anomaly(bundle: dict, feature_values: dict) -> dict:
    """Score a single customer through the anomaly bundle."""
    try:
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np

        if_model = bundle["isolation_forest"]
        ae_model = bundle.get("autoencoder")
        scaler   = bundle["scaler"]
        cols     = bundle["cols_used"]
        weights  = bundle.get("weights", {"isolation_forest": 0.45, "autoencoder": 0.55})

        row = pd.DataFrame([{c: feature_values.get(c, FEATURE_DEFAULTS.get(c, 0.0))
                              for c in cols}])
        X_scaled = scaler.transform(row).astype(np.float32)

        # IF score
        raw_if = -if_model.decision_function(X_scaled)
        # Normalise relative to training distribution offset
        if_score = float(np.clip(raw_if[0] / 0.5, 0, 1))

        # AE score
        ae_score = 0.0
        ae_available = False
        try:
            import torch
            if ae_model is not None:
                device = torch.device("cpu")
                ae_model_cpu = ae_model.to(device)
                tensor = torch.tensor(X_scaled, dtype=torch.float32)
                device2 = next(ae_model_cpu.parameters()).device
                tensor  = tensor.to(device2)
                with torch.no_grad():
                    recon  = ae_model_cpu(tensor)
                    error  = torch.mean((tensor - recon) ** 2).item()
                # Scale relative to p99 threshold (~0.49 from training)
                ae_score     = float(np.clip(error / 0.49, 0, 1))
                ae_available = True
        except Exception:
            pass

        if ae_available:
            combined = weights["isolation_forest"] * if_score + weights["autoencoder"] * ae_score
        else:
            combined = if_score

        combined = float(np.clip(combined, 0, 1))

        if combined >= 0.65:
            risk = "🔴 High"
        elif combined >= 0.40:
            risk = "🟡 Medium"
        else:
            risk = "🟢 Low"

        return {
            "combined": combined,
            "if_score": if_score,
            "ae_score": ae_score,
            "risk":     risk,
            "is_anomaly": combined >= bundle.get("thresholds", {}).get("anomaly_score", 0.50),
        }
    except Exception as e:
        return {"combined": 0.0, "if_score": 0.0, "ae_score": 0.0,
                "risk": "🟢 Low", "is_anomaly": False, "error": str(e)}


# =============================================================================
# Page Setup
# =============================================================================

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ Customer Intelligence Platform")
st.caption(
    "**Pipeline v2.5.0** — CLV Prediction · Customer Segmentation · Anomaly Detection"
)

# Load all bundles
clv_bundle    = load_clv_bundle()
seg_bundle    = load_seg_bundle()
anomaly_bundle = load_anomaly_bundle()
model         = clv_bundle["model"]
version       = clv_bundle.get("version", "?")
trained       = clv_bundle.get("timestamp", "unknown")

with st.sidebar:
    st.markdown("### 🏆 Model Info")
    st.info(
        f"**Champion:** {type(model).__name__}\n\n"
        f"**Version:** {version}\n\n"
        f"**Trained:** {trained[:10] if trained != 'unknown' else 'unknown'}"
    )
    st.markdown("---")
    st.markdown(
        "**CLV Segment thresholds:**\n"
        f"- 🐋 Whale: ≥ ${SEGMENT_P80:,.0f}\n"
        f"- 💰 Mid: ${SEGMENT_P20:,.0f} – ${SEGMENT_P80:,.0f}\n"
        f"- 📉 Low: $0.01 – ${SEGMENT_P20:,.0f}\n"
        f"- 💤 Churned: $0"
    )
    st.markdown("---")
    # Bundle status
    st.markdown("**Pipeline Status:**")
    st.markdown(f"{'✅' if seg_bundle else '⏳'} Segmentation bundle")
    st.markdown(f"{'✅' if anomaly_bundle else '⏳'} Anomaly bundle")

tab_single, tab_batch, tab_segments, tab_anomaly = st.tabs([
    "🔮 Single Customer",
    "📂 Batch CSV Upload",
    "🗂️ Customer Segments",
    "🚨 Anomaly Detection",
])


# =============================================================================
# Tab 1: Single Customer Predictor  (unchanged)
# =============================================================================

with tab_single:
    st.subheader("Input Customer Features")
    st.caption(
        "Adjust the 5 key levers below. Remaining 11 features are set to "
        "training-set medians."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        recency   = st.slider("Recency (days since first purchase)",
                              30, 730, 365, 10)
        frequency = st.slider("Frequency (number of invoices)", 1, 100, 4, 1)
    with col2:
        monetary   = st.slider("Monetary (avg order value, $)", 10.0, 5000.0, 300.0, 10.0)
        days_since = st.slider("Days Since Last Purchase", 1, 365, 60, 1)
    with col3:
        max_order   = st.slider("Max Single Order ($)", 10.0, 5000.0, 250.0, 10.0)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("▶️ Predict CLV", use_container_width=True, type="primary")

    if predict_btn:
        overrides   = {"Recency": float(recency), "Frequency": float(frequency),
                       "Monetary": float(monetary), "Days_Since_Purchase": float(days_since),
                       "Max_Single_Order": float(max_order)}
        feature_row = _build_feature_row(overrides)
        result      = _predict_clv(model, feature_row)
        segment     = _get_segment(result["dollar"])
        seg_color   = SEGMENT_CONFIG[segment]["color"]

        st.markdown("---")
        r1, r2, r3 = st.columns([2, 1, 2])
        with r1:
            st.metric("📈 Predicted 90-Day CLV", f"${result['dollar']:,.2f}",
                      f"Range: ${result['low']:,.0f} – ${result['high']:,.0f}")
        with r2:
            st.markdown(
                f'<div style="background:{seg_color};border-radius:10px;padding:14px 10px;'
                f'text-align:center;color:white;font-size:1.25rem;font-weight:600;">'
                f'{segment}</div>', unsafe_allow_html=True)
        with r3:
            st.markdown("**Feature inputs used**")
            st.dataframe(feature_row.T.rename(columns={0: "Value"}).round(3),
                         use_container_width=True, height=350)

        st.markdown("---")
        shap_col, lift_col = st.columns(2)
        with shap_col:
            st.subheader("🔍 SHAP Feature Contributions")
            if SHAP_AVAILABLE:
                fig_shap = _shap_waterfall_figure(model, feature_row)
                if fig_shap:
                    st.pyplot(fig_shap, use_container_width=True)
                    plt.close(fig_shap)
                else:
                    st.info("SHAP waterfall unavailable for this model type.")
            else:
                st.warning("Install `shap` to enable waterfall plots.")
        with lift_col:
            st.subheader("📊 Gain Chart Position")
            fig_lift = _gain_chart_figure(result["dollar"])
            if fig_lift:
                st.pyplot(fig_lift, use_container_width=True)
                plt.close(fig_lift)
            else:
                st.info("Run full pipeline to generate `business_lift.png`.")


# =============================================================================
# Tab 2: Batch CSV Upload  (unchanged)
# =============================================================================

with tab_batch:
    st.subheader("Batch Predict — Upload Customer CSV")
    st.caption(f"Upload a CSV with any subset of the 16 model features.")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded, encoding="ISO-8859-1")
            st.success(f"✅ Loaded {len(input_df):,} customers from `{uploaded.name}`")

            with st.spinner("Running predictions …"):
                result_df = _batch_predict(model, input_df)

            st.markdown("#### Preview — Predictions")
            preview_cols = [c for c in result_df.columns
                            if c in FEATURE_COLS[:5] or
                            c in ["CLV_Predicted_90d", "CLV_Low", "CLV_High", "Segment"]]
            st.dataframe(result_df[preview_cols].head(50), use_container_width=True)

            st.markdown("#### Segment Breakdown")
            seg_counts = (result_df["Segment"].value_counts()
                          .reset_index().rename(columns={"index": "Segment", "Segment": "Count"}))
            bc1, bc2 = st.columns([1, 2])
            with bc1:
                st.dataframe(seg_counts, use_container_width=True)
            with bc2:
                seg_agg = (result_df.groupby("Segment")["CLV_Predicted_90d"]
                           .agg(["count", "mean", "sum"])
                           .rename(columns={"count": "N", "mean": "Avg CLV ($)", "sum": "Total CLV ($)"})
                           .reset_index())
                st.dataframe(seg_agg.round(2), use_container_width=True)

            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=False)
            st.download_button("⬇️ Download Predictions as CSV",
                               data=csv_buffer.getvalue().encode("utf-8"),
                               file_name="clv_predictions.csv", mime="text/csv",
                               use_container_width=True, type="primary")
        except Exception as e:
            st.error(f"❌ Failed to process file: {e}")


# =============================================================================
# Tab 3: Customer Segments
# =============================================================================

with tab_segments:
    st.subheader("🗂️ Customer Segmentation Dashboard")

    if seg_bundle is None:
        st.warning(
            "⏳ Segmentation bundle not found. "
            "Run `run_segmentation_pipeline()` in `segmentation.ipynb` first."
        )
    else:
        # ── Header metrics ────────────────────────────────────────────────────
        metrics = seg_bundle.get("metrics", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Optimal Clusters (k)", seg_bundle.get("optimal_k", "?"))
        m2.metric("Silhouette Score",
                  f"{metrics.get('silhouette', 0):.4f}",
                  help="Higher is better (max 1.0)")
        m3.metric("Davies-Bouldin",
                  f"{metrics.get('davies_bouldin', 0):.4f}",
                  help="Lower is better")
        m4.metric("Calinski-Harabasz",
                  f"{int(metrics.get('calinski_harabasz', 0)):,}",
                  help="Higher is better")

        st.markdown("---")

        # ── Plots row ─────────────────────────────────────────────────────────
        st.markdown("#### Cluster Visualisations")
        plot_c1, plot_c2 = st.columns(2)

        with plot_c1:
            umap_path = _load_graph("seg_umap.png")
            if umap_path:
                st.image(str(umap_path), caption="UMAP — Coloured by Cluster",
                         use_container_width=True)
            else:
                st.info("UMAP plot not found (`seg_umap.png`).")

        with plot_c2:
            heatmap_path = _load_graph("seg_clv_heatmap.png")
            if heatmap_path:
                st.image(str(heatmap_path), caption="Segment × CLV Tier Heatmap",
                         use_container_width=True)
            else:
                st.info("CLV heatmap not found (`seg_clv_heatmap.png`).")

        # Second row of plots
        plot_c3, plot_c4 = st.columns(2)
        with plot_c3:
            profile_path = _load_graph("seg_cluster_profiles.png")
            if profile_path:
                st.image(str(profile_path), caption="Feature Heatmap per Cluster",
                         use_container_width=True)
        with plot_c4:
            elbow_path = _load_graph("seg_elbow_silhouette.png")
            if elbow_path:
                st.image(str(elbow_path), caption="Elbow + Silhouette — Optimal k Selection",
                         use_container_width=True)

        st.markdown("---")

        # ── Segment Business Summary ──────────────────────────────────────────
        st.markdown("#### Segment Business Summary")
        cust_df = load_customer_segments()

        if cust_df is not None and "Segment_Name" in cust_df.columns:
            # Build summary table
            clv_col = next((c for c in ["Predicted_CLV", "CLV_Predicted_90d", "Monetary"]
                            if c in cust_df.columns), None)

            if clv_col:
                summary = (
                    cust_df.groupby("Segment_Name")[clv_col]
                    .agg(N="count", Avg_CLV="mean", Total_CLV="sum")
                    .reset_index()
                )
                summary["Revenue_Share_%"] = (
                    summary["Total_CLV"] / summary["Total_CLV"].sum() * 100
                ).round(1)
                summary["Avg_CLV"]   = summary["Avg_CLV"].round(2)
                summary["Total_CLV"] = summary["Total_CLV"].round(2)
                st.dataframe(summary, use_container_width=True)
            else:
                # Fallback — just show counts
                summary = cust_df["Segment_Name"].value_counts().reset_index()
                summary.columns = ["Segment", "N"]
                st.dataframe(summary, use_container_width=True)
        else:
            st.info("Run segmentation pipeline to populate segment data.")

        st.markdown("---")

        # ── Segment Profiles Heatmap ──────────────────────────────────────────
        st.markdown("#### Feature Profiles per Segment")
        prof_df = load_segment_profiles()
        if prof_df is not None:
            st.dataframe(prof_df.round(3), use_container_width=True)

            # Render as heatmap
            try:
                import seaborn as sns
                numeric_cols = prof_df.select_dtypes(include=np.number).columns.tolist()
                id_col = prof_df.columns[0]  # first column is usually segment name/id

                if len(numeric_cols) > 0:
                    heat_data = prof_df.set_index(id_col)[numeric_cols] if id_col not in numeric_cols else prof_df[numeric_cols]
                    # Standardise columns for visual clarity
                    heat_std = (heat_data - heat_data.mean()) / (heat_data.std() + 1e-8)

                    fig, ax = plt.subplots(figsize=(14, max(4, len(heat_std) * 1.2)))
                    sns.heatmap(heat_std, annot=True, fmt=".2f", cmap="RdYlGn",
                                linewidths=0.3, ax=ax, cbar_kws={"label": "Standardised Value"})
                    ax.set_title("Segment Feature Profiles (standardised)", fontsize=12, pad=10)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            except Exception:
                pass  # dataframe view is sufficient fallback

        st.markdown("---")

        # ── Customer Lookup ───────────────────────────────────────────────────
        st.markdown("#### 🔎 Customer Segment Lookup")
        cust_df2 = load_customer_segments()

        if cust_df2 is not None:
            id_col = "CustomerID" if "CustomerID" in cust_df2.columns else cust_df2.columns[0]
            lookup_input = st.text_input(
                f"Enter {id_col}",
                placeholder="e.g. 12345",
                key="seg_lookup",
            )
            if lookup_input.strip():
                try:
                    # Try numeric match first, then string
                    try:
                        match = cust_df2[cust_df2[id_col] == int(lookup_input.strip())]
                    except ValueError:
                        match = cust_df2[cust_df2[id_col].astype(str) == lookup_input.strip()]

                    if len(match) == 0:
                        st.warning(f"No customer found with {id_col} = `{lookup_input}`")
                    else:
                        row = match.iloc[0]
                        lc1, lc2, lc3 = st.columns(3)

                        seg_name  = str(row.get("Segment_Name", row.get("Cluster", "Unknown")))
                        seg_color = SEG_COLOURS.get(seg_name, "#5C4DB1")

                        with lc1:
                            st.markdown(
                                f'<div style="background:{seg_color};border-radius:10px;'
                                f'padding:16px;text-align:center;color:white;'
                                f'font-size:1.2rem;font-weight:700;">'
                                f'Segment<br>{seg_name}</div>',
                                unsafe_allow_html=True,
                            )
                        with lc2:
                            if "Predicted_CLV" in row:
                                st.metric("Predicted CLV", f"${row['Predicted_CLV']:,.2f}")
                            if "CLV_Tier" in row:
                                st.metric("CLV Tier", str(row["CLV_Tier"]))
                        with lc3:
                            display_cols = [c for c in
                                            ["Recency", "Frequency", "Monetary",
                                             "Return_Rate", "Avg_Basket_Size"]
                                            if c in row.index]
                            if display_cols:
                                st.dataframe(
                                    pd.DataFrame(row[display_cols]).rename(columns={row.name: "Value"}).round(3),
                                    use_container_width=True,
                                )
                except Exception as e:
                    st.error(f"Lookup error: {e}")

        # ── Additional plots ──────────────────────────────────────────────────
        with st.expander("📊 More Visualisations", expanded=False):
            extra_plots = [
                ("seg_tsne.png",       "t-SNE Cluster Map"),
                ("seg_rfm_3d.png",     "3D RFM Scatter"),
                ("seg_dendrogram.png", "Hierarchical Dendrogram"),
                ("seg_pca_variance.png","PCA Scree Plot"),
                ("seg_dbscan_map.png", "DBSCAN Noise Map"),
            ]
            for fname, caption in extra_plots:
                p = _load_graph(fname)
                if p:
                    st.image(str(p), caption=caption, use_container_width=True)


# =============================================================================
# Tab 4: Anomaly Detection
# =============================================================================

with tab_anomaly:
    st.subheader("🚨 Customer Anomaly Detection")

    if anomaly_bundle is None:
        st.warning(
            "⏳ Anomaly bundle not found. "
            "Run `run_anomaly_pipeline()` in `segmentation.ipynb` first."
        )
    else:
        # ── Fleet metrics ──────────────────────────────────────────────────────
        a_metrics = anomaly_bundle.get("metrics", {})
        am1, am2, am3, am4 = st.columns(4)
        am1.metric("Total Customers", f"{a_metrics.get('n_total', 0):,}")
        am2.metric("Anomalies Flagged",
                   f"{a_metrics.get('n_anomaly', 0):,}",
                   f"{a_metrics.get('pct_anomaly', 0):.1f}%")
        am3.metric("High-Return Flags", f"{a_metrics.get('n_high_return', 0):,}")
        am4.metric("Whale Anomalies",   f"{a_metrics.get('n_whale_anomaly', 0):,}")

        st.markdown("---")

        # ── Diagnostic plots ───────────────────────────────────────────────────
        st.markdown("#### Anomaly Score Distributions")
        ap1, ap2 = st.columns(2)

        with ap1:
            dist_path = _load_graph("anomaly_score_distribution.png")
            if dist_path:
                st.image(str(dist_path), caption="Score Distribution + Threshold",
                         use_container_width=True)
            else:
                st.info("Run anomaly pipeline to generate plots.")

        with ap2:
            fi_path = _load_graph("anomaly_feature_importance.png")
            if fi_path:
                st.image(str(fi_path), caption="Feature Importance (IF + SHAP)",
                         use_container_width=True)

        ap3, ap4 = st.columns(2)
        with ap3:
            umap_path = _load_graph("anomaly_umap.png")
            if umap_path:
                st.image(str(umap_path), caption="UMAP — Coloured by Anomaly Score",
                         use_container_width=True)
        with ap4:
            ret_path = _load_graph("anomaly_return_flags.png")
            if ret_path:
                st.image(str(ret_path), caption="Return Rate vs Anomaly Score",
                         use_container_width=True)

        st.markdown("---")

        # ── Top anomalies table ────────────────────────────────────────────────
        st.markdown("#### Top Flagged Customers")
        anomaly_df = load_anomaly_scores()

        if anomaly_df is not None:
            top_cols = ["Anomaly_Score", "IF_Score", "AE_Score",
                        "is_anomaly", "is_high_return", "is_suspicious"]
            if "CustomerID" in anomaly_df.columns:
                top_cols = ["CustomerID"] + top_cols

            feature_display = [c for c in
                                ["Recency", "Frequency", "Monetary", "Return_Rate", "Max_Single_Order"]
                                if c in anomaly_df.columns]
            display_cols = top_cols + feature_display

            top50 = anomaly_df.nlargest(50, "Anomaly_Score")[
                [c for c in display_cols if c in anomaly_df.columns]
            ]

            # Colour-code anomaly score
            def _colour_score(val):
                if isinstance(val, (int, float)):
                    if val >= 0.65:   return "background-color: #E6394644"
                    elif val >= 0.50: return "background-color: #F4A26144"
                return ""

            st.dataframe(
                top50.style.map(_colour_score, subset=["Anomaly_Score"]),
                use_container_width=True,
                height=400,
            )

            # Download
            buf = io.StringIO()
            top50.to_csv(buf, index=False)
            st.download_button("⬇️ Download Top Anomalies CSV",
                               data=buf.getvalue().encode("utf-8"),
                               file_name="anomaly_top50.csv", mime="text/csv")

        st.markdown("---")

        # ── Single Customer Anomaly Scorer ─────────────────────────────────────
        st.markdown("#### 🔎 Score a Single Customer")
        st.caption("Enter customer feature values to get their live anomaly score.")

        cols_used = anomaly_bundle.get("cols_used", [
            "Recency", "Frequency", "Monetary", "Return_Rate",
            "Avg_Basket_Size", "Max_Single_Order", "Monetary_Percentile",
            "Purchase_Rate", "Unique_Products", "Days_Since_Purchase",
            "Interpurchase_Std", "Revenue_Per_Day",
        ])

        sc1, sc2, sc3 = st.columns(3)
        feature_inputs = {}

        slider_defs = {
            "Recency":            (1,    730,  365,  1),
            "Frequency":          (1,    200,  4,    1),
            "Monetary":           (0.0,  10000.0, 300.0, 10.0),
            "Return_Rate":        (0.0,  1.0,  0.04, 0.01),
            "Avg_Basket_Size":    (1.0,  50.0, 3.2,  0.1),
            "Max_Single_Order":   (0.0,  10000.0, 250.0, 10.0),
            "Monetary_Percentile":(0.0,  1.0,  0.50, 0.01),
            "Purchase_Rate":      (0.0,  0.5,  0.012,0.001),
            "Unique_Products":    (1,    200,  12,   1),
            "Days_Since_Purchase":(1,    730,  60,   1),
            "Interpurchase_Std":  (0.0,  200.0,45.0, 1.0),
            "Revenue_Per_Day":    (0.0,  50.0, 1.8,  0.1),
        }

        cols_cycle = [sc1, sc2, sc3]
        for i, col_name in enumerate(cols_used):
            if col_name in slider_defs:
                mn, mx, dv, st_val = slider_defs[col_name]
                with cols_cycle[i % 3]:
                    feature_inputs[col_name] = st.number_input(
                        col_name, min_value=float(mn), max_value=float(mx),
                        value=float(dv), step=float(st_val), key=f"an_{col_name}"
                    )

        score_btn = st.button("🚨 Calculate Anomaly Score", type="primary",
                              use_container_width=True, key="anomaly_score_btn")

        if score_btn:
            result = _score_customer_anomaly(anomaly_bundle, feature_inputs)
            st.markdown("---")
            rc1, rc2, rc3, rc4 = st.columns(4)

            risk_color = RISK_COLOURS.get(result["risk"], "#2A9D8F")
            rc1.markdown(
                f'<div style="background:{risk_color};border-radius:10px;padding:14px;'
                f'text-align:center;color:white;font-size:1.2rem;font-weight:700;">'
                f'Risk Level<br>{result["risk"]}</div>',
                unsafe_allow_html=True,
            )
            rc2.metric("Combined Score", f"{result['combined']:.4f}",
                       help="0 = normal, 1 = extreme anomaly")
            rc3.metric("Isolation Forest", f"{result['if_score']:.4f}")
            rc4.metric("Autoencoder",      f"{result['ae_score']:.4f}")

            if result["is_anomaly"]:
                st.error(
                    "⚠️ **This customer is flagged as anomalous.** "
                    "Review their transaction history for unusual patterns."
                )

                # Check high-return flag
                rr = feature_inputs.get("Return_Rate", 0)
                thresh = anomaly_bundle.get("thresholds", {}).get("return_rate", 0.30)
                if rr > thresh:
                    st.warning(
                        f"🔁 **High-return flag**: Return_Rate = {rr:.0%} "
                        f"(threshold: {thresh:.0%}). "
                        "Possible return fraud or systematic over-ordering."
                    )
            else:
                st.success("✅ Customer behaviour is within normal range.")

        # ── AE loss curve ───────────────────────────────────────────────────────
        with st.expander("📈 Autoencoder Training Loss Curve", expanded=False):
            ae_path = _load_graph("anomaly_reconstruction_error.png")
            if ae_path:
                st.image(str(ae_path), caption="AE Training Loss (MSE per epoch)",
                         use_container_width=True)
            top_path = _load_graph("anomaly_top_customers.png")
            if top_path:
                st.image(str(top_path), caption="Top 20 Anomalous Customer Profiles",
                         use_container_width=True)
                
                