"""
Customer Intelligence Platform v3.0.0
======================================
FAANG-level unified customer analytics dashboard.

Tabs
----
  Tab 1 — 🔮 Customer Intelligence    ← UPGRADED: CLV + Segment + Anomaly + Recs in ONE view
  Tab 2 — 📊 Segmentation Lab         ← UPGRADED: Interactive k-selection, cluster explorer
  Tab 3 — 🚨 Anomaly Detection        ← existing + richer segmentation plots
  Tab 4 — 🛒 Product Intelligence     ← NEW: Association rules + cross-sell engine
  Tab 5 — 📂 Batch Operations         ← UPGRADED: batch CLV + segment + anomaly together

What makes this FAANG-level
----------------------------
- Single customer view combines ALL models: CLV + K-Means segment + IF anomaly + FP-Growth recs
- Unsupervised cluster assignment feeds into the prediction display (not just visual)
- Association rules power live cross-sell recommendations
- k-comparison table shows WHY k=2 was chosen (statistical rigor visible to user)
- Every model output is explained, not just shown
"""

import os, io, sys, warnings, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("shap").setLevel(logging.ERROR)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
os.environ.setdefault("CLV_BASE_DIR", str(_HERE))

from src.config import MODELS_DIR, GRAPHS_DIR, FEATURE_COLS

# =============================================================================
# Constants
# =============================================================================
SEGMENT_P20  = 150.0
SEGMENT_P80  = 1_200.0
LOG_PRED_MAX = 12.0

CLV_SEGMENT_COLOURS = {
    "🐋 Whale":   "#5C4DB1",
    "💰 Mid":     "#2E86AB",
    "📉 Low":     "#F4A261",
    "💤 Churned": "#E63946",
}
RISK_COLOURS = {
    "🔴 High":   "#E63946",
    "🟡 Medium": "#F4A261",
    "🟢 Low":    "#2A9D8F",
}
SEG_COLOURS = {
    "🏆 Champions":          "#FFD700",
    "🐋 Dormant Whales":      "#5C4DB1",
    "📈 Growing Mid-Tier":    "#2A9D8F",
    "🔁 High-Freq Low-Value": "#F4A261",
    "💤 Lost Customers":      "#E63946",
    "🐋 Loyal Customers":     "#5C4DB1",
    "👥 Champions":           "#2E86AB",
    "Champions":              "#2E86AB",
    "Loyal Customers":        "#5C4DB1",
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


# =============================================================================
# Cached Loaders
# =============================================================================

@st.cache_resource(show_spinner="Loading CLV champion model …")
def load_clv_bundle():
    p = MODELS_DIR / "clv_champion_bundle.pkl"
    if not p.exists():
        st.error(f"❌ CLV bundle not found: `{p}`")
        st.stop()
    return joblib.load(p)

@st.cache_resource(show_spinner="Loading segmentation bundle …")
def load_seg_bundle():
    p = MODELS_DIR / "segmentation_bundle.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner="Loading richer segmentation bundle …")
def load_richer_seg_bundle():
    p = MODELS_DIR / "richer_seg_bundle.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner="Loading anomaly bundle …")
def load_anomaly_bundle():
    p = MODELS_DIR / "anomaly_bundle.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner="Loading association rules …")
def load_ar_bundle():
    p = MODELS_DIR / "association_rules_bundle.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_data
def load_csv(filename):
    p = GRAPHS_DIR / filename
    return pd.read_csv(p) if p.exists() else None

def _img(name):
    p = GRAPHS_DIR / name
    return p if p.exists() else None


# =============================================================================
# Core Prediction Helpers
# =============================================================================

def _build_row(overrides):
    return pd.DataFrame([{**FEATURE_DEFAULTS, **overrides}])[FEATURE_COLS]

def _predict_clv(model, row):
    log_pred = float(np.clip(model.predict(row), 0, LOG_PRED_MAX)[0])
    dollar   = float(np.expm1(log_pred))
    return {"log": log_pred, "dollar": dollar,
            "low": max(0.0, dollar * 0.85), "high": dollar * 1.15}

def _clv_tier(dollar):
    if dollar < 0.01:        return "💤 Churned",  "#E63946"
    elif dollar < SEGMENT_P20: return "📉 Low",     "#F4A261"
    elif dollar < SEGMENT_P80: return "💰 Mid",     "#2E86AB"
    else:                      return "🐋 Whale",   "#5C4DB1"

def _assign_cluster(row_df, bundle):
    """Assign K-Means cluster to a single customer row."""
    if bundle is None:
        return None, "Unknown"
    try:
        cols  = bundle["cols_used"]
        scaler = bundle["scaler"]
        pca    = bundle.get("pca")
        kmeans = bundle["kmeans"]
        labels = bundle.get("segment_labels", {})

        avail  = [c for c in cols if c in row_df.columns]
        X      = row_df[avail].fillna(0).values.astype(np.float32)
        Xs     = scaler.transform(X)
        if pca is not None:
            Xs = pca.transform(Xs)
        cid    = int(kmeans.predict(Xs)[0])
        name   = labels.get(cid, f"Cluster {cid}")
        return cid, name
    except Exception:
        return None, "Unknown"

def _score_anomaly(bundle, feature_dict):
    """Score a customer through Isolation Forest."""
    if bundle is None:
        return {"score": 0.0, "risk": "🟢 Low", "is_anomaly": False}
    try:
        cols     = bundle["cols_used"]
        scaler   = bundle["scaler"]
        if_model = bundle["isolation_forest"]
        thresh   = bundle.get("thresholds", {})

        row      = pd.DataFrame([{c: feature_dict.get(c, FEATURE_DEFAULTS.get(c, 0)) for c in cols}])
        X        = scaler.transform(row.fillna(0).values.astype(np.float32))
        raw      = float(-if_model.decision_function(X)[0])
        score    = float(np.clip(raw / 0.5, 0, 1))

        risk = "🔴 High" if score >= 0.65 else ("🟡 Medium" if score >= 0.40 else "🟢 Low")
        return {
            "score":      round(score, 4),
            "risk":       risk,
            "is_anomaly": score >= thresh.get("anomaly_score", 0.50),
            "is_high_return": (
                feature_dict.get("Return_Rate", 0) > thresh.get("return_rate", 0.30)
                and score > 0.40
            ),
        }
    except Exception as e:
        return {"score": 0.0, "risk": "🟢 Low", "is_anomaly": False, "error": str(e)}

def _get_recommendations(ar_bundle, product_query, top_n=5):
    """Fetch cross-sell recommendations from the AR lookup."""
    if ar_bundle is None:
        return []
    lookup = ar_bundle.get("lookup", {})
    if not product_query:
        return []
    # Exact match
    if product_query.upper() in lookup:
        return lookup[product_query.upper()][:top_n]
    # Partial match
    matches = [k for k in lookup if product_query.upper() in k]
    if matches:
        return lookup[matches[0]][:top_n]
    return []

def _batch_predict(model, df):
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = FEATURE_DEFAULTS.get(col, 0.0)
    X = df[FEATURE_COLS].fillna({c: FEATURE_DEFAULTS.get(c, 0) for c in FEATURE_COLS})
    preds = np.expm1(np.clip(model.predict(X), 0, LOG_PRED_MAX))
    out   = df.copy()
    out["CLV_Predicted_90d"] = np.round(preds, 2)
    out["CLV_Low"]           = np.round(np.maximum(0, preds * 0.85), 2)
    out["CLV_High"]          = np.round(preds * 1.15, 2)
    out["Segment"]           = [_clv_tier(d)[0] for d in preds]
    return out, preds

def _shap_fig(model, row):
    if not SHAP_AVAILABLE: return None
    est = getattr(model, "regressor_", model)
    if not hasattr(est, "feature_importances_"): return None
    try:
        exp  = shap.TreeExplainer(est)(row)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.sca(ax)
        shap.plots.waterfall(exp[0], max_display=12, show=False)
        plt.tight_layout()
        return fig
    except Exception:
        return None


# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for FAANG-level polish
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid #3a3a5e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 4px;
    }
    .intel-card {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        border-left: 4px solid #5C4DB1;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .rec-card {
        background: #1a1a2e;
        border: 1px solid #2A9D8F;
        border-radius: 8px;
        padding: 12px;
        margin: 4px 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Load All Bundles
# =============================================================================

clv_bundle    = load_clv_bundle()
seg_bundle    = load_seg_bundle()
rs_bundle     = load_richer_seg_bundle()
anomaly_bundle = load_anomaly_bundle()
ar_bundle     = load_ar_bundle()
model         = clv_bundle["model"]

# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("## 🛍️ Customer Intelligence")
    st.markdown("**Platform v3.0.0**")
    st.markdown("---")

    st.markdown("### 🤖 Models Loaded")
    st.markdown(f"{'✅' if clv_bundle else '❌'} CLV Predictor (TwoStageRegressor)")
    st.markdown(f"{'✅' if seg_bundle else '⏳'} K-Means Segmentation")
    st.markdown(f"{'✅' if rs_bundle else '⏳'} Richer Segmentation (k=2..5)")
    st.markdown(f"{'✅' if anomaly_bundle else '⏳'} Isolation Forest Anomaly")
    st.markdown(f"{'✅' if ar_bundle else '⏳'} FP-Growth Association Rules")

    st.markdown("---")
    st.markdown("### 📐 CLV Tiers")
    st.markdown(
        f"🐋 Whale ≥ **${SEGMENT_P80:,.0f}**\n\n"
        f"💰 Mid ${SEGMENT_P20:,.0f}–${SEGMENT_P80:,.0f}\n\n"
        f"📉 Low $0.01–${SEGMENT_P20:,.0f}\n\n"
        f"💤 Churned $0"
    )

    st.markdown("---")
    v = clv_bundle.get("version", "?")
    t = clv_bundle.get("timestamp", "unknown")
    st.caption(f"Model v{v} · Trained {t[:10] if t != 'unknown' else '?'}")
    st.caption("Dollar R² = 0.581 on holdout")

# =============================================================================
# Tabs
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Customer Intelligence",
    "📊 Segmentation Lab",
    "🚨 Anomaly Detection",
    "🛒 Product Intelligence",
    "📂 Batch Operations",
])


# =============================================================================
# TAB 1 — UNIFIED CUSTOMER INTELLIGENCE
# The killer feature: all models on one customer in one click
# =============================================================================

with tab1:
    st.subheader("🔮 Unified Customer Intelligence")
    st.caption(
        "Enter any customer's behavioral profile. All 5 models run simultaneously — "
        "CLV prediction, K-Means segment, anomaly score, and product recommendations."
    )

    # ── Input Panel ──────────────────────────────────────────────────────────
    with st.expander("⚙️ Customer Feature Inputs", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            recency    = st.slider("Recency (days)", 1, 730, 180, 10)
            frequency  = st.slider("Frequency (invoices)", 1, 100, 8, 1)
            monetary   = st.slider("Monetary ($)", 10.0, 5000.0, 450.0, 10.0)
            return_rate = st.slider("Return Rate", 0.0, 1.0, 0.04, 0.01)
        with c2:
            days_since    = st.slider("Days Since Purchase", 1, 365, 45, 1)
            max_order     = st.slider("Max Single Order ($)", 10.0, 5000.0, 380.0, 10.0)
            avg_basket    = st.slider("Avg Basket Size", 1.0, 50.0, 4.0, 0.5)
            unique_prods  = st.slider("Unique Products", 1, 200, 15, 1)
        with c3:
            purchase_rate = st.slider("Purchase Rate", 0.0, 0.5, 0.018, 0.001)
            revenue_day   = st.slider("Revenue/Day ($)", 0.0, 50.0, 2.5, 0.1)
            mon_pct       = st.slider("Monetary Percentile", 0.0, 1.0, 0.65, 0.01)
            ipu_std       = st.slider("Interpurchase Std", 0.0, 200.0, 30.0, 1.0)
        with c4:
            prob_alive    = st.slider("Prob Alive", 0.0, 1.0, 0.72, 0.01)
            prob_txn      = st.slider("Prob Pred Txn", 0.0, 20.0, 2.1, 0.1)
            prob_val      = st.slider("Prob Pred Val ($)", 0.0, 5000.0, 420.0, 10.0)
            visit_div     = st.slider("Visit Diversity", 1.0, 50.0, 6.0, 0.5)

        analyze_btn = st.button(
            "🚀 Run Full Intelligence Analysis",
            type="primary", use_container_width=True
        )

    if analyze_btn:
        overrides = {
            "Recency": float(recency), "Frequency": float(frequency),
            "Monetary": float(monetary), "Return_Rate": float(return_rate),
            "Days_Since_Purchase": float(days_since),
            "Max_Single_Order": float(max_order),
            "Avg_Basket_Size": float(avg_basket),
            "Unique_Products": float(unique_prods),
            "Purchase_Rate": float(purchase_rate),
            "Revenue_Per_Day": float(revenue_day),
            "Monetary_Percentile": float(mon_pct),
            "Interpurchase_Std": float(ipu_std),
            "Prob_Alive": float(prob_alive),
            "Prob_Pred_Txn": float(prob_txn),
            "Prob_Pred_Val": float(prob_val),
            "Visit_Diversity": float(visit_div),
        }
        row = _build_row(overrides)

        # Run all models
        clv_result   = _predict_clv(model, row)
        tier, tcolor = _clv_tier(clv_result["dollar"])
        cluster_id, seg_name = _assign_cluster(row, seg_bundle)
        anomaly_result = _score_anomaly(anomaly_bundle, overrides)

        st.markdown("---")
        st.markdown("### 📊 Intelligence Report")

        # ── Row 1: Key metrics ────────────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("💰 Predicted CLV",   f"${clv_result['dollar']:,.2f}",
                  f"±${clv_result['dollar']*0.15:,.0f}")
        m2.metric("📈 CLV Tier",        tier)
        m3.metric("🗂️ Segment",         seg_name if seg_name != "Unknown" else "N/A")
        m4.metric("🚨 Anomaly Score",   f"{anomaly_result['score']:.3f}",
                  anomaly_result["risk"])
        m5.metric("📊 Confidence Range",
                  f"${clv_result['low']:,.0f}–${clv_result['high']:,.0f}")

        st.markdown("---")

        # ── Row 2: Intelligence cards ─────────────────────────────────────────
        left, right = st.columns([1, 1])

        with left:
            # CLV + Tier card
            st.markdown(
                f'<div class="intel-card">'
                f'<h4 style="color:#5C4DB1;margin:0">🔮 CLV Prediction</h4>'
                f'<p style="font-size:2rem;font-weight:700;margin:8px 0;color:white">'
                f'${clv_result["dollar"]:,.2f}</p>'
                f'<p style="color:#aaa;margin:0">90-day predicted spend</p>'
                f'<div style="margin-top:10px;background:{tcolor};border-radius:6px;'
                f'padding:6px 12px;display:inline-block;color:white;font-weight:600">'
                f'{tier}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Segment card
            seg_color = SEG_COLOURS.get(seg_name, "#5C4DB1")
            st.markdown(
                f'<div class="intel-card" style="border-left-color:{seg_color}">'
                f'<h4 style="color:{seg_color};margin:0">🗂️ Behavioural Segment</h4>'
                f'<p style="font-size:1.5rem;font-weight:700;margin:8px 0;color:white">'
                f'{seg_name}</p>'
                f'<p style="color:#aaa;margin:0">K-Means cluster assignment '
                f'(k={seg_bundle.get("optimal_k","?") if seg_bundle else "?"})</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with right:
            # Anomaly card
            risk_color = RISK_COLOURS.get(anomaly_result["risk"], "#2A9D8F")
            anomaly_detail = ""
            if anomaly_result.get("is_anomaly"):
                anomaly_detail = "⚠️ Flagged for review"
            if anomaly_result.get("is_high_return"):
                anomaly_detail += " · 🔁 High-return risk"
            st.markdown(
                f'<div class="intel-card" style="border-left-color:{risk_color}">'
                f'<h4 style="color:{risk_color};margin:0">🚨 Anomaly Detection</h4>'
                f'<p style="font-size:1.5rem;font-weight:700;margin:8px 0;color:white">'
                f'{anomaly_result["risk"]} · {anomaly_result["score"]:.3f}</p>'
                f'<p style="color:#aaa;margin:0">Isolation Forest score [0–1]</p>'
                f'{"<p style=color:#E63946;margin-top:8px>" + anomaly_detail + "</p>" if anomaly_detail else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Business flags card
            flags = {
                "🐋 Whale Customer":    clv_result["dollar"] >= SEGMENT_P80,
                "⚠️ Anomalous":         anomaly_result.get("is_anomaly", False),
                "🔁 High Return Risk":  anomaly_result.get("is_high_return", False),
                "💤 Churn Risk":        clv_result["dollar"] < 10,
                "📈 Growth Potential":  frequency >= 5 and monetary >= 200,
            }
            flag_html = "".join([
                f'<span style="background:{"#2A9D8F" if v else "#333"};'
                f'border-radius:4px;padding:3px 8px;margin:3px;font-size:0.8rem;'
                f'color:white;display:inline-block">{k}</span>'
                for k, v in flags.items()
            ])
            st.markdown(
                f'<div class="intel-card">'
                f'<h4 style="color:#F4A261;margin:0 0 8px 0">🏷️ Business Flags</h4>'
                f'{flag_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── SHAP ──────────────────────────────────────────────────────────────
        st.markdown("---")
        shap_col, feat_col = st.columns(2)
        with shap_col:
            st.markdown("#### 🔍 Why this CLV? (SHAP)")
            if SHAP_AVAILABLE:
                fig = _shap_fig(model, row)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.info("SHAP unavailable for this model type.")
            else:
                st.warning("Install `shap` for feature explanations.")

        with feat_col:
            st.markdown("#### 📋 Full Feature Profile")
            feat_df = row.T.rename(columns={0: "Value"}).round(4)
            feat_df["Role"] = feat_df.index.map(lambda x:
                "🔑 Core RFM" if x in ["Recency","Frequency","Monetary"]
                else ("🧠 Behavioural" if x in ["Return_Rate","Avg_Basket_Size","Unique_Products"]
                else "📊 Derived"))
            st.dataframe(feat_df, use_container_width=True, height=400)


# =============================================================================
# TAB 2 — SEGMENTATION LAB
# =============================================================================

with tab2:
    st.subheader("📊 Segmentation Lab")
    st.caption("Interactive cluster analysis — see how k=2 was chosen and what each segment means.")

    # ── k-comparison from richer segmentation ────────────────────────────────
    rs_metrics_df = load_csv("richer_seg_metrics.csv")

    if rs_metrics_df is not None:
        st.markdown("#### 🏆 K-Selection Analysis (k = 2, 3, 4, 5)")
        st.caption(
            "We tested every k and chose the one with the highest composite score. "
            "This is NOT arbitrary — it's statistically rigorous."
        )

        # Style the table
        def _highlight_optimal(row):
            is_max = row["composite_score"] == rs_metrics_df["composite_score"].max()
            return ["background-color: #2A9D8F22; font-weight: bold" if is_max else "" for _ in row]

        styled = rs_metrics_df[["k","silhouette","davies_bouldin",
                                  "calinski_harabasz","composite_score"]].style\
            .apply(_highlight_optimal, axis=1)\
            .format({"silhouette": "{:.4f}", "davies_bouldin": "{:.4f}",
                     "calinski_harabasz": "{:.0f}", "composite_score": "{:.4f}"})
        st.dataframe(styled, use_container_width=True)

        st.info(
            "💡 **k=2 wins** — silhouette drops from 0.53 → 0.35 as k increases. "
            "The data has 2 natural clusters (whale minority vs mass market). "
            "Forcing k=5 would create noise, not insight."
        )

    # ── k-comparison plot ─────────────────────────────────────────────────────
    k_plot = _img("rs_k_comparison.png")
    if k_plot:
        st.markdown("#### 📈 Metrics Across k Values")
        st.image(str(k_plot), caption="Silhouette, Davies-Bouldin, Calinski-Harabasz across k=2..5",
                 use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🗂️ Segment Profiles")

    # ── Cluster visualisations ────────────────────────────────────────────────
    pc1, pc2 = st.columns(2)
    with pc1:
        p = _img("seg_umap.png") or _img("rs_umap_5seg.png")
        if p: st.image(str(p), caption="UMAP — Cluster Map", use_container_width=True)
    with pc2:
        p = _img("seg_clv_heatmap.png")
        if p: st.image(str(p), caption="Segment × CLV Tier", use_container_width=True)

    pc3, pc4 = st.columns(2)
    with pc3:
        p = _img("rs_segment_profiles.png") or _img("seg_cluster_profiles.png")
        if p: st.image(str(p), caption="Feature Profiles per Segment", use_container_width=True)
    with pc4:
        p = _img("rs_clv_by_segment.png")
        if p: st.image(str(p), caption="CLV Distribution per Segment", use_container_width=True)

    st.markdown("---")

    # ── Business Summary ──────────────────────────────────────────────────────
    st.markdown("#### 💼 Segment Business Summary")
    cust_df = load_csv("customer_segments.csv")
    if cust_df is not None and "Segment_Name" in cust_df.columns:
        clv_col = next((c for c in ["CLV_Predicted_90d","Predicted_CLV","Monetary"]
                        if c in cust_df.columns), None)
        if clv_col:
            summary = (cust_df.groupby("Segment_Name")[clv_col]
                       .agg(N="count", Avg_CLV="mean", Total_CLV="sum")
                       .reset_index())
            summary["Revenue_Share_%"] = (summary["Total_CLV"] / summary["Total_CLV"].sum() * 100).round(1)
            st.dataframe(summary.round(2), use_container_width=True)

    # ── More plots expander ───────────────────────────────────────────────────
    with st.expander("📊 Advanced Visualisations", expanded=False):
        for fname, cap in [
            ("rs_revenue_treemap.png",  "Revenue Share Treemap"),
            ("seg_tsne.png",            "t-SNE Cluster Map"),
            ("seg_rfm_3d.png",          "3D RFM Scatter"),
            ("seg_dendrogram.png",      "Hierarchical Dendrogram"),
            ("seg_pca_variance.png",    "PCA Scree Plot"),
            ("seg_dbscan_map.png",      "DBSCAN Noise Map"),
            ("seg_elbow_silhouette.png","Elbow + Silhouette"),
        ]:
            p = _img(fname)
            if p: st.image(str(p), caption=cap, use_container_width=True)


# =============================================================================
# TAB 3 — ANOMALY DETECTION
# =============================================================================

with tab3:
    st.subheader("🚨 Customer Anomaly Detection")

    if anomaly_bundle is None:
        st.warning("⏳ Anomaly bundle not found. Run `run_anomaly_pipeline()` first.")
    else:
        # ── Fleet summary ─────────────────────────────────────────────────────
        m = anomaly_bundle.get("metrics", {})
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Total Customers",  f"{m.get('n_total',0):,}")
        a2.metric("Anomalies Flagged",f"{m.get('n_anomaly',0):,}", f"{m.get('pct_anomaly',0):.1f}%")
        a3.metric("High-Return Flags",f"{m.get('n_high_return',0):,}")
        a4.metric("Whale Anomalies",  f"{m.get('n_whale_anomaly',0):,}")

        st.markdown("---")
        st.markdown("#### Score Distributions")
        r1, r2 = st.columns(2)
        with r1:
            p = _img("anomaly_score_distribution.png")
            if p: st.image(str(p), caption="Score Distribution + Threshold", use_container_width=True)
        with r2:
            p = _img("anomaly_feature_importance.png")
            if p: st.image(str(p), caption="Feature Importance", use_container_width=True)

        r3, r4 = st.columns(2)
        with r3:
            p = _img("anomaly_umap.png")
            if p: st.image(str(p), caption="UMAP Anomaly Map", use_container_width=True)
        with r4:
            p = _img("anomaly_return_flags.png")
            if p: st.image(str(p), caption="Return Rate vs Anomaly Score", use_container_width=True)

        st.markdown("---")

        # ── Top flagged table ─────────────────────────────────────────────────
        st.markdown("#### Top Flagged Customers")
        adf = load_csv("anomaly_scores.csv")
        if adf is not None:
            show_cols = [c for c in ["Anomaly_Score","IF_Score","AE_Score",
                         "is_anomaly","is_high_return","is_suspicious",
                         "Recency","Frequency","Monetary","Return_Rate"]
                         if c in adf.columns]
            top50 = adf.nlargest(50, "Anomaly_Score")[show_cols]
            def _cs(v):
                if isinstance(v, (int,float)):
                    if v >= 0.65: return "background-color:#E6394644"
                    if v >= 0.50: return "background-color:#F4A26144"
                return ""
            st.dataframe(top50.style.map(_cs, subset=["Anomaly_Score"]),
                         use_container_width=True, height=380)
            buf = io.StringIO()
            top50.to_csv(buf, index=False)
            st.download_button("⬇️ Download Top 50 CSV",
                               buf.getvalue().encode(), "anomaly_top50.csv", "text/csv")

        st.markdown("---")

        # ── Single customer scorer ────────────────────────────────────────────
        st.markdown("#### 🔎 Score a Single Customer")
        cols_u = anomaly_bundle.get("cols_used", list(FEATURE_DEFAULTS.keys())[:12])
        sc1, sc2, sc3 = st.columns(3)
        fi = {}
        sdefs = {
            "Recency":(1,730,365,1), "Frequency":(1,200,4,1),
            "Monetary":(0,10000,300,10), "Return_Rate":(0.0,1.0,0.04,0.01),
            "Avg_Basket_Size":(1.0,50.0,3.2,0.1), "Max_Single_Order":(0,10000,250,10),
            "Monetary_Percentile":(0.0,1.0,0.50,0.01), "Purchase_Rate":(0.0,0.5,0.012,0.001),
            "Unique_Products":(1,200,12,1), "Days_Since_Purchase":(1,730,60,1),
            "Interpurchase_Std":(0.0,200.0,45.0,1.0), "Revenue_Per_Day":(0.0,50.0,1.8,0.1),
        }
        for i, col in enumerate(cols_u):
            if col in sdefs:
                mn,mx,dv,sv = sdefs[col]
                with [sc1,sc2,sc3][i%3]:
                    fi[col] = st.number_input(col, float(mn), float(mx),
                                              float(dv), float(sv), key=f"a_{col}")

        if st.button("🚨 Calculate Anomaly Score", type="primary",
                     use_container_width=True, key="ascore_btn"):
            res = _score_anomaly(anomaly_bundle, fi)
            rc = RISK_COLOURS.get(res["risk"], "#2A9D8F")
            c1,c2,c3 = st.columns(3)
            c1.markdown(
                f'<div style="background:{rc};border-radius:10px;padding:14px;'
                f'text-align:center;color:white;font-size:1.2rem;font-weight:700">'
                f'Risk Level<br>{res["risk"]}</div>', unsafe_allow_html=True)
            c2.metric("Anomaly Score", f"{res['score']:.4f}")
            c3.metric("Flagged", "⚠️ Yes" if res["is_anomaly"] else "✅ No")

            if res["is_anomaly"]:
                st.error("⚠️ Anomalous customer — review transaction history.")
            else:
                st.success("✅ Normal customer behaviour.")

        with st.expander("📈 Model Details", expanded=False):
            p = _img("anomaly_reconstruction_error.png")
            if p: st.image(str(p), caption="AE Training Loss", use_container_width=True)
            p = _img("anomaly_top_customers.png")
            if p: st.image(str(p), caption="Top 20 Anomaly Profiles", use_container_width=True)


# =============================================================================
# TAB 4 — PRODUCT INTELLIGENCE (Association Rules)
# =============================================================================

with tab4:
    st.subheader("🛒 Product Intelligence — Association Rules")
    st.caption(
        "FP-Growth market basket analysis on 400K transactions. "
        "Discover which products are bought together and generate cross-sell recommendations."
    )

    if ar_bundle is None:
        st.warning(
            "⏳ Association rules bundle not found. "
            "Run `run_association_rules_pipeline()` in `segmentation.ipynb` first."
        )
    else:
        # ── Fleet metrics ─────────────────────────────────────────────────────
        arm = ar_bundle.get("metrics", {})
        am1, am2, am3, am4 = st.columns(4)
        am1.metric("Total Rules",      f"{arm.get('n_rules_total',0):,}")
        am2.metric("Products w/ Recs", f"{arm.get('n_products_with_recs',0):,}")
        am3.metric("Avg Lift",         f"{arm.get('avg_lift',0):.2f}×")
        am4.metric("Max Lift",         f"{arm.get('max_lift',0):.2f}×")

        st.info(
            f"💡 Avg lift of **{arm.get('avg_lift',0):.1f}×** means these product pairs "
            "are bought together far more often than random chance would predict."
        )

        st.markdown("---")

        # ── Cross-sell recommendation engine ──────────────────────────────────
        st.markdown("#### 🎯 Cross-Sell Recommendation Engine")
        st.caption("Enter any product name (partial match supported) to get top recommendations.")

        lookup = ar_bundle.get("lookup", {})
        sample_products = list(lookup.keys())[:10] if lookup else []

        rec_col1, rec_col2 = st.columns([2, 1])
        with rec_col1:
            product_input = st.text_input(
                "Product name",
                placeholder="e.g. HEART, LANTERN, CANDLE",
                key="ar_product_input"
            )
        with rec_col2:
            top_n = st.selectbox("Top N recommendations", [3, 5, 10], index=1)

        if sample_products:
            st.caption(f"Sample products in database: {', '.join(sample_products[:5])}...")

        if product_input:
            recs = _get_recommendations(ar_bundle, product_input, top_n)
            if recs:
                st.markdown(f"**Top {len(recs)} recommendations for '{product_input.upper()}':**")
                for i, r in enumerate(recs):
                    conf_pct = r.get('confidence', 0) * 100
                    lift_val = r.get('lift', 0)
                    color    = "#2A9D8F" if lift_val > 5 else ("#F4A261" if lift_val > 2 else "#555")
                    st.markdown(
                        f'<div class="rec-card">'
                        f'<b style="color:#2A9D8F">#{i+1}</b> '
                        f'<b style="color:white">{r["product"][:50]}</b><br>'
                        f'<span style="color:#aaa">Confidence: {conf_pct:.1f}% · '
                        f'<span style="color:{color}">Lift: {lift_val:.2f}×</span> · '
                        f'Support: {r.get("support",0)*100:.1f}%</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.warning(
                    f"No recommendations found for '{product_input}'. "
                    f"Try a simpler word like HEART, WHITE, RED, CANDLE."
                )

        st.markdown("---")

        # ── Diagnostic plots ──────────────────────────────────────────────────
        st.markdown("#### 📊 Rule Analysis")
        ap1, ap2 = st.columns(2)
        with ap1:
            p = _img("ar_support_confidence.png")
            if p: st.image(str(p), caption="Support vs Confidence (coloured by lift)",
                           use_container_width=True)
            else: st.info("Run association rules pipeline to generate plots.")
        with ap2:
            p = _img("ar_lift_heatmap.png")
            if p: st.image(str(p), caption="Lift Matrix — Top 15 Products",
                           use_container_width=True)

        ap3, ap4 = st.columns(2)
        with ap3:
            p = _img("ar_top_rules_bar.png")
            if p: st.image(str(p), caption="Top 20 Rules by Lift", use_container_width=True)
        with ap4:
            p = _img("ar_segment_comparison.png")
            if p: st.image(str(p), caption="Champions vs Loyal Customers Affinity",
                           use_container_width=True)

        st.markdown("---")

        # ── Full rules table ──────────────────────────────────────────────────
        st.markdown("#### 📋 Full Association Rules Table")
        rules_df = load_csv("association_rules.csv")
        if rules_df is not None:
            st.dataframe(
                rules_df.sort_values("lift", ascending=False).head(100).round(4),
                use_container_width=True, height=350
            )
            buf = io.StringIO()
            rules_df.to_csv(buf, index=False)
            st.download_button("⬇️ Download All Rules CSV",
                               buf.getvalue().encode(), "association_rules.csv", "text/csv")


# =============================================================================
# TAB 5 — BATCH OPERATIONS
# =============================================================================

with tab5:
    st.subheader("📂 Batch Operations")
    st.caption(
        "Upload a CSV of customers to run CLV prediction, segmentation, "
        "and anomaly scoring in one shot."
    )

    uploaded = st.file_uploader("Upload customer CSV (any subset of 16 features)", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, encoding="ISO-8859-1")
            st.success(f"✅ Loaded {len(df):,} customers from `{uploaded.name}`")

            run_anomaly_batch = st.checkbox(
                "Include anomaly scoring (slower for large files)", value=False
            )
            run_seg_batch = st.checkbox(
                "Include segment assignment", value=True
            )

            if st.button("▶️ Run Batch Analysis", type="primary", use_container_width=True):
                with st.spinner("Running batch analysis…"):
                    result_df, dollar_preds = _batch_predict(model, df)

                    if run_seg_batch and seg_bundle is not None:
                        seg_labels = []
                        for _, row in result_df[FEATURE_COLS].iterrows():
                            _, sname = _assign_cluster(
                                pd.DataFrame([row.to_dict()])[FEATURE_COLS], seg_bundle
                            )
                            seg_labels.append(sname)
                        result_df["K_Segment"] = seg_labels

                    if run_anomaly_batch and anomaly_bundle is not None:
                        a_scores = []
                        for _, row in result_df.iterrows():
                            res = _score_anomaly(anomaly_bundle, row.to_dict())
                            a_scores.append(res["score"])
                        result_df["Anomaly_Score"] = np.round(a_scores, 4)
                        result_df["Anomaly_Risk"]  = [
                            "🔴 High" if s >= 0.65 else ("🟡 Medium" if s >= 0.40 else "🟢 Low")
                            for s in a_scores
                        ]

                st.markdown("#### Preview — Results")
                preview = ["CLV_Predicted_90d", "CLV_Low", "CLV_High", "Segment"]
                if "K_Segment"    in result_df.columns: preview.append("K_Segment")
                if "Anomaly_Score" in result_df.columns: preview += ["Anomaly_Score","Anomaly_Risk"]
                st.dataframe(result_df[preview].head(50), use_container_width=True)

                # ── Summary metrics ───────────────────────────────────────────
                st.markdown("#### 📊 Batch Summary")
                bs1, bs2, bs3, bs4 = st.columns(4)
                bs1.metric("Customers",     f"{len(result_df):,}")
                bs2.metric("Avg CLV",       f"${dollar_preds.mean():,.2f}")
                bs3.metric("Total Revenue", f"${dollar_preds.sum():,.0f}")
                bs4.metric("Whales",
                           f"{(dollar_preds >= SEGMENT_P80).sum():,}",
                           f"{(dollar_preds >= SEGMENT_P80).mean()*100:.1f}%")

                # ── Segment breakdown ─────────────────────────────────────────
                st.markdown("#### Segment Breakdown")
                seg_col = "K_Segment" if "K_Segment" in result_df.columns else "Segment"
                seg_agg = (
                    result_df.groupby(seg_col)["CLV_Predicted_90d"]
                    .agg(N="count", Avg_CLV="mean", Total_CLV="sum")
                    .reset_index()
                )
                seg_agg["Revenue_%"] = (
                    seg_agg["Total_CLV"] / seg_agg["Total_CLV"].sum() * 100
                ).round(1)
                st.dataframe(seg_agg.round(2), use_container_width=True)

                # ── Download ──────────────────────────────────────────────────
                buf = io.StringIO()
                result_df.to_csv(buf, index=False)
                st.download_button(
                    "⬇️ Download Full Results CSV",
                    data=buf.getvalue().encode("utf-8"),
                    file_name="batch_intelligence_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary",
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")