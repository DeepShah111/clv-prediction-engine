"""
Customer Intelligence Platform v3.1.0
======================================
Visual refresh — cleaner, more professional appearance.
All functions and logic identical to v3.0.0.
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
# CSS — clean, minimal, professional
# =============================================================================
st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Global ── */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* ── Cards ── */
    .kpi-card {
        background: #16213e;
        border-radius: 10px;
        padding: 18px 20px;
        border-left: 3px solid #5C4DB1;
        margin-bottom: 10px;
    }
    .kpi-card .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8892a4;
        margin-bottom: 4px;
    }
    .kpi-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    .kpi-card .sub {
        font-size: 0.78rem;
        color: #8892a4;
        margin-top: 3px;
    }

    /* ── Result card ── */
    .result-card {
        background: #0f3460;
        border-radius: 10px;
        padding: 20px;
        height: 100%;
    }
    .result-card .card-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8892a4;
        margin-bottom: 8px;
    }
    .result-card .card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
    }
    .result-card .card-sub {
        font-size: 0.8rem;
        color: #8892a4;
        margin-top: 4px;
    }

    /* ── Flag pill ── */
    .flag-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 3px;
    }

    /* ── Rec card ── */
    .rec-item {
        background: #16213e;
        border-left: 3px solid #2A9D8F;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
    }
    .rec-item .rec-name { color: white; font-weight: 600; font-size: 0.9rem; }
    .rec-item .rec-stats { color: #8892a4; font-size: 0.78rem; margin-top: 2px; }

    /* ── Divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid #1e2d4a;
        margin: 20px 0;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid #1e2d4a;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        font-size: 0.85rem;
        font-weight: 500;
        border-radius: 6px 6px 0 0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background: #0a1628; }
    [data-testid="stSidebar"] .stMarkdown { color: #c8d0dc; }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #16213e;
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Cached Loaders  (identical to v3.0.0)
# =============================================================================

@st.cache_resource(show_spinner="Loading models…")
def load_clv_bundle():
    p = MODELS_DIR / "clv_champion_bundle.pkl"
    if not p.exists():
        st.error(f"❌ CLV bundle not found: `{p}`")
        st.stop()
    return joblib.load(p)

@st.cache_resource(show_spinner=False)
def load_seg_bundle():
    p = MODELS_DIR / "segmentation_bundle.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner=False)
def load_richer_seg_bundle():
    p = MODELS_DIR / "richer_seg_bundle.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner=False)
def load_anomaly_bundle():
    p = MODELS_DIR / "anomaly_bundle.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner=False)
def load_ar_bundle():
    p = MODELS_DIR / "association_rules_bundle.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_data(show_spinner=False)
def load_csv(filename):
    p = GRAPHS_DIR / filename
    return pd.read_csv(p) if p.exists() else None

def _img(name):
    p = GRAPHS_DIR / name
    return p if p.exists() else None


# =============================================================================
# Helpers  (identical to v3.0.0)
# =============================================================================

def _build_row(overrides):
    return pd.DataFrame([{**FEATURE_DEFAULTS, **overrides}])[FEATURE_COLS]

def _predict_clv(model, row):
    log_pred = float(np.clip(model.predict(row), 0, LOG_PRED_MAX)[0])
    dollar   = float(np.expm1(log_pred))
    return {"log": log_pred, "dollar": dollar,
            "low": max(0.0, dollar * 0.85), "high": dollar * 1.15}

def _clv_tier(dollar):
    if dollar < 0.01:          return "💤 Churned",  "#E63946"
    elif dollar < SEGMENT_P20: return "📉 Low",       "#F4A261"
    elif dollar < SEGMENT_P80: return "💰 Mid",       "#2E86AB"
    else:                      return "🐋 Whale",     "#5C4DB1"

def _assign_cluster(row_df, bundle):
    if bundle is None:
        return None, "Unknown"
    try:
        cols   = bundle["cols_used"]
        scaler = bundle["scaler"]
        pca    = bundle.get("pca")
        kmeans = bundle["kmeans"]
        labels = bundle.get("segment_labels", {})
        avail  = [c for c in cols if c in row_df.columns]
        X      = row_df[avail].fillna(0).values.astype(np.float32)
        Xs     = scaler.transform(X)
        if pca is not None:
            Xs = pca.transform(Xs)
        cid  = int(kmeans.predict(Xs)[0])
        return cid, labels.get(cid, f"Cluster {cid}")
    except Exception:
        return None, "Unknown"

def _score_anomaly(bundle, feature_dict):
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
        risk     = "🔴 High" if score >= 0.65 else ("🟡 Medium" if score >= 0.40 else "🟢 Low")
        return {
            "score":          round(score, 4),
            "risk":           risk,
            "is_anomaly":     score >= thresh.get("anomaly_score", 0.50),
            "is_high_return": (feature_dict.get("Return_Rate", 0) > thresh.get("return_rate", 0.30)
                               and score > 0.40),
        }
    except Exception:
        return {"score": 0.0, "risk": "🟢 Low", "is_anomaly": False}

def _get_recommendations(ar_bundle, product_query, top_n=5):
    if ar_bundle is None or not product_query:
        return []
    lookup  = ar_bundle.get("lookup", {})
    q       = product_query.upper()
    if q in lookup:
        return lookup[q][:top_n]
    matches = [k for k in lookup if q in k]
    return lookup[matches[0]][:top_n] if matches else []

def _batch_predict(model, df):
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = FEATURE_DEFAULTS.get(col, 0.0)
    X     = df[FEATURE_COLS].fillna({c: FEATURE_DEFAULTS.get(c, 0) for c in FEATURE_COLS})
    preds = np.expm1(np.clip(model.predict(X), 0, LOG_PRED_MAX))
    out   = df.copy()
    out["CLV_Predicted_90d"] = np.round(preds, 2)
    out["CLV_Low"]           = np.round(np.maximum(0, preds * 0.85), 2)
    out["CLV_High"]          = np.round(preds * 1.15, 2)
    out["Segment"]           = [_clv_tier(d)[0] for d in preds]
    return out, preds

def _shap_fig(model, row):
    if not SHAP_AVAILABLE:
        return None
    est = getattr(model, "regressor_", model)
    if not hasattr(est, "feature_importances_"):
        return None
    try:
        exp = shap.TreeExplainer(est)(row)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.sca(ax)
        shap.plots.waterfall(exp[0], max_display=12, show=False)
        plt.tight_layout()
        return fig
    except Exception:
        return None

def _kpi(label, value, sub="", accent="#5C4DB1"):
    st.markdown(
        f'<div class="kpi-card" style="border-left-color:{accent}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'{"<div class=sub>" + sub + "</div>" if sub else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )

def _result_card(title, value, sub="", accent="#5C4DB1"):
    st.markdown(
        f'<div class="result-card">'
        f'<div class="card-title">{title}</div>'
        f'<div class="card-value" style="color:{accent}">{value}</div>'
        f'{"<div class=card-sub>" + sub + "</div>" if sub else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )

# =============================================================================
# Load bundles
# =============================================================================
clv_bundle     = load_clv_bundle()
seg_bundle     = load_seg_bundle()
rs_bundle      = load_richer_seg_bundle()
anomaly_bundle = load_anomaly_bundle()
ar_bundle      = load_ar_bundle()
model          = clv_bundle["model"]

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("## Customer Intelligence")
    st.markdown("##### Platform v3.1.0")
    st.markdown("<hr style='border-color:#1e2d4a;margin:12px 0'>", unsafe_allow_html=True)

    st.markdown("**Models**")
    bundles = [
        ("CLV Predictor",         clv_bundle),
        ("K-Means Segmentation",  seg_bundle),
        ("Richer Seg (k=2..5)",   rs_bundle),
        ("Anomaly Detection",     anomaly_bundle),
        ("Association Rules",     ar_bundle),
    ]
    for name, b in bundles:
        icon = "✅" if b else "⏳"
        st.markdown(f"<small>{icon} {name}</small>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2d4a;margin:12px 0'>", unsafe_allow_html=True)
    st.markdown("**CLV Tiers**")
    st.markdown(
        "<small>"
        "🐋 Whale ≥ $1,200<br>"
        "💰 Mid $150 – $1,200<br>"
        "📉 Low $0.01 – $150<br>"
        "💤 Churned $0"
        "</small>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color:#1e2d4a;margin:12px 0'>", unsafe_allow_html=True)
    v = clv_bundle.get("version", "?")
    t = clv_bundle.get("timestamp", "unknown")
    st.markdown(
        f"<small style='color:#556070'>"
        f"Model v{v} · {t[:10] if t != 'unknown' else '?'}<br>"
        f"Dollar R² = 0.581"
        f"</small>",
        unsafe_allow_html=True,
    )

# =============================================================================
# Page header
# =============================================================================
st.markdown("## 🛍️ Customer Intelligence Platform")
st.markdown(
    "<p style='color:#8892a4;margin-top:-12px;margin-bottom:20px'>"
    "Unified CLV prediction · Behavioural segmentation · Anomaly detection · Product recommendations"
    "</p>",
    unsafe_allow_html=True,
)

# =============================================================================
# Tabs
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Intelligence",
    "📊 Segments",
    "🚨 Anomaly",
    "🛒 Products",
    "📂 Batch",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — UNIFIED CUSTOMER INTELLIGENCE
# ═════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown("#### Customer Profile")
    st.caption("Adjust the key inputs. All 5 models run on a single click.")

    # ── Primary inputs (always visible) ──────────────────────────────────────
    p1, p2, p3 = st.columns(3)
    with p1:
        recency   = st.slider("Recency (days since first purchase)", 1, 730, 180, 10)
        frequency = st.slider("Frequency (invoices)", 1, 100, 8, 1)
    with p2:
        monetary    = st.slider("Monetary — avg order value ($)", 10.0, 5000.0, 450.0, 10.0)
        days_since  = st.slider("Days since last purchase", 1, 365, 45, 1)
    with p3:
        max_order   = st.slider("Max single order ($)", 10.0, 5000.0, 380.0, 10.0)
        return_rate = st.slider("Return rate", 0.0, 1.0, 0.04, 0.01)

    # ── Advanced inputs (collapsed) ───────────────────────────────────────────
    with st.expander("⚙️ Advanced inputs", expanded=False):
        a1, a2, a3 = st.columns(3)
        with a1:
            avg_basket   = st.slider("Avg basket size", 1.0, 50.0, 4.0, 0.5)
            unique_prods = st.slider("Unique products", 1, 200, 15, 1)
            purchase_rate = st.slider("Purchase rate", 0.0, 0.5, 0.018, 0.001)
        with a2:
            revenue_day  = st.slider("Revenue / day ($)", 0.0, 50.0, 2.5, 0.1)
            mon_pct      = st.slider("Monetary percentile", 0.0, 1.0, 0.65, 0.01)
            ipu_std      = st.slider("Interpurchase std", 0.0, 200.0, 30.0, 1.0)
        with a3:
            prob_alive   = st.slider("Prob alive", 0.0, 1.0, 0.72, 0.01)
            prob_txn     = st.slider("Prob pred txn", 0.0, 20.0, 2.1, 0.1)
            prob_val     = st.slider("Prob pred val ($)", 0.0, 5000.0, 420.0, 10.0)
            visit_div    = st.slider("Visit diversity", 1.0, 50.0, 6.0, 0.5)

    analyze_btn = st.button(
        "🚀 Run Intelligence Analysis",
        type="primary", use_container_width=True
    )

    if analyze_btn:
        overrides = {
            "Recency": float(recency), "Frequency": float(frequency),
            "Monetary": float(monetary), "Return_Rate": float(return_rate),
            "Days_Since_Purchase": float(days_since), "Max_Single_Order": float(max_order),
            "Avg_Basket_Size": float(avg_basket), "Unique_Products": float(unique_prods),
            "Purchase_Rate": float(purchase_rate), "Revenue_Per_Day": float(revenue_day),
            "Monetary_Percentile": float(mon_pct), "Interpurchase_Std": float(ipu_std),
            "Prob_Alive": float(prob_alive), "Prob_Pred_Txn": float(prob_txn),
            "Prob_Pred_Val": float(prob_val), "Visit_Diversity": float(visit_div),
        }
        row = _build_row(overrides)

        # ── Run all models ────────────────────────────────────────────────────
        clv_result            = _predict_clv(model, row)
        tier, tcolor          = _clv_tier(clv_result["dollar"])
        cluster_id, seg_name  = _assign_cluster(row, seg_bundle)
        anomaly_result        = _score_anomaly(anomaly_bundle, overrides)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("#### Intelligence Report")

        # ── 4 result cards ────────────────────────────────────────────────────
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            _result_card(
                "Predicted 90-Day CLV",
                f"${clv_result['dollar']:,.2f}",
                f"Range: ${clv_result['low']:,.0f} – ${clv_result['high']:,.0f}",
                accent=tcolor,
            )
        with rc2:
            _result_card(
                "CLV Tier",
                tier,
                f"Threshold: ${SEGMENT_P20:,.0f} / ${SEGMENT_P80:,.0f}",
                accent=tcolor,
            )
        with rc3:
            seg_color = SEG_COLOURS.get(seg_name, "#5C4DB1")
            _result_card(
                "Behavioural Segment",
                seg_name if seg_name != "Unknown" else "—",
                f"K-Means · k={seg_bundle.get('optimal_k','?') if seg_bundle else '?'}",
                accent=seg_color,
            )
        with rc4:
            risk_color = RISK_COLOURS.get(anomaly_result["risk"], "#2A9D8F")
            _result_card(
                "Anomaly Risk",
                anomaly_result["risk"],
                f"Score: {anomaly_result['score']:.3f}  [0 = normal, 1 = extreme]",
                accent=risk_color,
            )

        # ── Business flags ────────────────────────────────────────────────────
        st.markdown("<div style='margin-top:10px'>", unsafe_allow_html=True)
        flags = {
            "🐋 Whale":          (clv_result["dollar"] >= SEGMENT_P80,  "#5C4DB1"),
            "⚠️ Anomalous":      (anomaly_result.get("is_anomaly", False), "#E63946"),
            "🔁 High Return":    (anomaly_result.get("is_high_return", False), "#F4A261"),
            "💤 Churn Risk":     (clv_result["dollar"] < 10,             "#E63946"),
            "📈 Growth Signal":  (frequency >= 5 and monetary >= 200,    "#2A9D8F"),
        }
        pills = "".join([
            f'<span class="flag-pill" style="background:{"" + c + "cc" if v else "#1e2d4a"};'
            f'color:{"white" if v else "#556070"}">{k}</span>'
            for k, (v, c) in flags.items()
        ])
        st.markdown(
            f'<div style="margin:8px 0 16px 0"><span style="color:#8892a4;'
            f'font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em">'
            f'Business Flags</span><br><br>{pills}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── SHAP + Feature profile ────────────────────────────────────────────
        shap_col, feat_col = st.columns([3, 2])
        with shap_col:
            st.markdown("##### Why this CLV? — SHAP")
            if SHAP_AVAILABLE:
                fig = _shap_fig(model, row)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.info("SHAP waterfall unavailable for this model type.")
            else:
                st.warning("Install `shap` to enable feature explanations.")

        with feat_col:
            st.markdown("##### Feature Profile")
            feat_df = row.T.rename(columns={0: "Value"}).round(4)
            feat_df["Type"] = feat_df.index.map(lambda x:
                "RFM" if x in ["Recency", "Frequency", "Monetary"]
                else ("Behavioural" if x in ["Return_Rate", "Avg_Basket_Size", "Unique_Products"]
                else "Derived"))
            st.dataframe(feat_df, use_container_width=True, height=400)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEGMENTATION LAB
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Segmentation Lab")
    st.caption("Statistical cluster analysis — how k=2 was selected and what each segment means.")

    # ── k-selection table ─────────────────────────────────────────────────────
    rs_metrics_df = load_csv("richer_seg_metrics.csv")
    if rs_metrics_df is not None:
        st.markdown("**K-Selection Results**")

        def _hl(row):
            is_best = row["composite_score"] == rs_metrics_df["composite_score"].max()
            return ["background-color:#2A9D8F18;font-weight:700" if is_best else "" for _ in row]

        cols_show = ["k", "silhouette", "davies_bouldin", "calinski_harabasz", "composite_score"]
        styled = (rs_metrics_df[cols_show].style
                  .apply(_hl, axis=1)
                  .format({"silhouette": "{:.4f}", "davies_bouldin": "{:.4f}",
                           "calinski_harabasz": "{:.0f}", "composite_score": "{:.4f}"}))
        st.dataframe(styled, use_container_width=True)
        st.info("k=2 has the highest composite score. Silhouette drops from 0.53 → 0.35 as k increases — splitting further adds noise, not insight.")

    k_plot = _img("rs_k_comparison.png")
    if k_plot:
        st.image(str(k_plot), caption="Metrics across k=2..5 — dashed line marks optimal k",
                 use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("**Cluster Visualisations**")

    c1, c2 = st.columns(2)
    with c1:
        p = _img("seg_umap.png") or _img("rs_umap_5seg.png")
        if p: st.image(str(p), caption="UMAP — Cluster Map", use_container_width=True)
    with c2:
        p = _img("seg_clv_heatmap.png")
        if p: st.image(str(p), caption="Segment × CLV Tier Heatmap", use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        p = _img("rs_segment_profiles.png") or _img("seg_cluster_profiles.png")
        if p: st.image(str(p), caption="Feature Profiles per Segment", use_container_width=True)
    with c4:
        p = _img("rs_clv_by_segment.png")
        if p: st.image(str(p), caption="CLV Distribution per Segment", use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("**Segment Business Summary**")
    cust_df = load_csv("customer_segments.csv")
    if cust_df is not None and "Segment_Name" in cust_df.columns:
        clv_col = next((c for c in ["CLV_Predicted_90d", "Predicted_CLV", "Monetary"]
                        if c in cust_df.columns), None)
        if clv_col:
            summary = (cust_df.groupby("Segment_Name")[clv_col]
                       .agg(N="count", Avg_CLV="mean", Total_CLV="sum")
                       .reset_index())
            summary["Revenue_%"] = (summary["Total_CLV"] / summary["Total_CLV"].sum() * 100).round(1)
            st.dataframe(summary.round(2), use_container_width=True)

    with st.expander("More visualisations", expanded=False):
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


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANOMALY DETECTION
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Anomaly Detection")

    if anomaly_bundle is None:
        st.warning("⏳ Run `run_anomaly_pipeline()` in segmentation.ipynb first.")
    else:
        m = anomaly_bundle.get("metrics", {})
        k1, k2, k3, k4 = st.columns(4)
        with k1: _kpi("Total Customers",   f"{m.get('n_total',0):,}")
        with k2: _kpi("Anomalies Flagged", f"{m.get('n_anomaly',0):,}",
                       f"{m.get('pct_anomaly',0):.1f}% of fleet", "#E63946")
        with k3: _kpi("High-Return Flags", f"{m.get('n_high_return',0):,}")
        with k4: _kpi("Whale Anomalies",   f"{m.get('n_whale_anomaly',0):,}")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("**Score Distributions**")
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
            if p: st.image(str(p), caption="UMAP — Anomaly Map", use_container_width=True)
        with r4:
            p = _img("anomaly_return_flags.png")
            if p: st.image(str(p), caption="Return Rate vs Anomaly Score", use_container_width=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("**Top Flagged Customers**")
        adf = load_csv("anomaly_scores.csv")
        if adf is not None:
            show_cols = [c for c in ["Anomaly_Score", "IF_Score", "AE_Score",
                                     "is_anomaly", "is_high_return", "is_suspicious",
                                     "Recency", "Frequency", "Monetary", "Return_Rate"]
                         if c in adf.columns]
            top50 = adf.nlargest(50, "Anomaly_Score")[show_cols]
            def _cs(v):
                if isinstance(v, (int, float)):
                    if v >= 0.65: return "background-color:#E6394633"
                    if v >= 0.50: return "background-color:#F4A26133"
                return ""
            st.dataframe(top50.style.map(_cs, subset=["Anomaly_Score"]),
                         use_container_width=True, height=360)
            buf = io.StringIO()
            top50.to_csv(buf, index=False)
            st.download_button("⬇️ Download CSV", buf.getvalue().encode(),
                               "anomaly_top50.csv", "text/csv")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("**Score a Single Customer**")
        cols_u = anomaly_bundle.get("cols_used", list(FEATURE_DEFAULTS.keys())[:12])
        sc1, sc2, sc3 = st.columns(3)
        fi = {}
        sdefs = {
            "Recency": (1,730,365,1), "Frequency": (1,200,4,1),
            "Monetary": (0,10000,300,10), "Return_Rate": (0.0,1.0,0.04,0.01),
            "Avg_Basket_Size": (1.0,50.0,3.2,0.1), "Max_Single_Order": (0,10000,250,10),
            "Monetary_Percentile": (0.0,1.0,0.50,0.01), "Purchase_Rate": (0.0,0.5,0.012,0.001),
            "Unique_Products": (1,200,12,1), "Days_Since_Purchase": (1,730,60,1),
            "Interpurchase_Std": (0.0,200.0,45.0,1.0), "Revenue_Per_Day": (0.0,50.0,1.8,0.1),
        }
        for i, col in enumerate(cols_u):
            if col in sdefs:
                mn, mx, dv, sv = sdefs[col]
                with [sc1, sc2, sc3][i % 3]:
                    fi[col] = st.number_input(col, float(mn), float(mx),
                                              float(dv), float(sv), key=f"a_{col}")

        if st.button("Calculate Anomaly Score", type="primary",
                     use_container_width=True, key="ascore_btn"):
            res = _score_anomaly(anomaly_bundle, fi)
            rc  = RISK_COLOURS.get(res["risk"], "#2A9D8F")
            x1, x2, x3 = st.columns(3)
            with x1:
                st.markdown(
                    f'<div class="result-card" style="text-align:center">'
                    f'<div class="card-title">Risk Level</div>'
                    f'<div class="card-value" style="color:{rc}">{res["risk"]}</div>'
                    f'</div>', unsafe_allow_html=True)
            x2.metric("Anomaly Score", f"{res['score']:.4f}")
            x3.metric("Status", "⚠️ Flagged" if res["is_anomaly"] else "✅ Normal")

            if res["is_anomaly"]:
                st.error("This customer is flagged — review transaction history.")
            else:
                st.success("Customer behaviour is within normal range.")

        with st.expander("Model details", expanded=False):
            p = _img("anomaly_reconstruction_error.png")
            if p: st.image(str(p), caption="Autoencoder Training Loss", use_container_width=True)
            p = _img("anomaly_top_customers.png")
            if p: st.image(str(p), caption="Top 20 Anomaly Profiles", use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRODUCT INTELLIGENCE
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### Product Intelligence — Association Rules")
    st.caption("FP-Growth market basket analysis on 400K transactions.")

    if ar_bundle is None:
        st.warning("⏳ Run `run_association_rules_pipeline()` first.")
    else:
        arm = ar_bundle.get("metrics", {})
        k1, k2, k3, k4 = st.columns(4)
        with k1: _kpi("Total Rules",        f"{arm.get('n_rules_total',0):,}")
        with k2: _kpi("Products with Recs", f"{arm.get('n_products_with_recs',0):,}")
        with k3: _kpi("Avg Lift",           f"{arm.get('avg_lift',0):.2f}×", accent="#2A9D8F")
        with k4: _kpi("Max Lift",           f"{arm.get('max_lift',0):.2f}×", accent="#2A9D8F")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("**Cross-Sell Recommendation Engine**")
        st.caption("Enter a product name (partial match supported).")

        lookup = ar_bundle.get("lookup", {})
        samples = list(lookup.keys())[:5]

        col_in, col_n = st.columns([3, 1])
        with col_in:
            product_input = st.text_input(
                "Product", placeholder="e.g. HEART, LANTERN, CANDLE, ALARM CLOCK",
                key="ar_input", label_visibility="collapsed"
            )
        with col_n:
            top_n = st.selectbox("Top N", [3, 5, 10], index=1, label_visibility="collapsed")

        if samples:
            st.caption(f"Try: {' · '.join(samples[:5])}")

        if product_input:
            recs = _get_recommendations(ar_bundle, product_input, top_n)
            if recs:
                st.markdown(f"**Recommendations for '{product_input.upper()}'**")
                for i, r in enumerate(recs):
                    lift_val = r.get("lift", 0)
                    lcolor   = "#2A9D8F" if lift_val > 5 else ("#F4A261" if lift_val > 2 else "#8892a4")
                    st.markdown(
                        f'<div class="rec-item">'
                        f'<div class="rec-name">#{i+1} &nbsp; {r["product"][:55]}</div>'
                        f'<div class="rec-stats">'
                        f'Confidence {r.get("confidence",0)*100:.1f}% &nbsp;·&nbsp; '
                        f'<span style="color:{lcolor}">Lift {lift_val:.2f}×</span> &nbsp;·&nbsp; '
                        f'Support {r.get("support",0)*100:.1f}%'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.warning(f"No results for '{product_input}'. Try: HEART, WHITE, RED, CANDLE.")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("**Rule Analysis**")
        p1, p2 = st.columns(2)
        with p1:
            p = _img("ar_support_confidence.png")
            if p: st.image(str(p), caption="Support vs Confidence", use_container_width=True)
            else: st.info("Run association rules pipeline to generate plots.")
        with p2:
            p = _img("ar_lift_heatmap.png")
            if p: st.image(str(p), caption="Lift Matrix — Top 15 Products", use_container_width=True)

        p3, p4 = st.columns(2)
        with p3:
            p = _img("ar_top_rules_bar.png")
            if p: st.image(str(p), caption="Top 20 Rules by Lift", use_container_width=True)
        with p4:
            p = _img("ar_segment_comparison.png")
            if p: st.image(str(p), caption="Champions vs Loyal Customers", use_container_width=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("**Full Rules Table**")
        rules_df = load_csv("association_rules.csv")
        if rules_df is not None:
            st.dataframe(rules_df.sort_values("lift", ascending=False).head(100).round(4),
                         use_container_width=True, height=320)
            buf = io.StringIO()
            rules_df.to_csv(buf, index=False)
            st.download_button("⬇️ Download Rules CSV",
                               buf.getvalue().encode(), "association_rules.csv", "text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — BATCH OPERATIONS
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("#### Batch Operations")
    st.caption("Upload a customer CSV — CLV prediction, segmentation, and anomaly scoring in one run.")

    uploaded = st.file_uploader(
        "Upload CSV (any subset of 16 features)", type=["csv"],
        label_visibility="collapsed"
    )

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, encoding="ISO-8859-1")
            st.success(f"Loaded {len(df):,} customers from `{uploaded.name}`")

            opt1, opt2 = st.columns(2)
            with opt1:
                run_seg  = st.checkbox("Include segment assignment", value=True)
            with opt2:
                run_anom = st.checkbox("Include anomaly scoring", value=False)

            if st.button("▶️ Run Batch Analysis", type="primary", use_container_width=True):
                with st.spinner("Running…"):
                    result_df, dollar_preds = _batch_predict(model, df)

                    if run_seg and seg_bundle is not None:
                        result_df["K_Segment"] = [
                            _assign_cluster(
                                pd.DataFrame([r.to_dict()])[FEATURE_COLS], seg_bundle
                            )[1]
                            for _, r in result_df[FEATURE_COLS].iterrows()
                        ]

                    if run_anom and anomaly_bundle is not None:
                        scores = [_score_anomaly(anomaly_bundle, r.to_dict())["score"]
                                  for _, r in result_df.iterrows()]
                        result_df["Anomaly_Score"] = np.round(scores, 4)
                        result_df["Risk"] = [
                            "🔴 High" if s >= 0.65 else ("🟡 Med" if s >= 0.40 else "🟢 Low")
                            for s in scores
                        ]

                st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
                st.markdown("**Summary**")
                b1, b2, b3, b4 = st.columns(4)
                with b1: _kpi("Customers",     f"{len(result_df):,}")
                with b2: _kpi("Avg CLV",       f"${dollar_preds.mean():,.0f}", accent="#2A9D8F")
                with b3: _kpi("Total Revenue", f"${dollar_preds.sum():,.0f}", accent="#2A9D8F")
                with b4: _kpi("Whales",
                               f"{(dollar_preds >= SEGMENT_P80).sum():,}",
                               f"{(dollar_preds >= SEGMENT_P80).mean()*100:.1f}%",
                               accent="#5C4DB1")

                st.markdown("**Results Preview**")
                preview = ["CLV_Predicted_90d", "CLV_Low", "CLV_High", "Segment"]
                if "K_Segment"     in result_df.columns: preview.append("K_Segment")
                if "Anomaly_Score" in result_df.columns: preview += ["Anomaly_Score", "Risk"]
                st.dataframe(result_df[preview].head(50), use_container_width=True)

                st.markdown("**Segment Breakdown**")
                seg_col = "K_Segment" if "K_Segment" in result_df.columns else "Segment"
                seg_agg = (result_df.groupby(seg_col)["CLV_Predicted_90d"]
                           .agg(N="count", Avg_CLV="mean", Total_CLV="sum")
                           .reset_index())
                seg_agg["Revenue_%"] = (seg_agg["Total_CLV"] / seg_agg["Total_CLV"].sum() * 100).round(1)
                st.dataframe(seg_agg.round(2), use_container_width=True)

                buf = io.StringIO()
                result_df.to_csv(buf, index=False)
                st.download_button(
                    "⬇️ Download Results",
                    data=buf.getvalue().encode(),
                    file_name="batch_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary",
                )

        except Exception as e:
            st.error(f"Error: {e}")