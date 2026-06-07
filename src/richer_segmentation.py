"""
Customer Intelligence Platform — Richer Segmentation Module v1.0.0
===================================================================
Upgrades the binary k=2 segmentation to a granular 5-segment model
with business-meaningful names.

What this module does
---------------------
1. Tests k = 2, 3, 4, 5 systematically with full evaluation metrics
2. Selects optimal k using a composite score (silhouette + CH + DB)
3. Names each cluster with a business label based on RFM + CLV profile:
   - 🏆 Champions          : high frequency, high monetary, low recency gap
   - 🐋 Dormant Whales      : high past spend, high recency gap (churning whales)
   - 📈 Growing Mid-Tier    : medium frequency, increasing purchase rate
   - 🔁 High-Freq Low-Value : frequent but small baskets
   - 💤 Lost Customers      : very low spend, high recency gap

4. Saves all k models for comparison
5. Generates 5 diagnostic plots
6. Updates customer_segments.csv with new granular labels

Plots Saved
-----------
Plot RS1: rs_k_comparison.png      - metrics across k=2..5
Plot RS2: rs_umap_5seg.png         - UMAP coloured by 5 segments
Plot RS3: rs_segment_profiles.png  - radar chart per segment
Plot RS4: rs_clv_by_segment.png    - CLV distribution per segment
Plot RS5: rs_revenue_treemap.png   - revenue share treemap

Usage
-----
    from src.richer_segmentation import run_richer_segmentation

    results = run_richer_segmentation(
        X_train      = X_train,
        X_test       = X_test,
        y_test_raw   = y_test_raw,
        dollar_preds = dollar_preds,
    )
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

from src.config import GRAPHS_DIR, MODELS_DIR, RANDOM_SEED, FEATURE_COLS

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional UMAP
# ---------------------------------------------------------------------------
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ===========================================================================
# Constants
# ===========================================================================

K_RANGE = [2, 3, 4, 5]

# Features used for segmentation (behavioural subset — same as segmentation.py)
SEG_FEATURE_COLS = [
    "Recency", "Frequency", "Monetary",
    "Purchase_Rate", "Days_Since_Purchase",
    "Unique_Products", "Avg_Basket_Size",
    "Return_Rate", "Max_Single_Order", "Monetary_Percentile",
]

# Business segment name assignment rules
# Each rule is evaluated in order; first match wins
# Keys map to (feature, comparator, threshold) tuples
SEGMENT_NAMING_RULES = {
    5: [
        # k=5 segment names
        ("🏆 Champions",           "high_monetary_low_recency"),
        ("🐋 Dormant Whales",       "high_monetary_high_recency"),
        ("📈 Growing Mid-Tier",     "mid_monetary_high_frequency"),
        ("🔁 High-Freq Low-Value",  "high_frequency_low_monetary"),
        ("💤 Lost Customers",       "low_monetary_high_recency"),
    ],
    4: [
        ("🏆 Champions",           "high_monetary_low_recency"),
        ("🐋 Dormant Whales",       "high_monetary_high_recency"),
        ("📈 Growing Mid-Tier",     "mid_monetary_high_frequency"),
        ("💤 Lost Customers",       "low_monetary_high_recency"),
    ],
    3: [
        ("🏆 Champions",           "high_monetary_low_recency"),
        ("💰 Mid-Value Customers",  "mid_monetary"),
        ("💤 Lost Customers",       "low_monetary_high_recency"),
    ],
    2: [
        ("🐋 Loyal Customers",      "high_monetary_low_recency"),
        ("👥 Champions",            "lower_value"),
    ],
}

# Plot style
STYLE   = "dark_background"
SEGMENT_PALETTE = {
    "🏆 Champions":          "#FFD700",
    "🐋 Dormant Whales":      "#5C4DB1",
    "📈 Growing Mid-Tier":    "#2A9D8F",
    "🔁 High-Freq Low-Value": "#F4A261",
    "💤 Lost Customers":      "#E63946",
    "🐋 Loyal Customers":     "#5C4DB1",
    "💰 Mid-Value Customers": "#2E86AB",
    "👥 Champions":           "#2E86AB",
}


# ===========================================================================
# Feature Preparation
# ===========================================================================

def _prepare_features(
    X: pd.DataFrame,
    scaler: Optional[RobustScaler] = None,
    pca: Optional[PCA] = None,
    n_pca_components: int = 6,
    fit: bool = True,
) -> Tuple[np.ndarray, RobustScaler, PCA]:
    """Scale + PCA-reduce features. Mirrors segmentation.py approach."""
    cols = [c for c in SEG_FEATURE_COLS if c in X.columns]
    X_sub = X[cols].fillna(X[cols].median())

    if scaler is None:
        scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_sub) if fit else scaler.transform(X_sub)

    if pca is None:
        n_comp = min(n_pca_components, X_scaled.shape[1], X_scaled.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled) if fit else pca.transform(X_scaled)

    if fit:
        var = pca.explained_variance_ratio_.sum()
        logger.info(f"PCA: {pca.n_components_} components explain {var*100:.1f}% variance")

    return X_pca.astype(np.float32), scaler, pca


# ===========================================================================
# K Evaluation
# ===========================================================================

def evaluate_k_range(
    X_pca: np.ndarray,
    k_range: List[int] = K_RANGE,
) -> pd.DataFrame:
    """
    Fits KMeans for each k in k_range and computes all 3 metrics.

    Returns
    -------
    metrics_df : DataFrame with columns [k, silhouette, davies_bouldin,
                 calinski_harabasz, composite_score]
    """
    records = []
    logger.info(f"Evaluating k = {k_range}...")

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=15)
        labels = km.fit_predict(X_pca)

        sil = silhouette_score(X_pca, labels)
        db  = davies_bouldin_score(X_pca, labels)
        ch  = calinski_harabasz_score(X_pca, labels)

        records.append({
            "k":                  k,
            "silhouette":         round(sil, 4),
            "davies_bouldin":     round(db, 4),
            "calinski_harabasz":  round(ch, 1),
            "inertia":            round(km.inertia_, 1),
        })
        logger.info(
            f"  k={k} | Silhouette={sil:.4f} | DB={db:.4f} | CH={ch:.0f}"
        )

    df = pd.DataFrame(records)

    # Composite score: normalise each metric to [0,1] and combine
    # silhouette: higher = better → normalise ascending
    # davies_bouldin: lower = better → normalise descending
    # calinski_harabasz: higher = better → normalise ascending
    df["sil_norm"] = (df["silhouette"] - df["silhouette"].min()) / \
                     (df["silhouette"].max() - df["silhouette"].min() + 1e-9)
    df["db_norm"]  = 1 - (df["davies_bouldin"] - df["davies_bouldin"].min()) / \
                     (df["davies_bouldin"].max() - df["davies_bouldin"].min() + 1e-9)
    df["ch_norm"]  = (df["calinski_harabasz"] - df["calinski_harabasz"].min()) / \
                     (df["calinski_harabasz"].max() - df["calinski_harabasz"].min() + 1e-9)

    # Weighted composite: silhouette gets highest weight (most interpretable)
    df["composite_score"] = (
        0.50 * df["sil_norm"] +
        0.25 * df["db_norm"]  +
        0.25 * df["ch_norm"]
    ).round(4)

    return df.drop(columns=["sil_norm", "db_norm", "ch_norm"])


def select_optimal_k(metrics_df: pd.DataFrame) -> int:
    """Select k with highest composite score."""
    best = metrics_df.loc[metrics_df["composite_score"].idxmax()]
    k    = int(best["k"])
    logger.info(
        f"Optimal k selected: {k} "
        f"(composite={best['composite_score']:.4f}, "
        f"silhouette={best['silhouette']:.4f})"
    )
    return k


# ===========================================================================
# Segment Naming
# ===========================================================================

def _compute_cluster_profiles(
    X: pd.DataFrame,
    labels: np.ndarray,
    dollar_preds: Optional[np.ndarray] = None,
    y_test_raw: Optional[pd.Series] = None,
    n_train: int = 0,
) -> pd.DataFrame:
    """Compute mean feature values per cluster for naming logic."""
    df = X.copy().reset_index(drop=True)
    df["_cluster"] = labels

    if dollar_preds is not None:
        clv = np.full(len(df), np.nan)
        clv[n_train:] = dollar_preds
        df["_clv"] = clv

    profile = df.groupby("_cluster").mean(numeric_only=True)
    return profile


def assign_segment_names(
    profiles: pd.DataFrame,
    k: int,
) -> Dict[int, str]:
    """
    Assigns business names to clusters based on their feature profiles.

    Logic:
    - Rank clusters by Monetary (spend) and Recency (days since first purchase)
    - Assign names based on relative position in these two dimensions
    - Higher Monetary + lower Days_Since_Purchase = Champions/Whales
    - Higher Monetary + higher Days_Since_Purchase = Dormant Whales
    - High Frequency + lower Monetary = High-Freq Low-Value
    - Low everything = Lost Customers

    Returns
    -------
    {cluster_id: segment_name}
    """
    monetary_col = "Monetary" if "Monetary" in profiles.columns else profiles.columns[0]
    recency_col  = "Days_Since_Purchase" if "Days_Since_Purchase" in profiles.columns else None
    freq_col     = "Frequency" if "Frequency" in profiles.columns else None

    # Rank clusters
    mon_rank  = profiles[monetary_col].rank(ascending=False)  # 1 = highest spend
    rec_rank  = profiles[recency_col].rank(ascending=True) if recency_col else None   # 1 = most recent
    freq_rank = profiles[freq_col].rank(ascending=False)   if freq_col   else None

    cluster_ids = sorted(profiles.index.tolist())
    names       = {}

    if k == 5:
        # Sort by composite (monetary desc, recency asc)
        if rec_rank is not None:
            composite = mon_rank + rec_rank
        else:
            composite = mon_rank

        sorted_clusters = composite.sort_values().index.tolist()

        labels_5 = [
            "🏆 Champions",
            "🐋 Dormant Whales",
            "📈 Growing Mid-Tier",
            "🔁 High-Freq Low-Value",
            "💤 Lost Customers",
        ]

        # Champion = highest monetary + most recent
        # Dormant Whale = high monetary + least recent
        # Growing Mid-Tier = mid monetary + high frequency
        # High-Freq Low-Value = high frequency + low monetary
        # Lost = lowest monetary + least recent

        # Sort by monetary desc
        by_monetary = profiles[monetary_col].sort_values(ascending=False).index.tolist()
        top2   = by_monetary[:2]
        bottom = by_monetary[-1]
        mid    = [c for c in by_monetary if c not in top2 and c != bottom]

        # Between top 2: the one with lower Days_Since_Purchase = Champion
        if recency_col and len(top2) == 2:
            rec_vals = profiles.loc[top2, recency_col]
            champion = rec_vals.idxmin()
            dormant  = rec_vals.idxmax()
        else:
            champion, dormant = top2[0], top2[1] if len(top2) > 1 else top2[0]

        names[champion] = "🏆 Champions"
        names[dormant]  = "🐋 Dormant Whales"
        names[bottom]   = "💤 Lost Customers"

        # Among mid clusters: freq-based split
        if len(mid) == 2 and freq_col:
            freq_vals = profiles.loc[mid, freq_col]
            names[freq_vals.idxmax()] = "🔁 High-Freq Low-Value"
            names[freq_vals.idxmin()] = "📈 Growing Mid-Tier"
        elif len(mid) == 1:
            names[mid[0]] = "📈 Growing Mid-Tier"

    elif k == 4:
        by_monetary = profiles[monetary_col].sort_values(ascending=False).index.tolist()
        top2   = by_monetary[:2]
        bottom = by_monetary[-1]
        mid    = [c for c in by_monetary if c not in top2 and c != bottom]

        if recency_col and len(top2) == 2:
            rec_vals = profiles.loc[top2, recency_col]
            names[rec_vals.idxmin()] = "🏆 Champions"
            names[rec_vals.idxmax()] = "🐋 Dormant Whales"
        else:
            names[top2[0]] = "🏆 Champions"
            if len(top2) > 1:
                names[top2[1]] = "🐋 Dormant Whales"

        if mid:
            names[mid[0]] = "📈 Growing Mid-Tier"
        names[bottom] = "💤 Lost Customers"

    elif k == 3:
        by_monetary = profiles[monetary_col].sort_values(ascending=False).index.tolist()
        names[by_monetary[0]] = "🏆 Champions"
        names[by_monetary[1]] = "💰 Mid-Value Customers"
        names[by_monetary[2]] = "💤 Lost Customers"

    else:  # k == 2
        by_monetary = profiles[monetary_col].sort_values(ascending=False).index.tolist()
        names[by_monetary[0]] = "🐋 Loyal Customers"
        names[by_monetary[1]] = "👥 Champions"

    # Fill any missing clusters
    for cid in cluster_ids:
        if cid not in names:
            names[cid] = f"Cluster {cid}"

    return names


# ===========================================================================
# Diagnostic Plots
# ===========================================================================

def _savefig(name: str) -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAPHS_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_k_comparison(metrics_df: pd.DataFrame, optimal_k: int) -> None:
    """Plot RS1: Metrics across k=2..5."""
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Richer Segmentation — K Selection Metrics",
                     color="white", fontsize=14, fontweight="bold")

        metrics_info = [
            ("silhouette",        "Silhouette Score",        "↑ higher = better", "#5C4DB1"),
            ("davies_bouldin",    "Davies-Bouldin Index",    "↓ lower = better",  "#E63946"),
            ("calinski_harabasz", "Calinski-Harabasz Score", "↑ higher = better", "#2A9D8F"),
        ]

        for ax, (col, title, hint, color) in zip(axes, metrics_info):
            ax.plot(metrics_df["k"], metrics_df[col], color=color,
                    marker="o", lw=2, ms=8)
            ax.axvline(optimal_k, color="#FFD700", ls="--", lw=1.5,
                       label=f"Optimal k={optimal_k}")
            ax.set_title(f"{title}\n{hint}", color="white", fontsize=11)
            ax.set_xlabel("k", color="white")
            ax.set_xticks(metrics_df["k"])
            ax.tick_params(colors="white")
            ax.legend(labelcolor="white", fontsize=8)
            for spine in ax.spines.values():
                spine.set_color("#333")

        plt.tight_layout()
    _savefig("rs_k_comparison.png")


def plot_umap_5seg(
    X_pca: np.ndarray,
    labels: np.ndarray,
    segment_names: Dict[int, str],
) -> None:
    """Plot RS2: UMAP coloured by granular segments."""
    if not UMAP_AVAILABLE:
        logger.info("UMAP not available — skipping rs_umap_5seg.png")
        return

    try:
        reducer   = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                              random_state=RANDOM_SEED, verbose=False)
        embedding = reducer.fit_transform(X_pca)
    except Exception as e:
        logger.warning(f"UMAP failed: {e}")
        return

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 8))

        for cluster_id, seg_name in segment_names.items():
            mask  = labels == cluster_id
            color = SEGMENT_PALETTE.get(seg_name, "#888888")
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=color, s=12, alpha=0.7, label=f"{seg_name} (n={mask.sum():,})",
                linewidths=0,
            )

        ax.set_title("UMAP — Granular Customer Segments",
                     color="white", fontsize=13, fontweight="bold")
        ax.set_xlabel("UMAP-1", color="white")
        ax.set_ylabel("UMAP-2", color="white")
        ax.tick_params(colors="white")
        ax.legend(labelcolor="white", fontsize=9, loc="upper right")
        for spine in ax.spines.values():
            spine.set_color("#333")

        plt.tight_layout()
    _savefig("rs_umap_5seg.png")


def plot_segment_profiles(
    profiles: pd.DataFrame,
    segment_names: Dict[int, str],
) -> None:
    """Plot RS3: Feature heatmap per segment."""
    display_cols = [c for c in SEG_FEATURE_COLS if c in profiles.columns]
    if not display_cols:
        return

    data = profiles[display_cols].copy()
    # Cast to numeric — object columns cause seaborn dtype error
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float64)
    data.index = [segment_names.get(i, f"Cluster {i}") for i in data.index]

    # Standardise for visual comparison
    data_std = (data - data.mean()) / (data.std() + 1e-8)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(14, max(4, len(data) * 1.4)))
        sns.heatmap(
            data_std, annot=True, fmt=".2f", cmap="RdYlGn",
            linewidths=0.3, ax=ax,
            cbar_kws={"label": "Standardised Value"},
        )
        ax.set_title("Segment Feature Profiles (standardised z-scores)",
                     color="white", fontsize=13, fontweight="bold", pad=12)
        ax.tick_params(colors="white", axis="x", rotation=40, labelsize=9)
        ax.tick_params(colors="white", axis="y", labelsize=10)
        plt.tight_layout()
    _savefig("rs_segment_profiles.png")


def plot_clv_by_segment(
    labels: np.ndarray,
    segment_names: Dict[int, str],
    dollar_preds: np.ndarray,
    n_train: int,
) -> None:
    """Plot RS4: CLV distribution per segment (test set only)."""
    test_labels = labels[n_train:]
    if len(test_labels) != len(dollar_preds):
        logger.warning("Label/pred length mismatch — skipping CLV plot")
        return

    df = pd.DataFrame({
        "Segment": [segment_names.get(l, f"Cluster {l}") for l in test_labels],
        "CLV":     dollar_preds,
    })

    order   = df.groupby("Segment")["CLV"].median().sort_values(ascending=False).index
    palette = {seg: SEGMENT_PALETTE.get(seg, "#888") for seg in order}

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 6))
        for seg in order:
            vals = df.loc[df["Segment"] == seg, "CLV"]
            ax.boxplot(
                vals, positions=[list(order).index(seg)],
                widths=0.6, patch_artist=True,
                boxprops=dict(facecolor=palette[seg], alpha=0.7),
                medianprops=dict(color="white", lw=2),
                whiskerprops=dict(color="white"),
                capprops=dict(color="white"),
                flierprops=dict(marker="o", color=palette[seg], alpha=0.3, ms=3),
            )

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=20, ha="right", color="white")
        ax.set_title("Predicted CLV Distribution per Segment",
                     color="white", fontsize=13, fontweight="bold")
        ax.set_ylabel("Predicted 90-Day CLV ($)", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333")
        plt.tight_layout()
    _savefig("rs_clv_by_segment.png")


def plot_revenue_treemap(
    labels: np.ndarray,
    segment_names: Dict[int, str],
    dollar_preds: np.ndarray,
    n_train: int,
) -> None:
    """Plot RS5: Revenue share treemap per segment."""
    try:
        import squarify
    except ImportError:
        logger.info("squarify not installed — skipping treemap. pip install squarify")
        return

    test_labels = labels[n_train:]
    if len(test_labels) != len(dollar_preds):
        return

    df = pd.DataFrame({
        "Segment": [segment_names.get(l, f"Cluster {l}") for l in test_labels],
        "CLV":     dollar_preds,
    })
    summary = df.groupby("Segment")["CLV"].sum().sort_values(ascending=False)
    total   = summary.sum()
    sizes   = summary.values
    labels_map = [
        f"{seg}\n${val:,.0f}\n({val/total*100:.1f}%)"
        for seg, val in summary.items()
    ]
    colors = [SEGMENT_PALETTE.get(seg, "#888") for seg in summary.index]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 7))
        squarify.plot(sizes=sizes, label=labels_map, color=colors,
                      alpha=0.85, ax=ax, text_kwargs={"color": "white", "fontsize": 9})
        ax.set_title("Revenue Share by Customer Segment",
                     color="white", fontsize=13, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
    _savefig("rs_revenue_treemap.png")


# ===========================================================================
# Save Results
# ===========================================================================

def save_richer_seg_results(
    X_all: pd.DataFrame,
    labels: np.ndarray,
    segment_names: Dict[int, str],
    dollar_preds: Optional[np.ndarray],
    y_test_raw: Optional[pd.Series],
    kmeans: KMeans,
    scaler: RobustScaler,
    pca: PCA,
    metrics_df: pd.DataFrame,
    optimal_k: int,
    n_train: int,
) -> None:
    """Save updated customer_segments_v2.csv and richer_seg_bundle.pkl."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Build output DataFrame
    out = X_all.copy().reset_index(drop=True)
    out["Cluster"]       = labels
    out["Segment_Name"]  = [segment_names.get(l, f"Cluster {l}") for l in labels]

    if dollar_preds is not None:
        clv = np.full(len(out), np.nan)
        clv[n_train:] = dollar_preds
        out["CLV_Predicted_90d"] = clv

    if y_test_raw is not None:
        actual = np.full(len(out), np.nan)
        actual[n_train:] = y_test_raw.values if hasattr(y_test_raw, "values") else np.array(y_test_raw)
        out["Actual_CLV"] = actual

    out.to_csv(GRAPHS_DIR / "customer_segments_v2.csv", index=False)
    logger.info(f"Saved: customer_segments_v2.csv ({len(out):,} rows)")

    metrics_df.to_csv(GRAPHS_DIR / "richer_seg_metrics.csv", index=False)
    logger.info("Saved: richer_seg_metrics.csv")

    bundle = {
        "kmeans":         kmeans,
        "scaler":         scaler,
        "pca":            pca,
        "cols_used":      [c for c in SEG_FEATURE_COLS if c in X_all.columns],
        "optimal_k":      optimal_k,
        "segment_names":  segment_names,
        "metrics":        metrics_df.to_dict("records"),
        "version":        "1.0.0",
        "timestamp":      datetime.utcnow().isoformat(),
    }
    joblib.dump(bundle, MODELS_DIR / "richer_seg_bundle.pkl")
    logger.info("Saved: richer_seg_bundle.pkl")


# ===========================================================================
# Summary Print
# ===========================================================================

def _print_summary(
    metrics_df: pd.DataFrame,
    optimal_k: int,
    segment_names: Dict[int, str],
    profiles: pd.DataFrame,
    dollar_preds: Optional[np.ndarray],
    labels: np.ndarray,
    n_train: int,
) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("  RICHER SEGMENTATION PIPELINE COMPLETE")
    print(sep)
    print(f"\n  K Selection Summary:")
    print(f"  {'k':>4} {'Silhouette':>12} {'Davies-Bouldin':>16} "
          f"{'Calinski-H':>12} {'Composite':>10}")
    print(f"  {'-'*56}")
    for _, row in metrics_df.iterrows():
        marker = " ◄ OPTIMAL" if int(row["k"]) == optimal_k else ""
        print(f"  {int(row['k']):>4} {row['silhouette']:>12.4f} "
              f"{row['davies_bouldin']:>16.4f} "
              f"{row['calinski_harabasz']:>12.0f} "
              f"{row['composite_score']:>10.4f}{marker}")

    print(f"\n  Segment Profiles (k={optimal_k}):")
    print(f"  {'Segment':<30} {'N':>6} {'Avg CLV':>10} {'Rev Share':>10}")
    print(f"  {'-'*58}")

    test_labels = labels[n_train:]
    for cid, seg_name in segment_names.items():
        mask = test_labels == cid
        n    = mask.sum()
        if dollar_preds is not None and n > 0:
            avg_clv   = dollar_preds[mask].mean()
            total_clv = dollar_preds[mask].sum()
            rev_share = total_clv / (dollar_preds.sum() + 1e-9) * 100
            print(f"  {seg_name:<30} {n:>6,} {avg_clv:>9.2f}$ {rev_share:>9.1f}%")
        else:
            total_mask = labels == cid
            print(f"  {seg_name:<30} {total_mask.sum():>6,}")

    print(f"\n{sep}")


# ===========================================================================
# Main Pipeline
# ===========================================================================

def run_richer_segmentation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test_raw: Optional[pd.Series] = None,
    dollar_preds: Optional[np.ndarray] = None,
    customer_ids: Optional[pd.Series] = None,
    k_range: List[int] = K_RANGE,
) -> Dict:
    """
    End-to-end richer segmentation pipeline.

    Workflow
    --------
    1. Prepare features (RobustScaler + PCA on combined train+test)
    2. Evaluate k = 2, 3, 4, 5 with all 3 metrics + composite score
    3. Select optimal k using composite score
    4. Fit final KMeans with optimal k
    5. Assign business segment names based on cluster profiles
    6. Generate 5 diagnostic plots
    7. Save updated segments CSV + bundle

    Parameters
    ----------
    X_train      : training features (FEATURE_COLS)
    X_test       : test features (FEATURE_COLS)
    y_test_raw   : actual dollar spend (test set)
    dollar_preds : CLV predictions from champion model
    customer_ids : customer ID series aligned to X_test
    k_range      : list of k values to evaluate (default [2,3,4,5])

    Returns
    -------
    dict with: kmeans, labels, segment_names, metrics_df, optimal_k
    """
    logger.info("=" * 70)
    logger.info("[RICHER SEG] Starting Richer Segmentation Pipeline v1.0.0")
    logger.info("=" * 70)

    X_all   = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    n_train = len(X_train)

    # ------------------------------------------------------------------
    # Step 1: Prepare features
    # ------------------------------------------------------------------
    logger.info("[RICHER SEG 1/7] Preparing features...")
    X_pca, scaler, pca = _prepare_features(X_all, fit=True)

    # ------------------------------------------------------------------
    # Step 2: Evaluate k range
    # ------------------------------------------------------------------
    logger.info("[RICHER SEG 2/7] Evaluating k range...")
    metrics_df = evaluate_k_range(X_pca, k_range)

    # ------------------------------------------------------------------
    # Step 3: Select optimal k
    # ------------------------------------------------------------------
    logger.info("[RICHER SEG 3/7] Selecting optimal k...")
    optimal_k = select_optimal_k(metrics_df)

    # ------------------------------------------------------------------
    # Step 4: Fit final KMeans
    # ------------------------------------------------------------------
    logger.info(f"[RICHER SEG 4/7] Fitting final KMeans (k={optimal_k})...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_SEED, n_init=20)
    labels = kmeans.fit_predict(X_pca)

    # ------------------------------------------------------------------
    # Step 5: Assign segment names
    # ------------------------------------------------------------------
    logger.info("[RICHER SEG 5/7] Assigning segment names...")
    profiles = _compute_cluster_profiles(
        X_all, labels, dollar_preds, y_test_raw, n_train
    )
    segment_names = assign_segment_names(profiles, optimal_k)
    logger.info(f"Segment names: {segment_names}")

    # ------------------------------------------------------------------
    # Step 6: Plots
    # ------------------------------------------------------------------
    logger.info("[RICHER SEG 6/7] Generating plots...")
    plot_k_comparison(metrics_df, optimal_k)
    plot_umap_5seg(X_pca, labels, segment_names)
    plot_segment_profiles(profiles, segment_names)

    if dollar_preds is not None:
        plot_clv_by_segment(labels, segment_names, dollar_preds, n_train)
        plot_revenue_treemap(labels, segment_names, dollar_preds, n_train)

    # ------------------------------------------------------------------
    # Step 7: Save
    # ------------------------------------------------------------------
    logger.info("[RICHER SEG 7/7] Saving results...")
    save_richer_seg_results(
        X_all, labels, segment_names, dollar_preds, y_test_raw,
        kmeans, scaler, pca, metrics_df, optimal_k, n_train,
    )

    _print_summary(
        metrics_df, optimal_k, segment_names, profiles,
        dollar_preds, labels, n_train,
    )

    print(f"\n  Files saved to: {GRAPHS_DIR}")
    print(f"  Bundle: {MODELS_DIR / 'richer_seg_bundle.pkl'}")

    return {
        "kmeans":        kmeans,
        "scaler":        scaler,
        "pca":           pca,
        "labels":        labels,
        "segment_names": segment_names,
        "metrics_df":    metrics_df,
        "optimal_k":     optimal_k,
        "profiles":      profiles,
        "X_all":         X_all,
        "n_train":       n_train,
    }