"""
Customer Intelligence Platform — Association Rules Module v1.0.0
================================================================
Market basket analysis using Apriori / FP-Growth algorithms.

Finds which products are frequently bought together, per customer segment,
enabling cross-sell recommendations and product affinity scoring.

Models
------
- FP-Growth (primary)  : fast, memory-efficient, scales to large catalogs
- Apriori (fallback)   : classic algorithm, used if mlxtend FP-Growth unavailable

Business Outputs
----------------
- Top product affinities per segment (Champions vs Loyal Customers)
- Cross-sell recommendation engine: given a product → top N co-purchased items
- Segment-level basket profiles
- Lift matrix heatmap for top products

Diagnostic Plots Saved
-----------------------
Plot AR1: ar_support_confidence.png   - scatter of support vs confidence
Plot AR2: ar_lift_heatmap.png         - lift matrix for top 15 products
Plot AR3: ar_top_rules_bar.png        - top 20 rules by lift per segment
Plot AR4: ar_segment_comparison.png   - Champions vs Loyal Customers affinity

CSVs Saved
----------
- association_rules.csv          - all rules with support/confidence/lift
- top_rules_per_segment.csv      - top 50 rules per segment
- product_recommendations.csv    - cross-sell lookup table

Usage
-----
    from src.association_rules import run_association_rules_pipeline

    ar_results = run_association_rules_pipeline(
        raw_df       = raw_df,
        customer_segments_df = customer_segments_df,
    )
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

from src.config import GRAPHS_DIR, MODELS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional mlxtend (FP-Growth + Apriori)
# ---------------------------------------------------------------------------
try:
    from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
    logger.info("mlxtend available — FP-Growth enabled.")
except ImportError:
    MLXTEND_AVAILABLE = False
    logger.warning(
        "mlxtend not installed. Run: pip install mlxtend\n"
        "Falling back to manual co-occurrence counting."
    )

# ===========================================================================
# Constants
# ===========================================================================

# Minimum support: item set must appear in at least this fraction of baskets
MIN_SUPPORT     = 0.02    # 2% — low enough to catch niche affinities
MIN_CONFIDENCE  = 0.20    # 20% — rule fires 1-in-5 times minimum
MIN_LIFT        = 1.10    # 10% lift above random — meaningful association
MAX_RULES       = 500     # cap to avoid memory issues on large datasets

# Product cleaning
MIN_PRODUCT_SUPPORT = 10  # product must appear in >= 10 baskets to be included
MAX_DESCRIPTION_LEN = 40  # truncate long product names for plots

# Top N for cross-sell recommendations
TOP_N_RECOMMENDATIONS = 5

# Plot style
STYLE  = 'dark_background'
ACCENT = '#5C4DB1'
SAFE   = '#2A9D8F'
WARN   = '#F4A261'
DANGER = '#E63946'

SEGMENT_COLOURS = {
    "Champions":       "#2E86AB",
    "Loyal Customers": "#5C4DB1",
    "All":             "#2A9D8F",
}


# ===========================================================================
# Data Preparation
# ===========================================================================

def _clean_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw transaction data for basket analysis.

    Steps:
    - Drop rows with missing Customer ID or Description
    - Remove cancellations (Invoice starts with 'C')
    - Remove negative quantities (returns)
    - Remove non-product stock codes (POST, D, M, BANK, etc.)
    - Normalise Description to uppercase stripped string
    """
    df = raw_df.copy()

    # Standardise column names
    df.columns = df.columns.str.strip()
    id_col   = "Customer ID" if "Customer ID" in df.columns else "CustomerID"
    inv_col  = "Invoice"     if "Invoice"     in df.columns else "InvoiceNo"
    desc_col = "Description"
    qty_col  = "Quantity"
    code_col = "StockCode"

    df = df.dropna(subset=[id_col, desc_col])
    df = df[~df[inv_col].astype(str).str.startswith("C")]
    df = df[df[qty_col] > 0]

    # Remove non-product codes
    non_product = {"POST", "D", "M", "BANK CHARGES", "PADS", "DOT", "CRUK"}
    df = df[~df[code_col].astype(str).str.upper().isin(non_product)]
    df = df[df[code_col].astype(str).str.match(r"^\d")]  # keep numeric codes only

    # Normalise description
    df[desc_col] = (
        df[desc_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .str[:MAX_DESCRIPTION_LEN]
    )

    # Cast customer ID to int
    df[id_col] = df[id_col].astype(float).astype(int)

    logger.info(
        f"Cleaned transactions: {len(df):,} rows | "
        f"{df[id_col].nunique():,} customers | "
        f"{df[desc_col].nunique():,} products | "
        f"{df[inv_col].nunique():,} baskets"
    )
    return df, id_col, inv_col, desc_col


def _build_basket_matrix(
    df: pd.DataFrame,
    inv_col: str,
    desc_col: str,
    min_product_support: int = MIN_PRODUCT_SUPPORT,
) -> pd.DataFrame:
    """
    Builds a binary basket matrix: rows = invoices, cols = products.
    Only keeps products appearing in >= min_product_support baskets.

    Returns
    -------
    basket : DataFrame of bool, shape (n_invoices, n_products)
    """
    # Filter low-frequency products
    product_counts = df.groupby(desc_col)[inv_col].nunique()
    valid_products = product_counts[product_counts >= min_product_support].index
    df_filtered = df[df[desc_col].isin(valid_products)]

    logger.info(
        f"After product filter (>={min_product_support} baskets): "
        f"{len(valid_products):,} products retained"
    )

    # Build basket matrix
    basket = (
        df_filtered
        .groupby([inv_col, desc_col])[desc_col]
        .count()
        .unstack(fill_value=0)
        .astype(bool)
    )

    logger.info(f"Basket matrix: {basket.shape[0]:,} baskets × {basket.shape[1]:,} products")
    return basket


def _get_segment_customer_ids(
    customer_segments_df: pd.DataFrame,
    segment_name: str,
    id_col: str = "Customer ID",
) -> Optional[set]:
    """Returns set of CustomerIDs for a given segment name, or None for all."""
    if segment_name == "All" or customer_segments_df is None:
        return None

    # Try to find CustomerID column
    id_candidates = ["CustomerID", "Customer ID", "customer_id"]
    seg_id_col = next((c for c in id_candidates if c in customer_segments_df.columns), None)

    if seg_id_col is None:
        logger.warning("No CustomerID column found in segments DataFrame — using all customers.")
        return None

    mask = customer_segments_df["Segment_Name"] == segment_name
    ids  = set(customer_segments_df.loc[mask, seg_id_col].astype(int).tolist())
    logger.info(f"Segment '{segment_name}': {len(ids):,} customers")
    return ids


# ===========================================================================
# FP-Growth / Apriori
# ===========================================================================

def _run_fpgrowth(
    basket: pd.DataFrame,
    min_support: float = MIN_SUPPORT,
    min_confidence: float = MIN_CONFIDENCE,
    min_lift: float = MIN_LIFT,
) -> pd.DataFrame:
    """
    Runs FP-Growth and extracts association rules.

    Returns empty DataFrame if mlxtend unavailable or no rules found.
    """
    if not MLXTEND_AVAILABLE:
        return _run_cooccurrence_fallback(basket)

    try:
        logger.info(
            f"Running FP-Growth | support>={min_support} | "
            f"confidence>={min_confidence} | lift>={min_lift}"
        )
        frequent_sets = fpgrowth(basket, min_support=min_support, use_colnames=True)

        if len(frequent_sets) == 0:
            logger.warning("FP-Growth found no frequent itemsets — try lowering min_support.")
            return pd.DataFrame()

        rules = association_rules(
            frequent_sets, metric="lift", min_threshold=min_lift
        )
        rules = rules[rules["confidence"] >= min_confidence]
        rules = rules.sort_values("lift", ascending=False).head(MAX_RULES)

        logger.info(f"FP-Growth: {len(frequent_sets):,} itemsets → {len(rules):,} rules")
        return rules

    except Exception as e:
        logger.warning(f"FP-Growth failed: {e} — falling back to co-occurrence.")
        return _run_cooccurrence_fallback(basket)


def _run_cooccurrence_fallback(basket: pd.DataFrame) -> pd.DataFrame:
    """
    Manual co-occurrence fallback when mlxtend is unavailable.
    Computes pairwise support, confidence, and lift for all product pairs.
    Returns a DataFrame in the same format as mlxtend association_rules.
    """
    logger.info("Running co-occurrence fallback...")
    n_baskets  = len(basket)
    products   = basket.columns.tolist()
    arr        = basket.values.astype(np.float32)

    # Support per product
    support_single = arr.mean(axis=0)

    records = []
    for i, p1 in enumerate(products):
        for j, p2 in enumerate(products):
            if i >= j:
                continue
            both      = (arr[:, i] & arr[:, j]).mean()
            if both < MIN_SUPPORT:
                continue
            conf_ij   = both / (support_single[i] + 1e-9)
            conf_ji   = both / (support_single[j] + 1e-9)
            lift      = both / (support_single[i] * support_single[j] + 1e-9)

            if lift >= MIN_LIFT:
                records.append({
                    "antecedents": frozenset([p1]),
                    "consequents": frozenset([p2]),
                    "support": both,
                    "confidence": conf_ij,
                    "lift": lift,
                })
                records.append({
                    "antecedents": frozenset([p2]),
                    "consequents": frozenset([p1]),
                    "support": both,
                    "confidence": conf_ji,
                    "lift": lift,
                })

    if not records:
        return pd.DataFrame()

    rules = pd.DataFrame(records).sort_values("lift", ascending=False).head(MAX_RULES)
    logger.info(f"Co-occurrence fallback: {len(rules):,} rules found")
    return rules


# ===========================================================================
# Cross-Sell Recommendation Engine
# ===========================================================================

def build_recommendation_lookup(rules: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Builds a product → top-N recommendations lookup from association rules.

    Returns
    -------
    lookup : {product_name: [{"product": ..., "confidence": ..., "lift": ...}]}
    """
    if rules is None or len(rules) == 0:
        return {}

    lookup = {}
    for _, row in rules.iterrows():
        antecedent = list(row["antecedents"])[0] if len(row["antecedents"]) == 1 else str(row["antecedents"])
        consequent = list(row["consequents"])[0] if len(row["consequents"]) == 1 else str(row["consequents"])

        if antecedent not in lookup:
            lookup[antecedent] = []
        lookup[antecedent].append({
            "product":    consequent,
            "confidence": round(float(row["confidence"]), 4),
            "lift":       round(float(row["lift"]), 4),
            "support":    round(float(row["support"]), 4),
        })

    # Sort each entry by lift and keep top N
    for k in lookup:
        lookup[k] = sorted(lookup[k], key=lambda x: x["lift"], reverse=True)[:TOP_N_RECOMMENDATIONS]

    logger.info(f"Recommendation lookup built: {len(lookup):,} products have recommendations")
    return lookup


def get_recommendations(
    product: str,
    lookup: Dict,
    top_n: int = TOP_N_RECOMMENDATIONS,
) -> List[Dict]:
    """
    Returns top-N cross-sell recommendations for a given product.

    Parameters
    ----------
    product : product name (exact match or partial)
    lookup  : recommendation lookup dict from build_recommendation_lookup()
    top_n   : number of recommendations to return

    Returns
    -------
    list of {product, confidence, lift, support} dicts
    """
    # Exact match first
    if product in lookup:
        return lookup[product][:top_n]

    # Partial match
    product_upper = product.upper()
    matches = [k for k in lookup if product_upper in k.upper()]
    if matches:
        return lookup[matches[0]][:top_n]

    return []


# ===========================================================================
# Diagnostic Plots
# ===========================================================================

def _savefig(name: str) -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAPHS_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_support_confidence(rules: pd.DataFrame, segment: str = "All") -> None:
    """Plot AR1: Support vs Confidence scatter coloured by lift."""
    if rules is None or len(rules) == 0:
        return

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 7))

        sc = ax.scatter(
            rules["support"], rules["confidence"],
            c=rules["lift"], cmap="RdYlGn",
            s=40, alpha=0.7, linewidths=0,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Lift", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        ax.set_title(
            f"Association Rules — Support vs Confidence\nSegment: {segment} | {len(rules):,} rules",
            color="white", fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Support", color="white")
        ax.set_ylabel("Confidence", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333")

        plt.tight_layout()
    _savefig("ar_support_confidence.png")


def plot_lift_heatmap(rules: pd.DataFrame, top_n: int = 15) -> None:
    """Plot AR2: Lift matrix heatmap for top N products."""
    if rules is None or len(rules) == 0:
        return

    # Get top products by frequency in rules
    all_products = []
    for _, row in rules.head(200).iterrows():
        all_products.extend(list(row["antecedents"]))
        all_products.extend(list(row["consequents"]))

    from collections import Counter
    top_products = [p for p, _ in Counter(all_products).most_common(top_n)]

    # Build lift matrix
    matrix = pd.DataFrame(1.0, index=top_products, columns=top_products)
    for _, row in rules.iterrows():
        ants = list(row["antecedents"])
        cons = list(row["consequents"])
        if len(ants) == 1 and len(cons) == 1:
            a, c = ants[0], cons[0]
            if a in top_products and c in top_products:
                matrix.loc[a, c] = row["lift"]

    # Truncate labels
    labels = [p[:25] + "…" if len(p) > 25 else p for p in top_products]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 11))
        sns.heatmap(
            matrix.values, xticklabels=labels, yticklabels=labels,
            cmap="RdYlGn", center=1.0, linewidths=0.3, ax=ax,
            cbar_kws={"label": "Lift"},
            vmin=0.5, vmax=matrix.values.max(),
        )
        ax.set_title(
            f"Product Lift Matrix — Top {top_n} Products\n(>1.0 = bought together more than random)",
            color="white", fontsize=12, fontweight="bold", pad=12
        )
        ax.tick_params(colors="white", axis="x", rotation=45, labelsize=7)
        ax.tick_params(colors="white", axis="y", labelsize=7)
        plt.tight_layout()
    _savefig("ar_lift_heatmap.png")


def plot_top_rules_bar(rules: pd.DataFrame, segment: str = "All", top_n: int = 20) -> None:
    """Plot AR3: Top rules by lift as horizontal bar chart."""
    if rules is None or len(rules) == 0:
        return

    top = rules.nlargest(top_n, "lift").copy()
    top["rule"] = (
        top["antecedents"].apply(lambda x: ", ".join(list(x)[:1])) +
        " → " +
        top["consequents"].apply(lambda x: ", ".join(list(x)[:1]))
    )
    top["rule"] = top["rule"].str[:55]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = [DANGER if l > 3 else (WARN if l > 2 else ACCENT)
                  for l in top["lift"]]
        bars = ax.barh(top["rule"][::-1], top["lift"][::-1],
                       color=colors[::-1], edgecolor="none")

        for bar, val in zip(bars, top["lift"][::-1]):
            ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}×", va="center", color="white", fontsize=8)

        ax.axvline(1.0, color="#555", ls="--", lw=1, label="Lift = 1 (random)")
        ax.set_title(
            f"Top {top_n} Association Rules by Lift\nSegment: {segment}",
            color="white", fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Lift", color="white")
        ax.tick_params(colors="white")
        ax.legend(labelcolor="white")
        for spine in ax.spines.values():
            spine.set_color("#333")
        plt.tight_layout()
    _savefig("ar_top_rules_bar.png")


def plot_segment_comparison(
    rules_champions: pd.DataFrame,
    rules_loyal: pd.DataFrame,
) -> None:
    """Plot AR4: Side-by-side top rules for Champions vs Loyal Customers."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor("#0D0D0D")

    for ax, rules, seg, color in [
        (axes[0], rules_champions, "Champions",       "#2E86AB"),
        (axes[1], rules_loyal,     "Loyal Customers", "#5C4DB1"),
    ]:
        ax.set_facecolor("#0D0D0D")
        if rules is None or len(rules) == 0:
            ax.text(0.5, 0.5, "No rules found", ha="center", va="center",
                    color="white", transform=ax.transAxes)
            ax.set_title(seg, color="white")
            continue

        top = rules.nlargest(15, "lift").copy()
        top["rule"] = (
            top["antecedents"].apply(lambda x: ", ".join(list(x)[:1])[:20]) +
            " →\n" +
            top["consequents"].apply(lambda x: ", ".join(list(x)[:1])[:20])
        )
        bars = ax.barh(top["rule"][::-1], top["lift"][::-1],
                       color=color, alpha=0.8, edgecolor="none")
        for bar, val in zip(bars, top["lift"][::-1]):
            ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}×", va="center", color="white", fontsize=7)

        ax.axvline(1.0, color="#555", ls="--", lw=1)
        ax.set_title(f"{seg}\n({len(rules):,} rules)", color="white",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Lift", color="white")
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")

    plt.suptitle("Product Affinity — Champions vs Loyal Customers",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _savefig("ar_segment_comparison.png")


# ===========================================================================
# Save Results
# ===========================================================================

def _rules_to_csv_safe(rules: pd.DataFrame) -> pd.DataFrame:
    """Convert frozenset columns to strings for CSV export."""
    df = rules.copy()
    if "antecedents" in df.columns:
        df["antecedents"] = df["antecedents"].apply(
            lambda x: ", ".join(sorted(x)) if isinstance(x, frozenset) else str(x)
        )
    if "consequents" in df.columns:
        df["consequents"] = df["consequents"].apply(
            lambda x: ", ".join(sorted(x)) if isinstance(x, frozenset) else str(x)
        )
    return df


def save_ar_results(
    all_rules: pd.DataFrame,
    rules_by_segment: Dict[str, pd.DataFrame],
    lookup: Dict,
    metrics: Dict,
) -> None:
    """Save all association rule outputs to GRAPHS_DIR."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Full rules CSV
    if all_rules is not None and len(all_rules) > 0:
        _rules_to_csv_safe(all_rules).to_csv(
            GRAPHS_DIR / "association_rules.csv", index=False
        )
        logger.info(f"Saved: association_rules.csv ({len(all_rules):,} rules)")

    # Top rules per segment
    segment_dfs = []
    for seg, rules in rules_by_segment.items():
        if rules is not None and len(rules) > 0:
            df = _rules_to_csv_safe(rules.head(50))
            df.insert(0, "Segment", seg)
            segment_dfs.append(df)

    if segment_dfs:
        pd.concat(segment_dfs).to_csv(
            GRAPHS_DIR / "top_rules_per_segment.csv", index=False
        )
        logger.info("Saved: top_rules_per_segment.csv")

    # Product recommendations lookup
    rec_rows = []
    for product, recs in lookup.items():
        for r in recs:
            rec_rows.append({"product": product, **r})
    if rec_rows:
        pd.DataFrame(rec_rows).to_csv(
            GRAPHS_DIR / "product_recommendations.csv", index=False
        )
        logger.info(f"Saved: product_recommendations.csv ({len(rec_rows):,} pairs)")

    # Bundle
    bundle = {
        "lookup":      lookup,
        "metrics":     metrics,
        "version":     "1.0.0",
        "timestamp":   datetime.utcnow().isoformat(),
    }
    joblib.dump(bundle, MODELS_DIR / "association_rules_bundle.pkl")
    logger.info(f"Saved: association_rules_bundle.pkl")


# ===========================================================================
# Leaderboard Print
# ===========================================================================

def _print_summary(metrics: Dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  ASSOCIATION RULES PIPELINE COMPLETE")
    print(sep)
    print(f"  Total rules found        : {metrics.get('n_rules_total', 0):,}")
    print(f"  Products with recs       : {metrics.get('n_products_with_recs', 0):,}")
    print(f"  Avg lift (all rules)     : {metrics.get('avg_lift', 0):.3f}×")
    print(f"  Max lift                 : {metrics.get('max_lift', 0):.3f}×")
    print(f"  Champions rules          : {metrics.get('n_rules_champions', 0):,}")
    print(f"  Loyal Customers rules    : {metrics.get('n_rules_loyal', 0):,}")
    print(sep)
    print(f"\n  Files saved to: {GRAPHS_DIR}")
    print(f"  Next: run_richer_segmentation() to upgrade k=2 → k=5")


# ===========================================================================
# Main Pipeline
# ===========================================================================

def run_association_rules_pipeline(
    raw_df: pd.DataFrame,
    customer_segments_df: Optional[pd.DataFrame] = None,
    min_support: float = MIN_SUPPORT,
    min_confidence: float = MIN_CONFIDENCE,
    min_lift: float = MIN_LIFT,
) -> Dict:
    """
    End-to-end association rules pipeline.

    Workflow
    --------
    1. Clean transaction data
    2. Build basket matrix (all customers)
    3. Run FP-Growth on all customers
    4. Run FP-Growth per segment (Champions / Loyal Customers)
    5. Build cross-sell recommendation lookup
    6. Generate 4 diagnostic plots
    7. Save CSVs + bundle

    Parameters
    ----------
    raw_df               : raw transaction DataFrame (Invoice, StockCode,
                           Description, Quantity, Customer ID)
    customer_segments_df : output of segmentation pipeline (with Segment_Name col)
    min_support          : minimum support threshold (default 0.02)
    min_confidence       : minimum confidence threshold (default 0.20)
    min_lift             : minimum lift threshold (default 1.10)

    Returns
    -------
    dict with: all_rules, rules_by_segment, lookup, metrics
    """
    logger.info("=" * 70)
    logger.info("[AR] Starting Association Rules Pipeline v1.0.0")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Clean transactions
    # ------------------------------------------------------------------
    logger.info("[AR 1/7] Cleaning transaction data...")
    clean_df, id_col, inv_col, desc_col = _clean_transactions(raw_df)

    # ------------------------------------------------------------------
    # Step 2: Build basket matrix — all customers
    # ------------------------------------------------------------------
    logger.info("[AR 2/7] Building basket matrix...")
    basket_all = _build_basket_matrix(clean_df, inv_col, desc_col)

    # ------------------------------------------------------------------
    # Step 3: FP-Growth — all customers
    # ------------------------------------------------------------------
    logger.info("[AR 3/7] Running FP-Growth on all customers...")
    all_rules = _run_fpgrowth(basket_all, min_support, min_confidence, min_lift)

    # ------------------------------------------------------------------
    # Step 4: Per-segment rules
    # ------------------------------------------------------------------
    logger.info("[AR 4/7] Running FP-Growth per segment...")
    rules_by_segment = {}
    segments_to_run  = ["All"]

    if customer_segments_df is not None and "Segment_Name" in customer_segments_df.columns:
        segments_to_run = list(customer_segments_df["Segment_Name"].unique()) + ["All"]

    for seg in segments_to_run:
        if seg == "All":
            rules_by_segment["All"] = all_rules
            continue

        cust_ids = _get_segment_customer_ids(customer_segments_df, seg, id_col)
        if cust_ids is None or len(cust_ids) == 0:
            continue

        # Filter transactions to this segment
        seg_df = clean_df[clean_df[id_col].isin(cust_ids)]
        if len(seg_df) == 0:
            logger.warning(f"No transactions found for segment '{seg}'")
            continue

        try:
            basket_seg          = _build_basket_matrix(seg_df, inv_col, desc_col,
                                                        min_product_support=5)
            rules_by_segment[seg] = _run_fpgrowth(
                basket_seg, min_support, min_confidence, min_lift
            )
            logger.info(
                f"Segment '{seg}': "
                f"{len(rules_by_segment[seg]):,} rules found"
            )
        except Exception as e:
            logger.warning(f"Segment '{seg}' failed: {e}")
            rules_by_segment[seg] = pd.DataFrame()

    # ------------------------------------------------------------------
    # Step 5: Build recommendation lookup
    # ------------------------------------------------------------------
    logger.info("[AR 5/7] Building cross-sell recommendation lookup...")
    lookup = build_recommendation_lookup(all_rules)

    # ------------------------------------------------------------------
    # Step 6: Plots
    # ------------------------------------------------------------------
    logger.info("[AR 6/7] Generating plots...")
    plot_support_confidence(all_rules, "All Customers")
    plot_lift_heatmap(all_rules)
    plot_top_rules_bar(all_rules, "All Customers")

    rules_champ = rules_by_segment.get("Champions", pd.DataFrame())
    rules_loyal = rules_by_segment.get("Loyal Customers", pd.DataFrame())
    plot_segment_comparison(rules_champ, rules_loyal)

    # ------------------------------------------------------------------
    # Step 7: Save
    # ------------------------------------------------------------------
    logger.info("[AR 7/7] Saving results...")

    metrics = {
        "n_rules_total":          len(all_rules) if all_rules is not None else 0,
        "n_products_with_recs":   len(lookup),
        "avg_lift":               float(all_rules["lift"].mean()) if all_rules is not None and len(all_rules) > 0 else 0,
        "max_lift":               float(all_rules["lift"].max())  if all_rules is not None and len(all_rules) > 0 else 0,
        "n_rules_champions":      len(rules_champ) if rules_champ is not None else 0,
        "n_rules_loyal":          len(rules_loyal) if rules_loyal is not None else 0,
        "min_support":            min_support,
        "min_confidence":         min_confidence,
        "min_lift":               min_lift,
    }

    save_ar_results(all_rules, rules_by_segment, lookup, metrics)
    _print_summary(metrics)

    return {
        "all_rules":        all_rules,
        "rules_by_segment": rules_by_segment,
        "lookup":           lookup,
        "metrics":          metrics,
        "basket":           basket_all,
    }