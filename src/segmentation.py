"""
Customer Segmentation Module - v1.0.0
======================================
Unsupervised customer segmentation using the same RFM + behavioral features
from the CLV pipeline (P1). Builds directly on top of feature_engineering.py
output - zero new data acquisition required.

Models implemented
------------------
1. K-Means          - Primary RFM segmentation (canonical, interpretable)
2. DBSCAN           - Density-based clustering (handles outliers natively)
3. Gaussian Mixture - Soft probabilistic cluster assignments
4. Hierarchical     - Agglomerative clustering + dendrogram
5. PCA              - Dimensionality reduction for visualization + preprocessing
6. UMAP             - Non-linear 2D embedding for cluster visualization
7. t-SNE            - Alternative 2D embedding (comparison with UMAP)

Evaluation metrics (no labels available)
-----------------------------------------
- Silhouette Score       - cluster cohesion vs separation [-1, 1], higher better
- Davies-Bouldin Index   - intra/inter cluster ratio, lower better
- Calinski-Harabasz      - variance ratio criterion, higher better
- Elbow Method           - inertia vs k for optimal K-Means k
- Reconstruction Error   - for Autoencoder (in anomaly.py)

Diagnostic plots saved
-----------------------
Plot S1: Elbow curve - inertia vs k (K-Means)
Plot S2: Silhouette score vs k
Plot S3: UMAP 2D - coloured by K-Means cluster
Plot S4: t-SNE 2D - coloured by K-Means cluster
Plot S5: Cluster profile heatmap - avg feature values per cluster
Plot S6: Segment × CLV heatmap - cluster vs CLV tier
Plot S7: Dendrogram - hierarchical clustering
Plot S8: RFM 3D scatter - coloured by cluster
Plot S9: PCA explained variance - cumulative scree plot
Plot S10: DBSCAN cluster map - noise points highlighted

Business outputs saved
-----------------------
- segment_profiles.csv     - avg RFM + CLV per cluster
- customer_segments.csv    - Customer-level cluster assignments + CLV tier
- segmentation_metrics.csv - all evaluation scores across models

Usage (from main_execution.ipynb or segmentation.ipynb)
---------------------------------------------------------
    from src.segmentation import (
        run_segmentation_pipeline,
        get_optimal_k,
        assign_segments,
    )

    seg_results = run_segmentation_pipeline(
        X_train     = X_train,
        X_test      = X_test,
        y_test_raw  = y_test_raw,
        customer_ids = customer_ids_test,
    )
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional

# Sklearn clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.pipeline import Pipeline

# Scipy for dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from src.config import GRAPHS_DIR, MODELS_DIR, RANDOM_SEED, FEATURE_COLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional UMAP - graceful degradation if not installed
# ---------------------------------------------------------------------------
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.info("umap-learn not installed - UMAP plots skipped. Run: pip install umap-learn")

# ---------------------------------------------------------------------------
# Optional Optuna - graceful degradation if not installed
# ---------------------------------------------------------------------------
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.info("optuna not installed - hyperparameter tuning skipped. Run: pip install optuna")


# ===========================================================================
# Constants
# ===========================================================================

# Core RFM features for primary segmentation - interpretable and canonical
RFM_COLS = ['Recency', 'Frequency', 'Monetary']

# Extended feature set for richer clustering
EXTENDED_COLS = [
    'Recency', 'Frequency', 'Monetary',
    'Days_Since_Purchase', 'Purchase_Rate',
    'Unique_Products', 'Avg_Basket_Size',
    'Return_Rate', 'Max_Single_Order',
    'Monetary_Percentile',
]

# K-Means search range
K_MIN = 2
K_MAX = 10

# DBSCAN defaults (tuned by Optuna if available)
DBSCAN_EPS_DEFAULT    = 1.5
DBSCAN_MIN_SAMPLES    = 5

# CLV tier thresholds (dollar scale, 90-day - mirrors streamlit_app.py)
CLV_TIER_THRESHOLDS = {
    'Whale':   1200.0,
    'Mid':     150.0,
    'Low':     0.01,
    'Churned': 0.0,
}

# Segment name mapping - assigned after profiling (updated post-analysis)
SEGMENT_LABELS = {
    0: 'Champions',
    1: 'Loyal Customers',
    2: 'At Risk',
    3: 'Lost / Churned',
    4: 'New Customers',
    5: 'Promising',
    6: 'Hibernating',
    7: 'Cannot Lose Them',
}

# Colour palette for clusters (up to 10)
CLUSTER_COLORS = [
    '#5C4DB1', '#2E86AB', '#F4A261', '#E63946',
    '#2A9D8F', '#E9C46A', '#264653', '#F77F00',
    '#80B918', '#9B2226',
]


# ===========================================================================
# Preprocessing
# ===========================================================================

def _prepare_features(
    X: pd.DataFrame,
    feature_set: str = 'extended',
    scaler: Optional[RobustScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, RobustScaler, list]:
    """
    Selects and scales features for clustering.

    Uses RobustScaler (median + IQR) instead of StandardScaler because:
    - CLV features are heavily right-skewed (whale customers)
    - StandardScaler's mean is distorted by outliers
    - RobustScaler preserves outlier structure without distorting cluster geometry

    Parameters
    ----------
    X            : input feature DataFrame (FEATURE_COLS)
    feature_set  : 'rfm' for core 3 features, 'extended' for 10 features
    scaler       : pre-fitted scaler (pass when transforming test data)
    fit_scaler   : if True, fit scaler on X (training); if False, transform only

    Returns
    -------
    X_scaled     : scaled numpy array
    scaler       : fitted RobustScaler
    cols_used    : list of column names used
    """
    cols = RFM_COLS if feature_set == 'rfm' else EXTENDED_COLS
    cols_available = [c for c in cols if c in X.columns]

    if len(cols_available) < len(cols):
        missing = set(cols) - set(cols_available)
        logger.warning(f"Missing features for segmentation: {missing}. Using available: {cols_available}")

    X_sub = X[cols_available].copy()

    # Fill any NaN with median (robust to outliers)
    X_sub = X_sub.fillna(X_sub.median())

    if scaler is None:
        scaler = RobustScaler()

    if fit_scaler:
        X_scaled = scaler.fit_transform(X_sub)
    else:
        X_scaled = scaler.transform(X_sub)

    return X_scaled, scaler, cols_available


# ===========================================================================
# Optimal K Selection
# ===========================================================================

def get_optimal_k(
    X_scaled: np.ndarray,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    save_plots: bool = True,
) -> int:
    """
    Determines optimal number of clusters using Elbow + Silhouette methods.

    The elbow method finds the k where adding another cluster gives diminishing
    returns on inertia reduction. The silhouette score finds the k where
    clusters are most cohesive and well-separated.

    When both methods agree → use that k.
    When they disagree → prefer silhouette (more meaningful business metric).

    Parameters
    ----------
    X_scaled  : scaled feature array
    k_min     : minimum k to test
    k_max     : maximum k to test
    save_plots: if True, saves elbow + silhouette plots to GRAPHS_DIR

    Returns
    -------
    optimal_k : int - recommended number of clusters
    """
    logger.info(f"[SEG] Finding optimal K in range [{k_min}, {k_max}]...")

    inertias    = []
    silhouettes = []
    db_scores   = []
    ch_scores   = []
    k_range     = range(k_min, k_max + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)

        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        db_scores.append(davies_bouldin_score(X_scaled, labels))
        ch_scores.append(calinski_harabasz_score(X_scaled, labels))

        logger.info(
            f"  k={k} | Inertia: {km.inertia_:,.0f} | "
            f"Silhouette: {silhouettes[-1]:.4f} | "
            f"DB: {db_scores[-1]:.4f} | "
            f"CH: {ch_scores[-1]:.0f}"
        )

    # Elbow detection - largest second derivative of inertia curve
    inertia_arr = np.array(inertias)
    second_deriv = np.diff(np.diff(inertia_arr))
    elbow_k = list(k_range)[np.argmax(second_deriv) + 1]

    # Silhouette optimal - highest silhouette score
    sil_k = list(k_range)[np.argmax(silhouettes)]

    # Final decision - prefer silhouette if they disagree
    optimal_k = sil_k
    logger.info(
        f"Elbow method suggests k={elbow_k} | "
        f"Silhouette suggests k={sil_k} | "
        f"Chosen: k={optimal_k}"
    )

    if save_plots:
        # Plot S1 + S2: Elbow curve and Silhouette score
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Elbow curve
        axes[0].plot(list(k_range), inertias, 'bo-', lw=2, markersize=6)
        axes[0].axvline(x=elbow_k, color='red', lw=2, linestyle='--',
                        label=f'Elbow at k={elbow_k}')
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia (Within-cluster SSE)', fontsize=12)
        axes[0].set_title('Plot S1: Elbow Method - Optimal K Selection', fontsize=13)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Silhouette score
        axes[1].plot(list(k_range), silhouettes, 'go-', lw=2, markersize=6)
        axes[1].axvline(x=sil_k, color='red', lw=2, linestyle='--',
                        label=f'Optimal at k={sil_k} (score={max(silhouettes):.4f})')
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Plot S2: Silhouette Score vs k', fontsize=13)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(GRAPHS_DIR / 'seg_elbow_silhouette.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        logger.info("Plot S1+S2 saved: seg_elbow_silhouette.png")

    return optimal_k


# ===========================================================================
# K-Means Segmentation (Primary Model)
# ===========================================================================

def fit_kmeans(
    X_scaled: np.ndarray,
    k: int,
    X_original: pd.DataFrame,
    cols_used: list,
) -> Tuple[KMeans, np.ndarray, pd.DataFrame]:
    """
    Fits K-Means on scaled features and returns cluster assignments + profiles.

    K-Means is the primary segmentation model because:
    - Interpretable centroids map directly to RFM business segments
    - Fast on ~4,300 customers
    - Results overlay cleanly onto CLV predictions

    Parameters
    ----------
    X_scaled    : scaled feature array
    k           : number of clusters
    X_original  : original unscaled DataFrame (for profile computation)
    cols_used   : feature column names

    Returns
    -------
    kmeans      : fitted KMeans model
    labels      : cluster assignment array (n_samples,)
    profiles    : DataFrame - avg feature values per cluster
    """
    logger.info(f"[SEG] Fitting K-Means with k={k}...")

    kmeans = KMeans(
        n_clusters   = k,
        random_state = RANDOM_SEED,
        n_init       = 20,          # more restarts = more stable solution
        max_iter     = 500,
        algorithm    = 'lloyd',
    )
    labels = kmeans.fit_predict(X_scaled)

    # Cluster profiles - business-interpretable centroids
    profile_df = X_original[cols_used].copy()
    profile_df['Cluster'] = labels
    profiles = (
        profile_df.groupby('Cluster')[cols_used]
        .mean()
        .round(2)
    )
    profiles['N'] = profile_df.groupby('Cluster').size()
    profiles['Pct'] = (profiles['N'] / len(labels) * 100).round(1)

    # Evaluation metrics
    sil   = silhouette_score(X_scaled, labels)
    db    = davies_bouldin_score(X_scaled, labels)
    ch    = calinski_harabasz_score(X_scaled, labels)

    logger.info(
        f"K-Means (k={k}) | Silhouette: {sil:.4f} | "
        f"Davies-Bouldin: {db:.4f} | Calinski-Harabasz: {ch:.0f}"
    )
    logger.info(f"Cluster sizes:\n{profiles[['N', 'Pct']].to_string()}")

    return kmeans, labels, profiles


# ===========================================================================
# DBSCAN Segmentation
# ===========================================================================

def fit_dbscan(
    X_scaled: np.ndarray,
    eps: float = DBSCAN_EPS_DEFAULT,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> Tuple[DBSCAN, np.ndarray, dict]:
    """
    Fits DBSCAN for density-based clustering with automatic outlier detection.

    DBSCAN assigns label=-1 to noise points (outliers) - these are customers
    whose behavioral profile does not fit any dense neighborhood. In the
    context of retail CLV, noise points often correspond to:
    - One-time high-value buyers (unusual purchase pattern)
    - B2B wholesale buyers with irregular order cycles
    - Fraudulent accounts (passed to anomaly.py for deeper analysis)

    Parameters
    ----------
    X_scaled    : scaled feature array
    eps         : maximum distance between neighbors
    min_samples : minimum points to form a dense region

    Returns
    -------
    dbscan  : fitted DBSCAN model
    labels  : cluster labels (-1 = noise/outlier)
    metrics : dict of evaluation metrics
    """
    logger.info(f"[SEG] Fitting DBSCAN (eps={eps}, min_samples={min_samples})...")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    noise_pct  = n_noise / len(labels) * 100

    logger.info(
        f"DBSCAN - Clusters found: {n_clusters} | "
        f"Noise points: {n_noise} ({noise_pct:.1f}%)"
    )

    metrics = {
        'n_clusters': n_clusters,
        'n_noise':    n_noise,
        'noise_pct':  noise_pct,
    }

    # Only compute silhouette if more than 1 cluster found (excluding noise)
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 1:
            sil = silhouette_score(X_scaled[mask], labels[mask])
            metrics['silhouette'] = sil
            logger.info(f"DBSCAN Silhouette (excl. noise): {sil:.4f}")

    return dbscan, labels, metrics


def tune_dbscan_optuna(
    X_scaled: np.ndarray,
    n_trials: int = 50,
) -> Tuple[float, int]:
    """
    Uses Optuna to find optimal DBSCAN eps and min_samples.
    Maximises silhouette score over the parameter space.
    Returns best (eps, min_samples). Falls back to defaults if Optuna absent.
    """
    if not OPTUNA_AVAILABLE:
        logger.info("Optuna not available - using DBSCAN defaults.")
        return DBSCAN_EPS_DEFAULT, DBSCAN_MIN_SAMPLES

    logger.info(f"[SEG] Tuning DBSCAN with Optuna ({n_trials} trials)...")

    def objective(trial):
        eps         = trial.suggest_float('eps', 0.3, 3.0)
        min_samples = trial.suggest_int('min_samples', 3, 15)

        labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters < 2:
            return -1.0   # Penalise degenerate solutions

        mask = labels != -1
        if mask.sum() < 10:
            return -1.0

        return silhouette_score(X_scaled[mask], labels[mask])

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_eps     = study.best_params['eps']
    best_min_smp = study.best_params['min_samples']

    logger.info(
        f"Optuna DBSCAN best: eps={best_eps:.3f}, "
        f"min_samples={best_min_smp} | "
        f"Silhouette: {study.best_value:.4f}"
    )
    return best_eps, best_min_smp


# ===========================================================================
# Gaussian Mixture Model
# ===========================================================================

def fit_gmm(
    X_scaled: np.ndarray,
    n_components: int,
    X_original: pd.DataFrame,
    cols_used: list,
) -> Tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """
    Fits Gaussian Mixture Model for soft probabilistic cluster assignments.

    GMM differs from K-Means in that each customer receives a probability
    of belonging to each cluster - not a hard assignment. This is useful for:
    - Customers on the boundary between 'Loyal' and 'At Risk' segments
    - Uncertainty-aware marketing (don't invest heavily in borderline cases)
    - More flexible cluster shapes (elliptical vs spherical for K-Means)

    Returns
    -------
    gmm         : fitted GaussianMixture model
    labels      : hard cluster assignments (argmax of probabilities)
    proba       : soft probabilities - shape (n_samples, n_components)
    """
    logger.info(f"[SEG] Fitting GMM with {n_components} components...")

    gmm = GaussianMixture(
        n_components    = n_components,
        covariance_type = 'full',
        random_state    = RANDOM_SEED,
        n_init          = 5,
        max_iter        = 300,
    )
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    proba  = gmm.predict_proba(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    bic = gmm.bic(X_scaled)
    aic = gmm.aic(X_scaled)

    logger.info(
        f"GMM ({n_components} components) | "
        f"Silhouette: {sil:.4f} | BIC: {bic:.0f} | AIC: {aic:.0f}"
    )

    return gmm, labels, proba


# ===========================================================================
# Hierarchical Clustering
# ===========================================================================

def fit_hierarchical(
    X_scaled: np.ndarray,
    n_clusters: int,
    save_dendrogram: bool = True,
) -> Tuple[AgglomerativeClustering, np.ndarray]:
    """
    Fits Agglomerative (bottom-up) hierarchical clustering.

    Hierarchical clustering doesn't require specifying k upfront - the
    dendrogram (Plot S7) shows the full merge tree at all levels, which
    helps validate that the chosen k is meaningful (large jumps in the
    dendrogram = natural cluster boundaries).

    Parameters
    ----------
    X_scaled        : scaled feature array
    n_clusters      : number of clusters to cut the dendrogram at
    save_dendrogram : if True, saves dendrogram plot to GRAPHS_DIR

    Returns
    -------
    model  : fitted AgglomerativeClustering
    labels : cluster assignments
    """
    logger.info(f"[SEG] Fitting Hierarchical Clustering (n={n_clusters})...")

    model = AgglomerativeClustering(
        n_clusters = n_clusters,
        linkage    = 'ward',          # minimises within-cluster variance
        metric     = 'euclidean',
    )
    labels = model.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db  = davies_bouldin_score(X_scaled, labels)
    logger.info(f"Hierarchical | Silhouette: {sil:.4f} | Davies-Bouldin: {db:.4f}")

    if save_dendrogram:
        try:
            # Subsample for dendrogram readability (scipy linkage is O(n²))
            max_dendro_samples = min(200, len(X_scaled))
            idx = np.random.RandomState(RANDOM_SEED).choice(
                len(X_scaled), max_dendro_samples, replace=False
            )
            X_sub = X_scaled[idx]

            Z = linkage(X_sub, method='ward')

            fig, ax = plt.subplots(figsize=(16, 6))
            dendrogram(
                Z, ax=ax,
                truncate_mode='lastp',
                p=30,
                leaf_rotation=90,
                leaf_font_size=10,
                show_contracted=True,
            )
            ax.set_title(
                f'Plot S7: Hierarchical Clustering Dendrogram\n'
                f'(Ward linkage, {max_dendro_samples} customer sample)',
                fontsize=13
            )
            ax.set_xlabel('Customer Index', fontsize=11)
            ax.set_ylabel('Distance (Ward)', fontsize=11)
            ax.axhline(y=ax.get_ylim()[1] * 0.4, color='red', lw=2,
                       linestyle='--', label=f'Cut at k={n_clusters}')
            ax.legend(fontsize=10)
            fig.tight_layout()
            fig.savefig(GRAPHS_DIR / 'seg_dendrogram.png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            logger.info("Plot S7 saved: seg_dendrogram.png")
        except Exception as e:
            logger.warning(f"Dendrogram plot failed: {e}")

    return model, labels


# ===========================================================================
# Dimensionality Reduction - PCA
# ===========================================================================

def fit_pca(
    X_scaled: np.ndarray,
    n_components: Optional[int] = None,
    variance_threshold: float = 0.95,
    save_plot: bool = True,
) -> Tuple[PCA, np.ndarray]:
    """
    Fits PCA for dimensionality reduction.

    Two purposes:
    1. Pre-processing - reduce to components explaining 95% of variance
       before clustering (reduces noise in high-dimensional space)
    2. Visualization - reduce to 2 components for scatter plots

    Parameters
    ----------
    X_scaled           : scaled feature array
    n_components       : fixed number of components (None = auto by variance)
    variance_threshold : if n_components is None, keep enough PCs for this
                         cumulative explained variance (default 95%)
    save_plot          : save scree plot

    Returns
    -------
    pca      : fitted PCA model
    X_pca    : transformed array
    """
    logger.info("[SEG] Fitting PCA...")

    # First fit full PCA to find variance threshold
    pca_full = PCA(random_state=RANDOM_SEED)
    pca_full.fit(X_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    if n_components is None:
        n_components = int(np.argmax(cumvar >= variance_threshold) + 1)
        logger.info(
            f"PCA: {n_components} components explain "
            f"{cumvar[n_components-1]*100:.1f}% of variance"
        )

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)

    if save_plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scree plot - individual explained variance
        axes[0].bar(
            range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_ * 100,
            color='#5C4DB1', alpha=0.8
        )
        axes[0].set_xlabel('Principal Component', fontsize=12)
        axes[0].set_ylabel('Explained Variance (%)', fontsize=12)
        axes[0].set_title('Plot S9: PCA Scree Plot', fontsize=13)
        axes[0].grid(axis='y', alpha=0.3)

        # Cumulative variance
        axes[1].plot(range(1, len(cumvar) + 1), cumvar * 100, 'bo-', lw=2)
        axes[1].axhline(y=variance_threshold * 100, color='red', lw=2,
                        linestyle='--', label=f'{variance_threshold*100:.0f}% threshold')
        axes[1].axvline(x=n_components, color='green', lw=2,
                        linestyle='--', label=f'{n_components} components selected')
        axes[1].set_xlabel('Number of Components', fontsize=12)
        axes[1].set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
        axes[1].set_title('Cumulative Explained Variance', fontsize=13)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(GRAPHS_DIR / 'seg_pca_variance.png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        logger.info("Plot S9 saved: seg_pca_variance.png")

    return pca, X_pca


# ===========================================================================
# 2D Embeddings - UMAP and t-SNE
# ===========================================================================

def compute_umap_embedding(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    title_suffix: str = "K-Means",
) -> Optional[np.ndarray]:
    """
    Computes UMAP 2D embedding and saves scatter plot coloured by cluster.

    UMAP (Uniform Manifold Approximation and Projection) preserves both
    local and global structure better than t-SNE. It is the preferred
    visualization for customer segmentation because:
    - Cluster boundaries are clearer
    - Runs faster than t-SNE on large datasets
    - Deterministic with fixed random_state

    Returns None if umap-learn is not installed.
    """
    if not UMAP_AVAILABLE:
        logger.info("UMAP not available - Plot S3 skipped.")
        return None

    logger.info("[SEG] Computing UMAP embedding...")

    reducer = umap.UMAP(
        n_components = 2,
        n_neighbors  = 15,
        min_dist     = 0.1,
        random_state = RANDOM_SEED,
        verbose      = False,
    )
    embedding = reducer.fit_transform(X_scaled)

    _save_2d_scatter(
        embedding, labels,
        title=f'Plot S3: UMAP 2D Embedding - {title_suffix} Clusters',
        fname='seg_umap.png',
        xlabel='UMAP Dimension 1',
        ylabel='UMAP Dimension 2',
    )

    return embedding


def compute_tsne_embedding(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    title_suffix: str = "K-Means",
) -> np.ndarray:
    """
    Computes t-SNE 2D embedding and saves scatter plot coloured by cluster.

    t-SNE is computationally expensive - subsampled to 1,000 customers
    if dataset is larger. Preserves local neighborhood structure well
    but can distort global distances (use UMAP for global structure).
    """
    logger.info("[SEG] Computing t-SNE embedding (may take 30-60 seconds)...")

    max_tsne = min(1000, len(X_scaled))
    if len(X_scaled) > max_tsne:
        idx = np.random.RandomState(RANDOM_SEED).choice(len(X_scaled), max_tsne, replace=False)
        X_sub = X_scaled[idx]
        labels_sub = labels[idx]
        logger.info(f"t-SNE subsampled to {max_tsne} customers for speed.")
    else:
        X_sub = X_scaled
        labels_sub = labels

    tsne = TSNE(
        n_components = 2,
        perplexity   = min(30, len(X_sub) // 4),
        random_state = RANDOM_SEED,
        n_iter       = 1000,
        verbose      = 0,
    )
    embedding = tsne.fit_transform(X_sub)

    _save_2d_scatter(
        embedding, labels_sub,
        title=f'Plot S4: t-SNE 2D Embedding - {title_suffix} Clusters',
        fname='seg_tsne.png',
        xlabel='t-SNE Dimension 1',
        ylabel='t-SNE Dimension 2',
    )

    return embedding


def _save_2d_scatter(
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str,
    fname: str,
    xlabel: str = 'Dim 1',
    ylabel: str = 'Dim 2',
) -> None:
    """Saves a 2D scatter plot coloured by cluster label."""
    unique_labels = sorted(set(labels))
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = '#888888' if label == -1 else CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        seg_name = 'Noise / Outlier' if label == -1 else SEGMENT_LABELS.get(label, f'Cluster {label}')
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, label=f'{seg_name} (n={mask.sum()})',
            alpha=0.6, s=20, edgecolors='none',
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc='best', markerscale=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / fname, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {fname}")


# ===========================================================================
# Cluster Profile & Business Visualizations
# ===========================================================================

def plot_cluster_profiles(
    X_original: pd.DataFrame,
    labels: np.ndarray,
    cols_used: list,
    model_name: str = "K-Means",
) -> pd.DataFrame:
    """Generates Plot S5: Cluster profile heatmap."""
    try:
        tmp = X_original[cols_used].copy()
        tmp["_cluster_"] = labels
        grp = tmp.groupby("_cluster_")[cols_used].mean()
        arr = grp.values.astype(float)
        col_names = list(grp.columns)
        row_names = [SEGMENT_LABELS.get(int(idx), f"Cluster {idx}") for idx in grp.index]
        col_min = arr.min(axis=0)
        col_max = arr.max(axis=0)
        arr_norm = (arr - col_min) / (col_max - col_min + 1e-8)
        profiles_norm = pd.DataFrame(arr_norm.astype(float), index=row_names, columns=col_names)
        fig, ax = plt.subplots(figsize=(14, max(6, len(profiles_norm) * 0.8)))
        sns.heatmap(
            profiles_norm.astype(float),
            annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
            linewidths=0.5, cbar_kws={"label": "Normalised Value (0=Low, 1=High)"},
        )
        ax.set_title(
            f"Plot S5: Customer Segment Profiles - {model_name} | Normalised average feature values per cluster",
            fontsize=13
        )
        ax.set_ylabel("Segment", fontsize=11)
        ax.set_xlabel("Feature", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=10)
        fig.tight_layout()
        fig.savefig(GRAPHS_DIR / "seg_cluster_profiles.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info("Plot S5 saved: seg_cluster_profiles.png")
        return pd.DataFrame(arr.astype(float), index=row_names, columns=col_names)
    except Exception as e:
        logger.error(f"plot_cluster_profiles failed: {e}", exc_info=True)
        plt.close("all")
        return pd.DataFrame()


def plot_segment_clv_heatmap(
    labels: np.ndarray,
    dollar_preds: np.ndarray,
    dollar_actual: np.ndarray,
) -> pd.DataFrame:
    """
    Generates Plot S6: Segment × CLV tier heatmap.

    This is the most business-valuable output of the combined CLV + segmentation
    pipeline. It answers: "Which customer segments drive the most future revenue?"

    Rows = unsupervised clusters (from K-Means)
    Cols = CLV tiers (from P1 CLV predictions)
    Values = percentage of customers in each cell

    Example insight:
        Cluster 1 → 80% Whale tier = "These are your VIP retention targets"
        Cluster 3 → 90% Churned tier = "Don't waste retention budget here"

    Parameters
    ----------
    labels        : K-Means cluster assignments
    dollar_preds  : CLV predicted dollar values (from P1 champion model)
    dollar_actual : actual dollar values (y_test_raw)
    """
    def _get_clv_tier(dollar):
        if dollar >= CLV_TIER_THRESHOLDS['Whale']:
            return '🐋 Whale'
        elif dollar >= CLV_TIER_THRESHOLDS['Mid']:
            return '💰 Mid'
        elif dollar >= CLV_TIER_THRESHOLDS['Low']:
            return '📉 Low'
        else:
            return '💤 Churned'

    clv_tiers = [_get_clv_tier(d) for d in dollar_preds]
    tier_order = ['🐋 Whale', '💰 Mid', '📉 Low', '💤 Churned']

    df = pd.DataFrame({
        'Cluster': [SEGMENT_LABELS.get(l, f'Cluster {l}') for l in labels],
        'CLV_Tier': clv_tiers,
        'CLV_Predicted': dollar_preds,
        'CLV_Actual': dollar_actual,
    })

    # Percentage heatmap
    crosstab = pd.crosstab(df['Cluster'], df['CLV_Tier'], normalize='index') * 100
    # Reorder columns
    crosstab = crosstab.reindex(columns=[t for t in tier_order if t in crosstab.columns])

    # Avg predicted CLV per cluster
    avg_clv = df.groupby('Cluster')['CLV_Predicted'].mean().round(0)

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(crosstab) * 0.9)))

    # Left: % distribution heatmap
    sns.heatmap(
        crosstab,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        ax=axes[0],
        linewidths=0.5,
        cbar_kws={'label': '% of Customers in Segment'},
        vmin=0, vmax=100,
    )
    axes[0].set_title(
        'Plot S6: Segment × CLV Tier Distribution (%)\n'
        'Row = Customer Segment, Col = CLV Tier',
        fontsize=13
    )
    axes[0].set_ylabel('Customer Segment (K-Means)', fontsize=11)
    axes[0].set_xlabel('CLV Tier (P1 Predictions)', fontsize=11)
    plt.sca(axes[0])
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)

    # Right: avg predicted CLV per segment (bar chart)
    avg_clv_sorted = avg_clv.sort_values(ascending=True)
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(avg_clv_sorted))]
    axes[1].barh(avg_clv_sorted.index, avg_clv_sorted.values, color=colors, alpha=0.85)
    axes[1].set_xlabel('Avg Predicted 90-Day CLV ($)', fontsize=12)
    axes[1].set_title('Avg Predicted CLV by Segment', fontsize=13)
    axes[1].grid(axis='x', alpha=0.3)
    for i, (name, val) in enumerate(avg_clv_sorted.items()):
        axes[1].text(val + 5, i, f'${val:,.0f}', va='center', fontsize=10)

    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / 'seg_clv_heatmap.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info("Plot S6 saved: seg_clv_heatmap.png")

    return df


def plot_rfm_3d(
    X_original: pd.DataFrame,
    labels: np.ndarray,
) -> None:
    """
    Generates Plot S8: 3D RFM scatter plot coloured by cluster.

    Provides the most intuitive visual of cluster separation in the core
    RFM space. Axes: Recency (x), Frequency (y), Monetary (z).
    """
    if not all(c in X_original.columns for c in RFM_COLS):
        logger.warning("RFM 3D plot skipped - RFM columns not available.")
        return

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')

    unique_labels = sorted(set(labels))
    for i, label in enumerate(unique_labels):
        mask  = labels == label
        color = '#888888' if label == -1 else CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        name  = 'Noise' if label == -1 else SEGMENT_LABELS.get(label, f'Cluster {label}')
        ax.scatter(
            X_original.loc[mask, 'Recency'],
            X_original.loc[mask, 'Frequency'],
            X_original.loc[mask, 'Monetary'],
            c=color, label=name, alpha=0.6, s=20,
        )

    ax.set_xlabel('Recency (days)', fontsize=10)
    ax.set_ylabel('Frequency (invoices)', fontsize=10)
    ax.set_zlabel('Monetary (avg order $)', fontsize=10)
    ax.set_title('Plot S8: RFM 3D Scatter - Customer Segments', fontsize=13)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / 'seg_rfm_3d.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info("Plot S8 saved: seg_rfm_3d.png")


def plot_dbscan_map(
    embedding: np.ndarray,
    dbscan_labels: np.ndarray,
) -> None:
    """
    Generates Plot S10: DBSCAN cluster map with noise points highlighted.
    Uses UMAP or t-SNE embedding for 2D positioning.
    """
    if embedding is None:
        logger.info("Plot S10 skipped - no 2D embedding available.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    unique = sorted(set(dbscan_labels))

    for label in unique:
        mask  = dbscan_labels == label
        if label == -1:
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c='red', marker='x', s=30, alpha=0.8,
                       label=f'Noise / Outlier (n={mask.sum()})', zorder=5)
        else:
            color = CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=color, s=20, alpha=0.6,
                       label=f'Cluster {label} (n={mask.sum()})')

    ax.set_title(
        'Plot S10: DBSCAN Clusters - Noise Points Highlighted in Red\n'
        'Red × = customers that do not fit any dense neighborhood',
        fontsize=13
    )
    ax.set_xlabel('Embedding Dim 1', fontsize=11)
    ax.set_ylabel('Embedding Dim 2', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / 'seg_dbscan_map.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info("Plot S10 saved: seg_dbscan_map.png")


# ===========================================================================
# Customer Assignment & Export
# ===========================================================================

def assign_segments(
    X: pd.DataFrame,
    kmeans_model: KMeans,
    scaler: RobustScaler,
    cols_used: list,
    pca=None,
    dollar_preds: Optional[np.ndarray] = None,
    customer_ids: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Assigns segment labels to a DataFrame of customers.

    Can be called on new customers at inference time - scaler and kmeans
    are pre-fitted on training data.

    Parameters
    ----------
    X             : customer feature DataFrame
    kmeans_model  : fitted KMeans
    scaler        : fitted RobustScaler
    cols_used     : feature columns used during training
    dollar_preds  : optional CLV predictions to append
    customer_ids  : optional customer ID series to include

    Returns
    -------
    result_df : DataFrame with Cluster, Segment_Name, CLV_Predicted columns
    """
    X_scaled, _, _ = _prepare_features(X, scaler=scaler, fit_scaler=False)
    # Apply PCA if provided (KMeans was trained on PCA-reduced features)
    if pca is not None:
        X_for_kmeans = pca.transform(X_scaled)
    else:
        X_for_kmeans = X_scaled
    labels = kmeans_model.predict(X_for_kmeans)
    result = X[cols_used].copy()

    if customer_ids is not None:
        result.insert(0, 'Customer_ID', customer_ids.values)

    result['Cluster']      = labels
    result['Segment_Name'] = [SEGMENT_LABELS.get(l, f'Cluster {l}') for l in labels]

    if dollar_preds is not None:
        result['CLV_Predicted_90d'] = np.round(dollar_preds, 2)

        def _tier(d):
            if d >= CLV_TIER_THRESHOLDS['Whale']:  return '🐋 Whale'
            elif d >= CLV_TIER_THRESHOLDS['Mid']:   return '💰 Mid'
            elif d >= CLV_TIER_THRESHOLDS['Low']:   return '📉 Low'
            else:                                   return '💤 Churned'

        result['CLV_Tier'] = [_tier(d) for d in dollar_preds]

    return result


def save_segmentation_results(
    customer_df: pd.DataFrame,
    profiles: pd.DataFrame,
    metrics: dict,
) -> None:
    """
    Saves all segmentation outputs to GRAPHS_DIR and MODELS_DIR.

    Files saved:
    - customer_segments.csv     - customer-level assignments
    - segment_profiles.csv      - cluster-level profiles
    - segmentation_metrics.csv  - model evaluation scores
    """
    customer_df.to_csv(GRAPHS_DIR / 'customer_segments.csv', index=False)
    profiles.to_csv(GRAPHS_DIR / 'segment_profiles.csv')

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(GRAPHS_DIR / 'segmentation_metrics.csv', index=False)

    logger.info(
        f"Segmentation results saved:\n"
        f"  customer_segments.csv   ({len(customer_df):,} customers)\n"
        f"  segment_profiles.csv    ({len(profiles)} clusters)\n"
        f"  segmentation_metrics.csv"
    )


# ===========================================================================
# Model Serialization
# ===========================================================================

def save_segmentation_bundle(
    kmeans:     KMeans,
    scaler:     RobustScaler,
    pca:        PCA,
    cols_used:  list,
    optimal_k:  int,
    metrics:    dict,
) -> None:
    """
    Serializes the segmentation bundle for Streamlit app and API inference.

    Bundle contents:
    - kmeans       : fitted K-Means model
    - scaler       : fitted RobustScaler
    - pca          : fitted PCA (for visualization)
    - cols_used    : feature columns
    - optimal_k    : number of clusters
    - metrics      : evaluation scores
    - segment_labels: human-readable cluster names
    - version      : model version string
    - timestamp    : training timestamp
    """
    bundle = {
        'kmeans':          kmeans,
        'scaler':          scaler,
        'pca':             pca,
        'cols_used':       cols_used,
        'optimal_k':       optimal_k,
        'metrics':         metrics,
        'segment_labels':  SEGMENT_LABELS,
        'clv_thresholds':  CLV_TIER_THRESHOLDS,
        'version':         '1.0.0',
        'timestamp':       pd.Timestamp.now().isoformat(),
    }
    output_path = MODELS_DIR / 'segmentation_bundle.pkl'
    joblib.dump(bundle, output_path)
    logger.info(f"Segmentation bundle saved: {output_path}")


# ===========================================================================
# Master Pipeline Function
# ===========================================================================

def run_segmentation_pipeline(
    X_train:       pd.DataFrame,
    X_test:        pd.DataFrame,
    y_test_raw:    pd.Series,
    dollar_preds:  Optional[np.ndarray]  = None,
    customer_ids:  Optional[pd.Series]   = None,
    feature_set:   str                   = 'extended',
    tune_dbscan:   bool                  = True,
    run_tsne:      bool                  = True,
) -> dict:
    """
    Master pipeline - runs all segmentation models end to end.

    Workflow
    --------
    - Prepare + scale features (RobustScaler)
    - PCA variance analysis + dimensionality reduction
    - K-Means find optimal k, fit primary segmentation
    - DBSCAN density-based clustering + outlier detection
    - GMM soft probabilistic assignments
    - Hierarchical agglomerative clustering + dendrogram
    - UMAP 2D non-linear embedding
    - t-SNE 2D alternative embedding
    - Cluster profiles feature heatmap
    - Segment x CLV heatmap business linkage to P1
    - RFM 3D scatter
    - DBSCAN noise map
    - Save all results + bundle

    Parameters
    ----------
    X_train       : training features (FEATURE_COLS)
    X_test        : test features (FEATURE_COLS) - segmented customers
    y_test_raw    : actual dollar spend (test set)
    dollar_preds  : CLV predictions from P1 champion model (optional)
    customer_ids  : customer ID series for output CSV (optional)
    feature_set   : 'rfm' or 'extended' feature set
    tune_dbscan   : use Optuna to tune DBSCAN params (requires optuna)
    run_tsne      : compute t-SNE embedding (slow on large datasets)

    Returns
    -------
    dict with keys: kmeans, labels, profiles, customer_df, metrics, embedding
    """
    logger.info("=" * 70)
    logger.info("[SEG] Starting Customer Segmentation Pipeline v1.0.0")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Feature preparation
    # ------------------------------------------------------------------
    logger.info("[SEG 1/13] Preparing features...")
    X_scaled_train, scaler, cols_used = _prepare_features(
        X_train, feature_set=feature_set, fit_scaler=True
    )
    X_scaled_test, _, _ = _prepare_features(
        X_test, feature_set=feature_set,
        scaler=scaler, fit_scaler=False
    )

    # Combine train + test for clustering (unsupervised - use all data)
    X_all     = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    X_scaled  = np.vstack([X_scaled_train, X_scaled_test])
    n_train   = len(X_train)

    logger.info(f"Feature set: {feature_set} | Features: {cols_used}")
    logger.info(f"Total customers: {len(X_all):,} ({n_train:,} train + {len(X_test):,} test)")

    # ------------------------------------------------------------------
    # Step 2: PCA
    # ------------------------------------------------------------------
    logger.info("[SEG 2/13] PCA - variance analysis...")
    pca, X_pca = fit_pca(X_scaled, variance_threshold=0.95, save_plot=True)

    # ------------------------------------------------------------------
    # Step 3: K-Means - optimal k + fit
    # ------------------------------------------------------------------
    logger.info("[SEG 3/13] K-Means - finding optimal k...")
    optimal_k = get_optimal_k(X_pca, save_plots=True)

    logger.info(f"[SEG 4/13] K-Means - fitting with k={optimal_k}...")
    kmeans, km_labels, profiles = fit_kmeans(X_pca, optimal_k, X_all, cols_used)

    km_sil = silhouette_score(X_pca, km_labels)
    km_db  = davies_bouldin_score(X_pca, km_labels)
    km_ch  = calinski_harabasz_score(X_pca, km_labels)

    # ------------------------------------------------------------------
    # Step 5: DBSCAN
    # ------------------------------------------------------------------
    logger.info("[SEG 5/13] DBSCAN...")
    if tune_dbscan:
        best_eps, best_min_smp = tune_dbscan_optuna(X_pca, n_trials=30)
    else:
        best_eps, best_min_smp = DBSCAN_EPS_DEFAULT, DBSCAN_MIN_SAMPLES

    dbscan, db_labels, db_metrics = fit_dbscan(X_pca, eps=best_eps, min_samples=best_min_smp)

    # ------------------------------------------------------------------
    # Step 6: GMM
    # ------------------------------------------------------------------
    logger.info("[SEG 6/13] GMM...")
    gmm, gmm_labels, gmm_proba = fit_gmm(X_pca, optimal_k, X_all, cols_used)

    # ------------------------------------------------------------------
    # Step 7: Hierarchical
    # ------------------------------------------------------------------
    logger.info("[SEG 7/13] Hierarchical clustering...")
    hier_model, hier_labels = fit_hierarchical(X_pca, optimal_k, save_dendrogram=True)

    # ------------------------------------------------------------------
    # Step 8: UMAP
    # ------------------------------------------------------------------
    logger.info("[SEG 8/13] UMAP embedding...")
    umap_embedding = compute_umap_embedding(X_pca, km_labels, title_suffix="K-Means")

    # ------------------------------------------------------------------
    # Step 9: t-SNE
    # ------------------------------------------------------------------
    if run_tsne:
        logger.info("[SEG 9/13] t-SNE embedding...")
        tsne_embedding = compute_tsne_embedding(X_pca, km_labels, title_suffix="K-Means")
    else:
        tsne_embedding = None
        logger.info("[SEG 9/13] t-SNE skipped (run_tsne=False).")

    # ------------------------------------------------------------------
    # Step 10: Cluster profiles heatmap
    # ------------------------------------------------------------------
    logger.info("[SEG 10/13] Cluster profiles...")
    plot_cluster_profiles(X_all, km_labels, cols_used, model_name="K-Means")

    # ------------------------------------------------------------------
    # Step 11: Segment × CLV heatmap (only if CLV predictions available)
    # ------------------------------------------------------------------
    logger.info("[SEG 11/13] Segment × CLV heatmap...")
    if dollar_preds is not None:
        # Use test-set labels only for CLV linkage
        test_labels   = km_labels[n_train:]
        dollar_actual = y_test_raw.values if hasattr(y_test_raw, 'values') else np.array(y_test_raw)
        customer_df   = plot_segment_clv_heatmap(test_labels, dollar_preds, dollar_actual)
    else:
        logger.info("CLV predictions not provided - Segment×CLV heatmap skipped.")
        customer_df = None

    # ------------------------------------------------------------------
    # Step 12: RFM 3D + DBSCAN map
    # ------------------------------------------------------------------
    logger.info("[SEG 12/13] RFM 3D scatter + DBSCAN map...")
    plot_rfm_3d(X_all, km_labels)

    embedding_for_dbscan = umap_embedding if umap_embedding is not None else tsne_embedding
    if embedding_for_dbscan is not None:
        plot_dbscan_map(embedding_for_dbscan, db_labels)

    # ------------------------------------------------------------------
    # Step 13: Save results + bundle
    # ------------------------------------------------------------------
    logger.info("[SEG 13/13] Saving results and bundle...")

    all_metrics = {
        'model':                'K-Means',
        'optimal_k':            optimal_k,
        'silhouette':           km_sil,
        'davies_bouldin':       km_db,
        'calinski_harabasz':    km_ch,
        'dbscan_n_clusters':    db_metrics.get('n_clusters', 0),
        'dbscan_noise_pct':     db_metrics.get('noise_pct', 0),
        'dbscan_silhouette':    db_metrics.get('silhouette', None),
        'gmm_silhouette':       silhouette_score(X_pca, gmm_labels),
        'hier_silhouette':      silhouette_score(X_pca, hier_labels),
        'pca_n_components':     pca.n_components_,
        'pca_variance_explained': float(pca.explained_variance_ratio_.sum()),
    }

    if customer_df is not None:
        assignment_df = assign_segments(
            X_test, kmeans, scaler, cols_used,
            pca=pca,
            dollar_preds=dollar_preds,
            customer_ids=customer_ids,
        )
    else:
        assignment_df = None

    save_segmentation_bundle(kmeans, scaler, pca, cols_used, optimal_k, all_metrics)

    # Summary print
    print(f"\n{'='*65}")
    print(f"  SEGMENTATION PIPELINE COMPLETE")
    print(f"{'='*65}")
    print(f"  Champion: K-Means (k={optimal_k})")
    print(f"  Silhouette Score     : {km_sil:.4f}  (higher = better, max 1.0)")
    print(f"  Davies-Bouldin Index : {km_db:.4f}  (lower = better)")
    print(f"  Calinski-Harabasz    : {km_ch:.0f}  (higher = better)")
    print(f"  DBSCAN clusters found: {db_metrics.get('n_clusters', 'N/A')}")
    print(f"  DBSCAN noise points  : {db_metrics.get('noise_pct', 0):.1f}% of customers")
    print(f"  PCA components used  : {pca.n_components_} "
          f"({all_metrics['pca_variance_explained']*100:.1f}% variance)")
    print(f"{'='*65}")
    print(f"\n  Plots saved to: {GRAPHS_DIR}")
    print(f"  Bundle saved to: {MODELS_DIR / 'segmentation_bundle.pkl'}")

    logger.info("Segmentation pipeline complete.")

    return {
        'kmeans':       kmeans,
        'scaler':       scaler,
        'pca':          pca,
        'labels':       km_labels,
        'profiles':     profiles,
        'customer_df':  assignment_df,
        'metrics':      all_metrics,
        'embedding':    umap_embedding,
        'dbscan_labels': db_labels,
        'gmm_labels':   gmm_labels,
        'hier_labels':  hier_labels,
        'cols_used':    cols_used,
        'optimal_k':    optimal_k,
    }