"""
Customer Anomaly Detection Module - v1.0.0
==========================================
Identifies anomalous customers using two complementary approaches:

1. Isolation Forest  - tree-based anomaly scoring, excels at high-dimensional
                       tabular data, no distributional assumptions, fast O(n log n)

2. Autoencoder       - PyTorch deep learning, learns a compressed representation of
                       "normal" customers; anomalies have high reconstruction error.
                       Falls back gracefully if torch is not installed.

The two scores are ensembled into a combined AnomalyScore (0-1, higher = more anomalous).

Business Outputs
----------------
- High-return customers flagged (Return_Rate > threshold + anomaly score)
- Suspicious transaction patterns (whale monetary + short recency + high return)
- Per-customer anomaly report with interpretable feature contributions

Diagnostic Plots Saved
-----------------------
Plot A1: anomaly_score_distribution.png   - score histogram + threshold line
Plot A2: anomaly_feature_importance.png   - SHAP-style IF feature contributions
Plot A3: anomaly_umap.png                 - UMAP coloured by anomaly score
Plot A4: anomaly_reconstruction_error.png - Autoencoder epoch loss curve
Plot A5: anomaly_top_customers.png        - top 20 anomalous customer profiles
Plot A6: anomaly_return_flags.png         - return rate vs anomaly score scatter

Business CSVs Saved
--------------------
- anomaly_scores.csv        - per-customer anomaly scores + flags
- anomaly_top50.csv         - top 50 most anomalous customers
- high_return_customers.csv - customers flagged for abnormal return behaviour

Usage
-----
    from src.anomaly import run_anomaly_pipeline

    anomaly_results = run_anomaly_pipeline(
        X_train      = X_train,
        X_test       = X_test,
        y_test_raw   = y_test_raw,
        dollar_preds = dollar_preds,
        customer_ids = None,
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
from datetime import datetime
from typing import Optional, Dict, Tuple, List

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score

from src.config import GRAPHS_DIR, MODELS_DIR, RANDOM_SEED, FEATURE_COLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PyTorch - graceful degradation
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    logger.info("PyTorch available - Autoencoder enabled.")
except ImportError:
    TORCH_AVAILABLE = False
    logger.info(
        "torch not installed - Autoencoder skipped. "
        "Run: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    )

# ---------------------------------------------------------------------------
# Optional UMAP
# ---------------------------------------------------------------------------
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.info("umap-learn not installed - UMAP anomaly plot skipped.")

# ---------------------------------------------------------------------------
# Optional SHAP for IF feature importance
# ---------------------------------------------------------------------------
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.info("shap not installed - using permutation importance instead.")


# ===========================================================================
# Constants
# ===========================================================================

# Features used for anomaly detection (covers all major behavioral dimensions)
ANOMALY_FEATURE_COLS = [
    'Recency',
    'Frequency',
    'Monetary',
    'Return_Rate',
    'Avg_Basket_Size',
    'Max_Single_Order',
    'Monetary_Percentile',
    'Purchase_Rate',
    'Unique_Products',
    'Days_Since_Purchase',
    'Interpurchase_Std',
    'Revenue_Per_Day',
]

# Isolation Forest hyperparameters
IF_CONTAMINATION     = 0.05     # Expected fraction of anomalies in dataset (5%)
IF_N_ESTIMATORS      = 200      # More trees = more stable scores
IF_MAX_SAMPLES       = 'auto'   # 256 samples per tree (IF paper default)

# Autoencoder hyperparameters
AE_HIDDEN_DIMS       = [64, 32, 16, 32, 64]   # Symmetric bottleneck at 16
AE_BOTTLENECK_DIM    = 8
AE_LEARNING_RATE     = 1e-3
AE_EPOCHS            = 100
AE_BATCH_SIZE        = 64
AE_DROPOUT           = 0.1
AE_WEIGHT_DECAY      = 1e-5

# Ensemble weights
WEIGHT_IF            = 0.45    # Isolation Forest weight
WEIGHT_AE            = 0.55    # Autoencoder weight (slightly higher - richer signal)

# Business flag thresholds
ANOMALY_SCORE_THRESHOLD  = 0.50   # Combined score above this → flagged anomaly
RETURN_RATE_THRESHOLD    = 0.30   # Return_Rate above this is suspicious
HIGH_RETURN_SCORE_MIN    = 0.40   # Must also have anomaly score > this

# Plot style
STYLE = 'dark_background'
ACCENT = '#5C4DB1'
DANGER = '#E63946'
SAFE   = '#2A9D8F'
WARN   = '#F4A261'


# ===========================================================================
# Autoencoder (PyTorch)
# ===========================================================================

if TORCH_AVAILABLE:
    class CustomerAutoencoder(nn.Module):
        """
        Symmetric autoencoder for customer feature reconstruction.

        Architecture:
            Encoder: input → 64 → 32 → 16 → bottleneck(8)
            Decoder: bottleneck(8) → 16 → 32 → 64 → input

        Anomaly score = per-sample MSE reconstruction error.
        High error = the model failed to reconstruct → unusual pattern.
        """

        def __init__(self, input_dim: int):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(AE_DROPOUT),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(AE_DROPOUT),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, AE_BOTTLENECK_DIM),
            )

            self.decoder = nn.Sequential(
                nn.Linear(AE_BOTTLENECK_DIM, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(AE_DROPOUT),
                nn.Linear(32, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(AE_DROPOUT),
                nn.Linear(64, input_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.encoder(x))

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

        def reconstruction_error(self, x: torch.Tensor) -> np.ndarray:
            """Per-sample MSE reconstruction error."""
            self.eval()
            # Ensure input tensor is on the same device as model weights
            device = next(self.parameters()).device
            x = x.to(device)
            with torch.no_grad():
                x_hat = self.forward(x)
                errors = torch.mean((x - x_hat) ** 2, dim=1)
            return errors.cpu().numpy()


# ===========================================================================
# Feature Preparation
# ===========================================================================

def _prepare_anomaly_features(
    X: pd.DataFrame,
    scaler: Optional[RobustScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, RobustScaler, List[str]]:
    """
    Selects and scales features for anomaly detection.

    Uses RobustScaler: robust to whale customers distorting scale,
    preserves outlier structure while centering on median.

    Parameters
    ----------
    X          : input DataFrame with FEATURE_COLS
    scaler     : pre-fitted scaler (pass for transform-only mode)
    fit_scaler : True → fit + transform; False → transform only

    Returns
    -------
    X_scaled   : numpy array, shape (n, len(ANOMALY_FEATURE_COLS))
    scaler     : fitted RobustScaler
    cols_used  : column names actually used
    """
    cols_available = [c for c in ANOMALY_FEATURE_COLS if c in X.columns]
    missing = set(ANOMALY_FEATURE_COLS) - set(cols_available)
    if missing:
        logger.warning(f"Anomaly features missing from input: {missing}")

    X_sub = X[cols_available].copy()
    X_sub = X_sub.fillna(X_sub.median())

    if scaler is None:
        scaler = RobustScaler()

    if fit_scaler:
        X_scaled = scaler.fit_transform(X_sub)
    else:
        X_scaled = scaler.transform(X_sub)

    return X_scaled.astype(np.float32), scaler, cols_available


# ===========================================================================
# Isolation Forest
# ===========================================================================

def fit_isolation_forest(
    X_scaled: np.ndarray,
    contamination: float = IF_CONTAMINATION,
) -> Tuple[IsolationForest, np.ndarray, np.ndarray]:
    """
    Fits Isolation Forest and returns per-sample anomaly scores.

    Isolation Forest isolates anomalies by randomly partitioning features.
    Anomalies require fewer splits → shorter path lengths → high anomaly score.

    Returns
    -------
    model       : fitted IsolationForest
    if_scores   : raw IF scores in [-1, 1] (negative = more anomalous)
    if_norm     : normalised scores in [0, 1] (1 = most anomalous)
    """
    logger.info(f"Fitting Isolation Forest | n={len(X_scaled):,} | contamination={contamination}")

    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        max_samples=IF_MAX_SAMPLES,
        contamination=contamination,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # decision_function: higher = more normal → flip sign → higher = more anomalous
    raw_scores = model.decision_function(X_scaled)
    if_scores  = -raw_scores  # flip: high = anomalous

    # Normalise to [0, 1] using MinMaxScaler
    norm = MinMaxScaler()
    if_norm = norm.fit_transform(if_scores.reshape(-1, 1)).ravel()

    n_flagged = (model.predict(X_scaled) == -1).sum()
    logger.info(f"Isolation Forest: {n_flagged:,} anomalies flagged ({n_flagged/len(X_scaled)*100:.1f}%)")

    return model, if_scores, if_norm


# ===========================================================================
# Autoencoder Training & Scoring
# ===========================================================================

def train_autoencoder(
    X_scaled: np.ndarray,
) -> Tuple[object, np.ndarray, List[float]]:
    """
    Trains a PyTorch Autoencoder and returns per-sample reconstruction errors.

    Only runs if TORCH_AVAILABLE is True. Returns (None, zeros, []) otherwise.

    Returns
    -------
    model      : trained CustomerAutoencoder (or None)
    ae_norm    : normalised reconstruction error [0, 1] (1 = most anomalous)
    loss_curve : per-epoch training loss
    """
    if not TORCH_AVAILABLE:
        logger.info("PyTorch not available - Autoencoder skipped.")
        return None, np.zeros(len(X_scaled), dtype=np.float32), []

    logger.info(
        f"Training Autoencoder | n={len(X_scaled):,} | "
        f"input_dim={X_scaled.shape[1]} | epochs={AE_EPOCHS}"
    )

    torch.manual_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Autoencoder device: {device}")

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    dataset  = TensorDataset(X_tensor)
    loader   = DataLoader(dataset, batch_size=AE_BATCH_SIZE, shuffle=True)

    model = CustomerAutoencoder(input_dim=X_scaled.shape[1]).to(device)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=AE_LEARNING_RATE,
        weight_decay=AE_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=AE_EPOCHS, eta_min=1e-5
    )
    criterion = nn.MSELoss()

    loss_curve = []
    model.train()
    for epoch in range(AE_EPOCHS):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimiser.zero_grad()
            recon  = model(batch)
            loss   = criterion(recon, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += loss.item() * len(batch)

        epoch_loss /= len(X_scaled)
        loss_curve.append(epoch_loss)
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1:>3}/{AE_EPOCHS} | Loss: {epoch_loss:.6f}")

    # Compute per-sample reconstruction errors on train set
    # Return raw errors — caller normalises after scoring full dataset
    model.eval()
    errors  = model.reconstruction_error(X_tensor)

    logger.info(
        f"Autoencoder trained | Final loss: {loss_curve[-1]:.6f} | "
        f"Train error range: [{errors.min():.6f}, {errors.max():.6f}]"
    )

    return model, errors, loss_curve


# ===========================================================================
# Ensemble Scoring
# ===========================================================================

def compute_ensemble_score(
    if_norm: np.ndarray,
    ae_norm: np.ndarray,
    ae_available: bool = True,
) -> np.ndarray:
    """
    Weighted average of Isolation Forest and Autoencoder scores.

    If AE is unavailable, full weight falls to Isolation Forest.

    Returns
    -------
    combined : np.ndarray [0, 1], higher = more anomalous
    """
    if ae_available and TORCH_AVAILABLE:
        combined = WEIGHT_IF * if_norm + WEIGHT_AE * ae_norm
    else:
        combined = if_norm.copy()

    logger.info(
        f"Ensemble score | Mean: {combined.mean():.4f} | "
        f"Max: {combined.max():.4f} | "
        f"Flagged (>{ANOMALY_SCORE_THRESHOLD}): "
        f"{(combined > ANOMALY_SCORE_THRESHOLD).sum():,}"
    )
    return combined


# ===========================================================================
# Business Flag Logic
# ===========================================================================

def build_anomaly_dataframe(
    X_all: pd.DataFrame,
    combined_score: np.ndarray,
    if_norm: np.ndarray,
    ae_norm: np.ndarray,
    cols_used: List[str],
    customer_ids: Optional[pd.Series] = None,
    dollar_preds: Optional[np.ndarray] = None,
    y_test_raw: Optional[pd.Series] = None,
    n_train: int = 0,
) -> pd.DataFrame:
    """
    Assembles per-customer anomaly DataFrame with business flags.

    Flags applied:
    - is_anomaly        : combined score > ANOMALY_SCORE_THRESHOLD
    - is_high_return    : Return_Rate > RETURN_RATE_THRESHOLD AND score > HIGH_RETURN_SCORE_MIN
    - is_whale_anomaly  : Monetary_Percentile > 90 AND is_anomaly
    - is_suspicious     : high return + anomaly + short recency (< 30 days)

    Returns
    -------
    df : DataFrame with one row per customer + all scores + flags
    """
    df = X_all.copy().reset_index(drop=True)

    df['IF_Score']       = if_norm
    df['AE_Score']       = ae_norm
    df['Anomaly_Score']  = combined_score
    df['is_anomaly']     = combined_score > ANOMALY_SCORE_THRESHOLD

    # High return flag
    if 'Return_Rate' in df.columns:
        df['is_high_return'] = (
            (df['Return_Rate'] > RETURN_RATE_THRESHOLD) &
            (combined_score > HIGH_RETURN_SCORE_MIN)
        )
    else:
        df['is_high_return'] = False

    # Whale anomaly flag (high value + anomalous)
    if 'Monetary_Percentile' in df.columns:
        df['is_whale_anomaly'] = (
            (df['Monetary_Percentile'] > 90) & df['is_anomaly']
        )
    else:
        df['is_whale_anomaly'] = False

    # Suspicious flag: high return + anomaly + recent activity
    if 'Recency' in df.columns:
        df['is_suspicious'] = (
            df['is_high_return'] & df['is_anomaly'] & (df['Recency'] < 30)
        )
    else:
        df['is_suspicious'] = False

    # Add CLV predictions for test-set rows
    if dollar_preds is not None:
        clv_col = np.full(len(df), np.nan)
        clv_col[n_train:] = dollar_preds
        df['Predicted_CLV'] = clv_col

    if y_test_raw is not None:
        actual_col = np.full(len(df), np.nan)
        actual_col[n_train:] = y_test_raw.values if hasattr(y_test_raw, 'values') else np.array(y_test_raw)
        df['Actual_CLV'] = actual_col

    # Customer ID
    if customer_ids is not None:
        id_all = np.full(len(df), np.nan, dtype=object)
        id_all[n_train:] = customer_ids.values if hasattr(customer_ids, 'values') else np.array(customer_ids)
        df.insert(0, 'CustomerID', id_all)

    return df


# ===========================================================================
# Isolation Forest Feature Importance
# ===========================================================================

def compute_if_feature_importance(
    model: IsolationForest,
    X_scaled: np.ndarray,
    cols_used: List[str],
    X_all_df: pd.DataFrame,
) -> pd.Series:
    """
    Estimates feature importance for Isolation Forest via permutation.

    For each feature, we measure how much the mean anomaly score changes
    when the feature is randomly shuffled. Larger change = more important.

    Returns
    -------
    importances : pd.Series indexed by feature name, sorted descending
    """
    if SHAP_AVAILABLE:
        # SHAP TreeExplainer for Isolation Forest
        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            importance  = np.abs(shap_values).mean(axis=0)
            return pd.Series(importance, index=cols_used).sort_values(ascending=False)
        except Exception as e:
            logger.warning(f"SHAP failed: {e} - falling back to permutation importance.")

    # Permutation importance fallback
    base_scores = -model.decision_function(X_scaled)
    importances = {}
    rng = np.random.RandomState(RANDOM_SEED)

    for i, col in enumerate(cols_used):
        X_perm = X_scaled.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        perm_scores = -model.decision_function(X_perm)
        importances[col] = np.abs(perm_scores.mean() - base_scores.mean())

    return pd.Series(importances).sort_values(ascending=False)


# ===========================================================================
# Diagnostic Plots
# ===========================================================================

def _savefig(name: str) -> None:
    """Save current figure to GRAPHS_DIR."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAPHS_DIR / name
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0D0D0D')
    plt.close()
    logger.info(f"Saved: {path}")


def plot_score_distribution(
    combined_score: np.ndarray,
    if_norm: np.ndarray,
    ae_norm: Optional[np.ndarray] = None,
) -> None:
    """Plot A1: Anomaly score distributions + threshold line."""
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Customer Anomaly Score Distribution', fontsize=14,
                     color='white', fontweight='bold', y=1.02)

        # Left: combined score histogram
        ax = axes[0]
        ax.hist(combined_score, bins=50, color=ACCENT, alpha=0.8, edgecolor='none')
        ax.axvline(ANOMALY_SCORE_THRESHOLD, color=DANGER, lw=2, ls='--',
                   label=f'Threshold ({ANOMALY_SCORE_THRESHOLD})')
        n_flagged = (combined_score > ANOMALY_SCORE_THRESHOLD).sum()
        ax.set_title(f'Combined Anomaly Score\n({n_flagged} customers flagged)', color='white')
        ax.set_xlabel('Anomaly Score', color='white')
        ax.set_ylabel('Count', color='white')
        ax.legend()
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444')

        # Right: component score comparison
        ax2 = axes[1]
        ax2.hist(if_norm, bins=50, color=SAFE,   alpha=0.6, label='Isolation Forest', edgecolor='none')
        if ae_norm is not None and TORCH_AVAILABLE:
            ax2.hist(ae_norm, bins=50, color=WARN, alpha=0.6, label='Autoencoder',       edgecolor='none')
        ax2.set_title('Score Components', color='white')
        ax2.set_xlabel('Score', color='white')
        ax2.set_ylabel('Count', color='white')
        ax2.legend()
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_color('#444')

        plt.tight_layout()
    _savefig('anomaly_score_distribution.png')


def plot_feature_importance(
    importances: pd.Series,
) -> None:
    """Plot A2: Anomaly feature importance bar chart."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = [DANGER if i < 3 else ACCENT for i in range(len(importances))]
        bars = ax.barh(importances.index[::-1], importances.values[::-1],
                       color=colors[::-1], edgecolor='none')

        ax.set_title('Feature Importance for Anomaly Detection\n(Permutation / SHAP)',
                     color='white', fontsize=13, fontweight='bold')
        ax.set_xlabel('Importance', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#333')

        # Value labels
        for bar, val in zip(bars, importances.values[::-1]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', color='white', fontsize=8)

        plt.tight_layout()
    _savefig('anomaly_feature_importance.png')


def plot_umap_anomaly(
    X_scaled: np.ndarray,
    combined_score: np.ndarray,
) -> None:
    """Plot A3: UMAP embedding coloured by anomaly score."""
    if not UMAP_AVAILABLE:
        logger.info("UMAP not available - anomaly UMAP plot skipped.")
        return

    try:
        reducer = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1,
            random_state=RANDOM_SEED, verbose=False,
        )
        embedding = reducer.fit_transform(X_scaled)
    except Exception as e:
        logger.warning(f"UMAP failed: {e}")
        return

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 8))

        sc = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=combined_score, cmap='RdYlGn_r',
            s=8, alpha=0.7, linewidths=0,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Anomaly Score', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # Highlight flagged anomalies
        flagged = combined_score > ANOMALY_SCORE_THRESHOLD
        ax.scatter(
            embedding[flagged, 0], embedding[flagged, 1],
            facecolors='none', edgecolors=DANGER, s=40, lw=0.8,
            label=f'Flagged anomalies ({flagged.sum():,})', zorder=5,
        )

        ax.set_title('UMAP - Customer Anomaly Map', color='white', fontsize=13, fontweight='bold')
        ax.set_xlabel('UMAP-1', color='white')
        ax.set_ylabel('UMAP-2', color='white')
        ax.tick_params(colors='white')
        ax.legend(labelcolor='white')
        for spine in ax.spines.values():
            spine.set_color('#333')

        plt.tight_layout()
    _savefig('anomaly_umap.png')


def plot_reconstruction_error(
    loss_curve: List[float],
) -> None:
    """Plot A4: Autoencoder training loss curve."""
    if not loss_curve:
        return

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(loss_curve, color=ACCENT, lw=2)
        ax.set_title('Autoencoder Training Loss (MSE)', color='white',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch', color='white')
        ax.set_ylabel('Reconstruction Loss', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#333')

        # Highlight convergence region (last 20%)
        conv_start = int(len(loss_curve) * 0.8)
        ax.axvspan(conv_start, len(loss_curve), alpha=0.1, color=SAFE,
                   label=f'Convergence zone (epoch {conv_start}+)')
        ax.legend(labelcolor='white')
        ax.text(len(loss_curve) * 0.02, loss_curve[0] * 0.95,
                f'Final loss: {loss_curve[-1]:.6f}', color='white', fontsize=9)

        plt.tight_layout()
    _savefig('anomaly_reconstruction_error.png')


def plot_top_anomalous_customers(
    anomaly_df: pd.DataFrame,
    cols_used: List[str],
    top_n: int = 20,
) -> None:
    """Plot A5: Heatmap of top-N most anomalous customers feature values."""
    top = anomaly_df.nlargest(top_n, 'Anomaly_Score')
    plot_cols = [c for c in cols_used if c in top.columns]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 8))

        data = top[plot_cols].copy()
        # Cast to numeric — object columns cause seaborn dtype error
        data = data.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float64)
        # Standardise each column for visual comparison
        data = (data - data.mean()) / (data.std() + 1e-8)

        sns.heatmap(
            data.values,
            xticklabels=plot_cols,
            yticklabels=[f"Rank {i+1}" for i in range(len(top))],
            cmap='RdYlGn_r', center=0, linewidths=0.3,
            ax=ax, cbar_kws={'label': 'Standardised Value'},
        )
        ax.set_title(f'Top {top_n} Most Anomalous Customers — Feature Profile',
                     color='white', fontsize=13, fontweight='bold', pad=12)
        ax.tick_params(colors='white', axis='x', rotation=45)
        ax.tick_params(colors='white', axis='y')
        ax.set_xlabel('Feature', color='white')
        ax.set_ylabel('Customer Rank', color='white')

        plt.tight_layout()
    _savefig('anomaly_top_customers.png')


def plot_return_flag_scatter(
    anomaly_df: pd.DataFrame,
) -> None:
    """Plot A6: Return Rate vs Anomaly Score scatter with flags highlighted."""
    if 'Return_Rate' not in anomaly_df.columns:
        return

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 7))

        # Normal customers
        normal = ~anomaly_df['is_high_return']
        ax.scatter(
            anomaly_df.loc[normal, 'Return_Rate'],
            anomaly_df.loc[normal, 'Anomaly_Score'],
            alpha=0.4, s=10, color=SAFE, label='Normal', linewidths=0,
        )

        # High-return flagged
        flagged = anomaly_df['is_high_return']
        if flagged.sum() > 0:
            ax.scatter(
                anomaly_df.loc[flagged, 'Return_Rate'],
                anomaly_df.loc[flagged, 'Anomaly_Score'],
                alpha=0.8, s=30, color=DANGER,
                label=f'High-Return Flagged ({flagged.sum():,})', linewidths=0,
            )

        # Suspicious flags
        susp = anomaly_df.get('is_suspicious', pd.Series(False, index=anomaly_df.index))
        if susp.sum() > 0:
            ax.scatter(
                anomaly_df.loc[susp, 'Return_Rate'],
                anomaly_df.loc[susp, 'Anomaly_Score'],
                alpha=1.0, s=60, marker='*', color='#FFD700',
                label=f'Suspicious ({susp.sum():,})', linewidths=0.5, edgecolors='white',
            )

        # Threshold lines
        ax.axhline(ANOMALY_SCORE_THRESHOLD, color=DANGER, ls='--', lw=1,
                   alpha=0.6, label=f'Anomaly threshold ({ANOMALY_SCORE_THRESHOLD})')
        ax.axvline(RETURN_RATE_THRESHOLD,   color=WARN,   ls='--', lw=1,
                   alpha=0.6, label=f'Return threshold ({RETURN_RATE_THRESHOLD})')

        ax.set_title('Return Rate vs Anomaly Score', color='white',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Return Rate', color='white')
        ax.set_ylabel('Anomaly Score', color='white')
        ax.tick_params(colors='white')
        ax.legend(labelcolor='white', fontsize=8)
        for spine in ax.spines.values():
            spine.set_color('#333')

        plt.tight_layout()
    _savefig('anomaly_return_flags.png')


# ===========================================================================
# Save Results
# ===========================================================================

def save_anomaly_results(
    anomaly_df: pd.DataFrame,
    if_model: IsolationForest,
    ae_model: Optional[object],
    scaler: RobustScaler,
    cols_used: List[str],
    metrics: Dict,
) -> None:
    """
    Saves all anomaly outputs to GRAPHS_DIR and MODELS_DIR.

    Files saved:
    - anomaly_scores.csv         : full per-customer scores
    - anomaly_top50.csv          : top 50 anomalies
    - high_return_customers.csv  : return-flagged customers
    - anomaly_bundle.pkl         : models + scaler + metadata
    """
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # CSVs
    output_cols = ['Anomaly_Score', 'IF_Score', 'AE_Score',
                   'is_anomaly', 'is_high_return', 'is_whale_anomaly', 'is_suspicious']
    if 'CustomerID' in anomaly_df.columns:
        output_cols = ['CustomerID'] + output_cols
    base_feature_cols = [c for c in cols_used if c in anomaly_df.columns]

    save_df = anomaly_df[output_cols + base_feature_cols].copy()
    save_df.to_csv(GRAPHS_DIR / 'anomaly_scores.csv', index=False)
    logger.info(f"Saved: {GRAPHS_DIR / 'anomaly_scores.csv'}")

    top50 = anomaly_df.nlargest(50, 'Anomaly_Score')[output_cols + base_feature_cols]
    top50.to_csv(GRAPHS_DIR / 'anomaly_top50.csv', index=False)
    logger.info(f"Saved: {GRAPHS_DIR / 'anomaly_top50.csv'}")

    high_return = anomaly_df[anomaly_df['is_high_return']][output_cols + base_feature_cols]
    high_return.to_csv(GRAPHS_DIR / 'high_return_customers.csv', index=False)
    logger.info(f"Saved: {GRAPHS_DIR / 'high_return_customers.csv'} ({len(high_return):,} rows)")

    # Bundle
    bundle = {
        'isolation_forest': if_model,
        'autoencoder':      ae_model,
        'scaler':           scaler,
        'cols_used':        cols_used,
        'thresholds': {
            'anomaly_score':   ANOMALY_SCORE_THRESHOLD,
            'return_rate':     RETURN_RATE_THRESHOLD,
            'high_return_min': HIGH_RETURN_SCORE_MIN,
        },
        'weights': {
            'isolation_forest': WEIGHT_IF,
            'autoencoder':      WEIGHT_AE,
        },
        'metrics':   metrics,
        'version':   '1.0.0',
        'timestamp': datetime.utcnow().isoformat(),
    }
    bundle_path = MODELS_DIR / 'anomaly_bundle.pkl'
    joblib.dump(bundle, bundle_path)
    logger.info(f"Saved: {bundle_path}")


# ===========================================================================
# Score Leaderboard Print
# ===========================================================================

def _print_leaderboard(metrics: Dict) -> None:
    sep = "=" * 75
    print(f"\n{sep}")
    print("  ANOMALY DETECTION LEADERBOARD")
    print(f"{sep}")
    header = f"  {'Model':<22} {'Mean Score':>12} {'Flagged':>10} {'% Flagged':>10}"
    print(header)
    print("-" * 75)

    for m in metrics.get('models', []):
        print(
            f"  {m['name']:<22} {m['mean_score']:>12.4f} "
            f"{m['n_flagged']:>10,} {m['pct_flagged']:>9.1f}%"
        )
    print(sep)

    print(f"\n  Business Flags Summary:")
    print(f"  Total customers analysed : {metrics.get('n_total', 'N/A'):,}")
    print(f"  Anomalies flagged        : {metrics.get('n_anomaly', 0):,}  "
          f"({metrics.get('pct_anomaly', 0):.1f}%)")
    print(f"  High-return customers    : {metrics.get('n_high_return', 0):,}")
    print(f"  Whale anomalies          : {metrics.get('n_whale_anomaly', 0):,}")
    print(f"  Suspicious customers     : {metrics.get('n_suspicious', 0):,}")
    print(sep)


# ===========================================================================
# assign_anomaly_score — inference for new/single customers
# ===========================================================================

def assign_anomaly_score(
    X: pd.DataFrame,
    bundle_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Assigns anomaly scores to new customer(s) using saved bundle.

    Parameters
    ----------
    X           : DataFrame with FEATURE_COLS (one or more rows)
    bundle_path : path to anomaly_bundle.pkl; defaults to MODELS_DIR/anomaly_bundle.pkl

    Returns
    -------
    df : input X + Anomaly_Score + IF_Score + AE_Score + business flags
    """
    if bundle_path is None:
        bundle_path = MODELS_DIR / 'anomaly_bundle.pkl'

    bundle   = joblib.load(bundle_path)
    if_model = bundle['isolation_forest']
    ae_model = bundle['autoencoder']
    scaler   = bundle['scaler']
    cols     = bundle['cols_used']

    X_scaled, _, _ = _prepare_anomaly_features(X, scaler=scaler, fit_scaler=False)

    # IF score
    raw    = -if_model.decision_function(X_scaled)
    if_s   = MinMaxScaler().fit_transform(raw.reshape(-1, 1)).ravel()

    # AE score
    if ae_model is not None and TORCH_AVAILABLE:
        device   = torch.device('cpu')
        ae_model = ae_model.to(device)
        tensor   = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        ae_raw   = ae_model.reconstruction_error(tensor)
        ae_s     = MinMaxScaler().fit_transform(ae_raw.reshape(-1, 1)).ravel()
        combined = WEIGHT_IF * if_s + WEIGHT_AE * ae_s
    else:
        ae_s     = np.zeros(len(X_scaled))
        combined = if_s

    out = X.copy().reset_index(drop=True)
    out['Anomaly_Score'] = combined
    out['IF_Score']      = if_s
    out['AE_Score']      = ae_s
    out['is_anomaly']    = combined > ANOMALY_SCORE_THRESHOLD

    if 'Return_Rate' in out.columns:
        out['is_high_return'] = (
            (out['Return_Rate'] > RETURN_RATE_THRESHOLD) &
            (combined > HIGH_RETURN_SCORE_MIN)
        )

    return out


# ===========================================================================
# Main Pipeline
# ===========================================================================

def run_anomaly_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test_raw: Optional[pd.Series] = None,
    dollar_preds: Optional[np.ndarray] = None,
    customer_ids: Optional[pd.Series] = None,
) -> Dict:
    """
    End-to-end anomaly detection pipeline.

    Workflow
    --------
    1.  Prepare features (RobustScaler on combined train+test)
    2.  Isolation Forest — fit + score all customers
    3.  Autoencoder     — train on train set, score all customers
    4.  Ensemble        — weighted combination of IF + AE scores
    5.  Business flags  — is_anomaly, is_high_return, is_whale_anomaly, is_suspicious
    6.  Diagnostic plots:
        - Score distribution (A1)
        - Feature importance (A2)
        - UMAP anomaly map (A3)
        - Reconstruction error curve (A4)
        - Top-20 anomaly profiles heatmap (A5)
        - Return rate vs score scatter (A6)
    7.  Save CSVs + bundle

    Parameters
    ----------
    X_train      : training features (FEATURE_COLS)
    X_test       : test features (FEATURE_COLS)
    y_test_raw   : actual dollar spend (test set)
    dollar_preds : CLV predictions from champion model (optional)
    customer_ids : customer ID series aligned to X_test (optional)

    Returns
    -------
    dict with keys: anomaly_df, if_model, ae_model, scaler, metrics
    """
    logger.info("=" * 70)
    logger.info("[ANOMALY] Starting Customer Anomaly Detection Pipeline v1.0.0")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Feature preparation
    # ------------------------------------------------------------------
    logger.info("[ANOMALY 1/7] Preparing features...")
    X_all     = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    n_train   = len(X_train)

    X_scaled, scaler, cols_used = _prepare_anomaly_features(X_all, fit_scaler=True)
    X_scaled_train = X_scaled[:n_train]

    logger.info(
        f"Features: {cols_used}\n"
        f"Total customers: {len(X_all):,} ({n_train:,} train + {len(X_test):,} test)"
    )

    # ------------------------------------------------------------------
    # Step 2: Isolation Forest
    # ------------------------------------------------------------------
    logger.info("[ANOMALY 2/7] Fitting Isolation Forest...")
    if_model, if_scores_raw, if_norm = fit_isolation_forest(X_scaled)

    # ------------------------------------------------------------------
    # Step 3: Autoencoder (train on train set, score all)
    # ------------------------------------------------------------------
    logger.info("[ANOMALY 3/7] Training Autoencoder...")
    ae_model, ae_norm_train, loss_curve = train_autoencoder(X_scaled_train)

    if TORCH_AVAILABLE and ae_model is not None:
        device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ae_model = ae_model.to(device)
        # Score the FULL dataset (train + test) and normalise across the full range
        # This is critical — normalising only on train produces near-zero test scores
        X_all_t  = torch.tensor(X_scaled, dtype=torch.float32)
        ae_raw   = ae_model.reconstruction_error(X_all_t)   # shape (n_all,)

        # Clip at 99th percentile before normalising — one extreme outlier
        # (error=1052 vs mean=0.44) would collapse all other scores to ~0
        # via MinMaxScaler. Clipping preserves relative differences.
        p99      = np.percentile(ae_raw, 99)
        ae_clip  = np.clip(ae_raw, 0, p99)
        norm_ae  = MinMaxScaler()
        ae_norm  = norm_ae.fit_transform(ae_clip.reshape(-1, 1)).ravel()
        logger.info(
            f"AE full-dataset errors | "
            f"raw range=[{ae_raw.min():.4f}, {ae_raw.max():.4f}] mean={ae_raw.mean():.4f} | "
            f"p99 clip={p99:.4f} | "
            f"normalised mean={ae_norm.mean():.4f} flagged={( ae_norm > 0.50).sum()}"
        )
        ae_avail = True
    else:
        ae_norm   = np.zeros(len(X_all))
        ae_avail  = False

    # ------------------------------------------------------------------
    # Step 4: Ensemble
    # ------------------------------------------------------------------
    logger.info("[ANOMALY 4/7] Computing ensemble scores...")
    combined = compute_ensemble_score(if_norm, ae_norm, ae_available=ae_avail)

    # ------------------------------------------------------------------
    # Step 5: Business flags
    # ------------------------------------------------------------------
    logger.info("[ANOMALY 5/7] Building anomaly report with business flags...")
    anomaly_df = build_anomaly_dataframe(
        X_all, combined, if_norm, ae_norm, cols_used,
        customer_ids=customer_ids,
        dollar_preds=dollar_preds,
        y_test_raw=y_test_raw,
        n_train=n_train,
    )

    # ------------------------------------------------------------------
    # Step 6: Feature importance
    # ------------------------------------------------------------------
    logger.info("[ANOMALY 6/7] Computing feature importance...")
    importances = compute_if_feature_importance(if_model, X_scaled, cols_used, X_all)

    # ------------------------------------------------------------------
    # Step 7: Plots + save
    # ------------------------------------------------------------------
    logger.info("[ANOMALY 7/7] Generating plots and saving results...")

    plot_score_distribution(combined, if_norm, ae_norm if ae_avail else None)
    plot_feature_importance(importances)
    plot_umap_anomaly(X_scaled, combined)
    plot_reconstruction_error(loss_curve)
    plot_top_anomalous_customers(anomaly_df, cols_used, top_n=20)
    plot_return_flag_scatter(anomaly_df)

    # Metrics summary
    metrics = {
        'n_total':          len(anomaly_df),
        'n_anomaly':        int(anomaly_df['is_anomaly'].sum()),
        'pct_anomaly':      anomaly_df['is_anomaly'].mean() * 100,
        'n_high_return':    int(anomaly_df['is_high_return'].sum()),
        'n_whale_anomaly':  int(anomaly_df['is_whale_anomaly'].sum()),
        'n_suspicious':     int(anomaly_df['is_suspicious'].sum()),
        'if_contamination': IF_CONTAMINATION,
        'ae_trained':       ae_avail,
        'ae_final_loss':    loss_curve[-1] if loss_curve else None,
        'models': [
            {
                'name':       'Isolation Forest',
                'mean_score': float(if_norm.mean()),
                'n_flagged':  int((if_norm > ANOMALY_SCORE_THRESHOLD).sum()),
                'pct_flagged': (if_norm > ANOMALY_SCORE_THRESHOLD).mean() * 100,
            },
            {
                'name':       'Autoencoder' if ae_avail else 'Autoencoder (N/A)',
                'mean_score': float(ae_norm.mean()),
                'n_flagged':  int((ae_norm > ANOMALY_SCORE_THRESHOLD).sum()),
                'pct_flagged': (ae_norm > ANOMALY_SCORE_THRESHOLD).mean() * 100,
            },
            {
                'name':       'Ensemble (Combined)',
                'mean_score': float(combined.mean()),
                'n_flagged':  int((combined > ANOMALY_SCORE_THRESHOLD).sum()),
                'pct_flagged': (combined > ANOMALY_SCORE_THRESHOLD).mean() * 100,
            },
        ],
    }

    save_anomaly_results(anomaly_df, if_model, ae_model, scaler, cols_used, metrics)

    # Print summary
    _print_leaderboard(metrics)
    print(f"\n  Files saved to: {GRAPHS_DIR}")
    print(f"  Bundle saved  : {MODELS_DIR / 'anomaly_bundle.pkl'}")
    print(f"\n  Next steps:")
    print(f"  1. Add Anomaly tab to streamlit_app.py")
    print(f"  2. Add Segments tab to streamlit_app.py")
    print(f"  3. Build FastAPI endpoint (api/main.py)")

    return {
        'anomaly_df':    anomaly_df,
        'if_model':      if_model,
        'ae_model':      ae_model,
        'scaler':        scaler,
        'cols_used':     cols_used,
        'importances':   importances,
        'metrics':       metrics,
        'combined_score': combined,
        'if_norm':       if_norm,
        'ae_norm':       ae_norm,
    }