"""
Centralized configuration, path management, and logging setup.
All pipeline constants flow from this single source of truth.

UPGRADED v2.5.0 (Round 6):
  - MODEL_VERSION bumped to 2.5.0
  - No structural changes — all upgrades in modeling.py / evaluation.py
"""
import os
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

BASE_DIR = Path(
    os.environ.get(
        "CLV_BASE_DIR",
        Path(__file__).resolve().parent.parent
    )
)

DATA_DIR      = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
GRAPHS_DIR    = ARTIFACTS_DIR / "graphs"
MODELS_DIR    = ARTIFACTS_DIR / "models"
LOG_FILE      = BASE_DIR / "clv_pipeline_run.log"


def setup_logging() -> None:
    """
    Configures root logger with RotatingFileHandler and StreamHandler.
    Idempotent — safe to call multiple times.
    Must be called AFTER Google Drive is mounted.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
    )

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, mode='a'
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


# ---------------------------------------------------------------------------
# Project Constants
# ---------------------------------------------------------------------------
FILEPATH      = DATA_DIR / "online_retail_II.csv"
RANDOM_SEED   = 42
SPLIT_DAYS    = 90
TARGET_COL    = 'Next_3M_Spend'
MODEL_VERSION = "2.5.0"

FEATURE_COLS = [
    # --- Core RFM ---
    'Recency',
    'Frequency',
    'Monetary',

    # --- BTYD Probabilistic Features ---
    'Prob_Pred_Txn',
    'Prob_Pred_Val',
    'Prob_Alive',

    # --- Behavioral Consistency ---
    'Interpurchase_Std',

    # --- Behavioral Velocity ---
    'Purchase_Rate',
    'Days_Since_Purchase',
    'Revenue_Per_Day',

    # --- High-Signal Behavioral Features ---
    'Unique_Products',
    'Visit_Diversity',
    'Avg_Basket_Size',
    'Return_Rate',

    # --- Whale-Detection Features ---
    'Monetary_Percentile',
    'Max_Single_Order',
]


def setup_directories() -> None:
    """Creates required directory structure. Call once from entrypoint."""
    logger = logging.getLogger(__name__)
    for directory in [DATA_DIR, GRAPHS_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    logger.info("Directory structure verified.")