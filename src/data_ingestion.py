"""
Data ingestion and cleaning module.
Handles raw CSV loading, schema validation, and e-commerce domain cleaning.
"""
import logging
import pandas as pd
from pathlib import Path
from src.config import FILEPATH

logger = logging.getLogger(__name__)

# Schema contract — pipeline fails fast if upstream data changes
REQUIRED_COLUMNS = {'Customer ID', 'InvoiceDate', 'Quantity', 'Price'}


def load_data(filepath: Path = FILEPATH) -> pd.DataFrame:
    """Loads raw transaction data from the configured path."""
    logger.info("[1/8] Loading Data Source")

    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found at '{filepath}'. "
            "Ensure Google Drive is mounted and CLV_BASE_DIR is set correctly."
        )

    # Customer ID is intentionally NOT cast here — the raw CSV stores it as
    # float (due to NaN rows), so casting to str at read time produces '12345.0'
    # strings. Clean casting to int → str is handled in clean_data() after
    # NaN rows are dropped.
    dtype_dict = {'Quantity': 'Int32', 'Price': 'float64'}  # FIX D-1: was int32/float32 — Int32 handles NaNs, float64 prevents price precision loss

    try:
        raw_data = pd.read_csv(filepath, encoding='ISO-8859-1', dtype=dtype_dict)
        logger.info(f"Data loaded. Shape: {raw_data.shape}")
        return raw_data
    except Exception as e:
        logger.error(f"Data parsing failed: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw transaction data.
    Applies schema validation, date parsing, and domain-specific filters.
    Each filtering step is individually accounted for in the logs.
    """
    logger.info("[2/8] Cleaning Data & Handling Returns")

    # --- Schema Validation ---
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Input data is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    cleaned_df = df.copy()
    step_counts = {'initial': len(cleaned_df)}

    # --- Step 1: Drop missing Customer IDs ---
    cleaned_df = cleaned_df.dropna(subset=['Customer ID'])
    step_counts['after_customer_id_drop'] = len(cleaned_df)

    # --- Step 2: Cast Customer ID cleanly to avoid '12345.0' string artifacts ---
    # Raw CSV encodes Customer ID as float due to NaN rows. After dropping NaNs,
    # we cast float → int → str to produce clean IDs e.g. '12345' not '12345.0'.
    cleaned_df['Customer ID'] = (
        pd.to_numeric(cleaned_df['Customer ID'], errors='coerce')
        .astype('Int64')          # nullable integer — handles any residual NaNs
        .astype(str)
    )
    # Drop any rows where coercion produced '<NA>' strings
    cleaned_df = cleaned_df[cleaned_df['Customer ID'] != '<NA>']

    # --- Step 3: Parse dates ---
    cleaned_df['InvoiceDate'] = pd.to_datetime(cleaned_df['InvoiceDate'])

    # --- Step 4: Remove zero/negative prices FIRST so return count is accurate ---
    # FIX D-2: price filter moved before return count — return count was previously
    # inflated by including returns with zero/negative prices in the audit log.
    cleaned_df = cleaned_df[cleaned_df['Price'] > 0]
    step_counts['after_price_filter'] = len(cleaned_df)

    # --- Step 5: Count returns AFTER price filter for accurate reporting ---
    returns_count = (cleaned_df['Quantity'] < 0).sum()  # FIX D-2: moved after price filter

    # --- Step 6: Remove returns/cancellations ---
    cleaned_df = cleaned_df[cleaned_df['Quantity'] > 0]
    step_counts['after_quantity_filter'] = len(cleaned_df)

    # --- Step 7: Deduplicate transactions ---
    # Online Retail II is known to contain duplicate invoice rows that inflate
    # Frequency and TotalAmount. Deduplicate on the natural transaction key.
    dupe_key_cols = [c for c in ['Invoice', 'StockCode', 'Customer ID', 'InvoiceDate']
                     if c in cleaned_df.columns]
    n_before_dedup = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=dupe_key_cols)
    step_counts['after_dedup'] = len(cleaned_df)

    # --- Step 8: Calculate revenue — upcast to float64 to prevent precision loss ---
    cleaned_df['TotalAmount'] = (
        cleaned_df['Quantity'].astype('float64') * cleaned_df['Price'].astype('float64')
    )

    # --- Audit Log ---
    logger.info(
        f"Rows dropped — Missing Customer ID: "
        f"{step_counts['initial'] - step_counts['after_customer_id_drop']}"
    )
    logger.info(
        f"Rows dropped — Zero/Negative Price: "
        f"{step_counts['after_customer_id_drop'] - step_counts['after_price_filter']}"
    )
    logger.info(f"Return transactions excluded: {returns_count}")
    logger.info(
        f"Rows dropped — Quantity filter: "
        f"{step_counts['after_price_filter'] - step_counts['after_quantity_filter']}"
    )
    logger.info(f"Duplicate rows removed: {n_before_dedup - step_counts['after_dedup']}")
    logger.info(f"Final valid transactions: {step_counts['after_dedup']}")

    return cleaned_df