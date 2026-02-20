"""
Data ingestion and cleaning module.
"""
import pandas as pd
import gdown
from src.config import logger, DOWNLOAD_URL, FILEPATH

def load_data(filepath: str = FILEPATH, url: str = DOWNLOAD_URL) -> pd.DataFrame:
    logger.info("[1/8] Loading Data Source")
    
    try:
        try:
            pd.read_csv(filepath, nrows=1)
            logger.info(" Local file found. Skipping download.")
        except FileNotFoundError:
            logger.info(" Local file not found. Downloading via gdown")
            gdown.download(url, filepath, quiet=False)

        dtype_dict = {'Quantity': 'int32', 'Price': 'float32', 'Customer ID': 'str'}
        raw_data = pd.read_csv(filepath, encoding='ISO-8859-1', dtype=dtype_dict)
        logger.info(f"Data Loaded. Shape: {raw_data.shape}")
        return raw_data
        
    except Exception as e:
        raise RuntimeError(f"Data Load Failed: {e}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("[2/8] Cleaning Data & Handling Returns...")
    cleaned_df = df.copy()
    
    cleaned_df.dropna(subset=['Customer ID'], inplace=True)
    cleaned_df['InvoiceDate'] = pd.to_datetime(cleaned_df['InvoiceDate'])
    
    cleaned_df['TotalAmount'] = cleaned_df['Quantity'] * cleaned_df['Price']
    cleaned_df = cleaned_df[cleaned_df['Price'] > 0]
    
    logger.info(f"Cleaning Complete. Valid Transactions: {len(cleaned_df)}")
    return cleaned_df