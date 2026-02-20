"""
Configuration settings for the CLV Prediction Engine.
"""
import logging
import warnings
import seaborn as sns
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)

# Logging Configuration 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("clv_pipeline_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Project Constants 
FILE_ID = "1LwlhwSCCMOq0wcW9Yghn8UbrEmWVF8f4"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"
FILEPATH = "online_retail_II.csv"

# Modeling Parameters 
SPLIT_DAYS = 90
TARGET_COL = 'Next_3M_Spend'
FEATURE_COLS = [
    'Recency', 
    'Frequency', 
    'Monetary', 
    'Prob_Pred_Txn', 
    'Prob_Pred_Val', 
    'Interpurchase_Std'
]