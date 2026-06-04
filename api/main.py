"""
Customer Intelligence Platform — FastAPI Endpoint
==================================================
Production REST API for CLV prediction, customer segmentation,
and anomaly detection.

Endpoints
---------
POST /predict-clv       → 90-day CLV prediction + segment + confidence range
POST /segment-customer  → cluster assignment + segment name + profile
POST /detect-anomaly    → anomaly score + risk level + business flags
GET  /health            → model versions + bundle status + uptime

Usage
-----
    uvicorn api.main:app --reload --port 8000

    # Or from project root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Docs available at:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
"""

import os
import sys
import logging
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or api/ subfolder
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).resolve().parent          # api/
_PROJECT = _HERE.parent                             # project root
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

os.environ.setdefault("CLV_BASE_DIR", str(_PROJECT))

from src.config import MODELS_DIR, FEATURE_COLS, RANDOM_SEED

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App constants
# ---------------------------------------------------------------------------
APP_VERSION     = "1.0.0"
APP_START_TIME  = time.time()

LOG_PRED_MAX    = 12.0
SEGMENT_P20     = 150.0
SEGMENT_P80     = 1_200.0

ANOMALY_THRESHOLD     = 0.50
RETURN_RATE_THRESHOLD = 0.30

FEATURE_DEFAULTS = {
    "Recency": 365.0, "Frequency": 4.0, "Monetary": 300.0,
    "Prob_Pred_Txn": 1.5, "Prob_Pred_Val": 280.0, "Prob_Alive": 0.65,
    "Interpurchase_Std": 45.0, "Purchase_Rate": 0.012,
    "Days_Since_Purchase": 60.0, "Revenue_Per_Day": 1.8,
    "Unique_Products": 12.0, "Visit_Diversity": 5.0,
    "Avg_Basket_Size": 3.2, "Return_Rate": 0.04,
    "Monetary_Percentile": 0.50, "Max_Single_Order": 250.0,
}

SEGMENT_LABELS = {
    "whale":   "🐋 Whale",
    "mid":     "💰 Mid",
    "low":     "📉 Low",
    "churned": "💤 Churned",
}


# ===========================================================================
# Model Registry — loaded once at startup
# ===========================================================================

class ModelRegistry:
    """
    Singleton that loads all three model bundles at startup.
    Exposes clv_model, seg_bundle, anomaly_bundle with version metadata.
    """

    def __init__(self):
        self.clv_bundle     = None
        self.clv_model      = None
        self.seg_bundle     = None
        self.anomaly_bundle = None
        self._status        = {}
        self._load_time     = None

    def load(self) -> None:
        """Load all bundles. Called once at FastAPI startup."""
        errors = []

        # CLV champion bundle
        clv_path = MODELS_DIR / "clv_champion_bundle.pkl"
        try:
            self.clv_bundle = joblib.load(clv_path)
            self.clv_model  = self.clv_bundle["model"]
            self._status["clv"] = {
                "loaded":  True,
                "version": self.clv_bundle.get("version", "unknown"),
                "model":   type(self.clv_model).__name__,
            }
            logger.info(f"CLV bundle loaded | {self._status['clv']}")
        except Exception as e:
            errors.append(f"CLV: {e}")
            self._status["clv"] = {"loaded": False, "error": str(e)}

        # Segmentation bundle
        seg_path = MODELS_DIR / "segmentation_bundle.pkl"
        try:
            self.seg_bundle = joblib.load(seg_path)
            self._status["segmentation"] = {
                "loaded":   True,
                "version":  self.seg_bundle.get("version", "unknown"),
                "optimal_k": self.seg_bundle.get("optimal_k", "unknown"),
            }
            logger.info(f"Segmentation bundle loaded | {self._status['segmentation']}")
        except Exception as e:
            errors.append(f"Segmentation: {e}")
            self._status["segmentation"] = {"loaded": False, "error": str(e)}

        # Anomaly bundle
        anomaly_path = MODELS_DIR / "anomaly_bundle.pkl"
        try:
            self.anomaly_bundle = joblib.load(anomaly_path)
            self._status["anomaly"] = {
                "loaded":  True,
                "version": self.anomaly_bundle.get("version", "unknown"),
            }
            logger.info(f"Anomaly bundle loaded | {self._status['anomaly']}")
        except Exception as e:
            errors.append(f"Anomaly: {e}")
            self._status["anomaly"] = {"loaded": False, "error": str(e)}

        self._load_time = datetime.utcnow().isoformat()

        if errors:
            logger.warning(f"Some bundles failed to load: {errors}")
        else:
            logger.info("All model bundles loaded successfully.")

    @property
    def status(self) -> Dict:
        return self._status

    @property
    def load_time(self) -> Optional[str]:
        return self._load_time


registry = ModelRegistry()


# ===========================================================================
# FastAPI App
# ===========================================================================

app = FastAPI(
    title="Customer Intelligence Platform API",
    description=(
        "REST API for CLV prediction, customer segmentation, and anomaly detection. "
        "Built on TwoStageRegressor (CatBoost) champion model v2.5.0."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load all model bundles when the server starts."""
    logger.info("FastAPI startup — loading model registry...")
    registry.load()
    logger.info("Model registry ready.")


# ===========================================================================
# Request / Response Schemas
# ===========================================================================

class CustomerFeatures(BaseModel):
    """
    Customer feature input for CLV prediction and segmentation.
    All fields are optional — missing values filled with training-set medians.
    """
    Recency:             Optional[float] = Field(None, ge=0,    description="Days since first purchase")
    Frequency:           Optional[float] = Field(None, ge=0,    description="Number of invoices")
    Monetary:            Optional[float] = Field(None, ge=0,    description="Average order value ($)")
    Prob_Pred_Txn:       Optional[float] = Field(None, ge=0,    description="BG/NBD predicted transactions")
    Prob_Pred_Val:       Optional[float] = Field(None, ge=0,    description="Gamma-Gamma predicted value")
    Prob_Alive:          Optional[float] = Field(None, ge=0, le=1, description="Probability customer is alive")
    Interpurchase_Std:   Optional[float] = Field(None, ge=0,    description="Std of days between purchases")
    Purchase_Rate:       Optional[float] = Field(None, ge=0,    description="Purchases per day")
    Days_Since_Purchase: Optional[float] = Field(None, ge=0,    description="Days since last purchase")
    Revenue_Per_Day:     Optional[float] = Field(None, ge=0,    description="Revenue per active day ($)")
    Unique_Products:     Optional[float] = Field(None, ge=0,    description="Unique products purchased")
    Visit_Diversity:     Optional[float] = Field(None, ge=0,    description="Visit diversity score")
    Avg_Basket_Size:     Optional[float] = Field(None, ge=0,    description="Average items per basket")
    Return_Rate:         Optional[float] = Field(None, ge=0, le=1, description="Fraction of orders returned")
    Monetary_Percentile: Optional[float] = Field(None, ge=0, le=1, description="Monetary percentile rank")
    Max_Single_Order:    Optional[float] = Field(None, ge=0,    description="Largest single order value ($)")

    model_config = {"extra": "ignore"}


class AnomalyFeatures(BaseModel):
    """
    Customer feature input for anomaly detection.
    Uses the 12-feature anomaly subset.
    """
    Recency:             Optional[float] = Field(None, ge=0)
    Frequency:           Optional[float] = Field(None, ge=0)
    Monetary:            Optional[float] = Field(None, ge=0)
    Return_Rate:         Optional[float] = Field(None, ge=0, le=1)
    Avg_Basket_Size:     Optional[float] = Field(None, ge=0)
    Max_Single_Order:    Optional[float] = Field(None, ge=0)
    Monetary_Percentile: Optional[float] = Field(None, ge=0, le=1)
    Purchase_Rate:       Optional[float] = Field(None, ge=0)
    Unique_Products:     Optional[float] = Field(None, ge=0)
    Days_Since_Purchase: Optional[float] = Field(None, ge=0)
    Interpurchase_Std:   Optional[float] = Field(None, ge=0)
    Revenue_Per_Day:     Optional[float] = Field(None, ge=0)

    model_config = {"extra": "ignore"}


class CLVResponse(BaseModel):
    predicted_clv_90d:  float
    clv_low:            float
    clv_high:           float
    segment:            str
    segment_key:        str
    log_prediction:     float
    model_version:      str
    features_used:      Dict[str, float]


class SegmentResponse(BaseModel):
    cluster_id:         int
    segment_name:       str
    silhouette_score:   float
    optimal_k:          int
    feature_profile:    Dict[str, float]
    model_version:      str


class AnomalyResponse(BaseModel):
    anomaly_score:      float
    if_score:           float
    risk_level:         str
    is_anomaly:         bool
    is_high_return:     bool
    flags:              Dict[str, bool]
    model_version:      str


class HealthResponse(BaseModel):
    status:             str
    api_version:        str
    uptime_seconds:     float
    models:             Dict[str, Any]
    timestamp:          str


# ===========================================================================
# Helper Functions
# ===========================================================================

def _build_feature_row(data: CustomerFeatures) -> pd.DataFrame:
    """Merge input with defaults, return single-row DataFrame in FEATURE_COLS order."""
    row = FEATURE_DEFAULTS.copy()
    for field, val in data.model_dump().items():
        if val is not None:
            row[field] = val
    return pd.DataFrame([row])[FEATURE_COLS]


def _get_segment(dollar: float) -> tuple:
    """Return (segment_label, segment_key) for a dollar CLV value."""
    if dollar < 0.01:
        return SEGMENT_LABELS["churned"], "churned"
    elif dollar < SEGMENT_P20:
        return SEGMENT_LABELS["low"], "low"
    elif dollar < SEGMENT_P80:
        return SEGMENT_LABELS["mid"], "mid"
    else:
        return SEGMENT_LABELS["whale"], "whale"


def _require_bundle(bundle, name: str) -> None:
    """Raise 503 if a required bundle isn't loaded."""
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"{name} bundle not loaded. Check /health for details."
        )


# ===========================================================================
# Endpoints
# ===========================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Returns model versions, bundle load status, and API uptime.
    Use this to verify all models are loaded before sending predictions.
    """
    uptime = round(time.time() - APP_START_TIME, 2)
    all_loaded = all(
        v.get("loaded", False) for v in registry.status.values()
    )
    return HealthResponse(
        status    = "healthy" if all_loaded else "degraded",
        api_version = APP_VERSION,
        uptime_seconds = uptime,
        models    = registry.status,
        timestamp = datetime.utcnow().isoformat(),
    )


@app.post("/predict-clv", response_model=CLVResponse, tags=["Prediction"])
async def predict_clv(customer: CustomerFeatures):
    """
    Predicts 90-day Customer Lifetime Value.

    - Missing features filled with training-set medians
    - Returns dollar CLV + ±15% confidence range
    - Segment assigned based on predicted spend thresholds:
      Whale ≥ $1,200 | Mid $150–$1,200 | Low $0.01–$150 | Churned $0
    """
    _require_bundle(registry.clv_model, "CLV")

    feature_row = _build_feature_row(customer)

    try:
        log_pred = float(np.clip(registry.clv_model.predict(feature_row), 0, LOG_PRED_MAX)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    dollar   = float(np.expm1(log_pred))
    clv_low  = round(max(0.0, dollar * 0.85), 2)
    clv_high = round(dollar * 1.15, 2)
    segment, seg_key = _get_segment(dollar)

    return CLVResponse(
        predicted_clv_90d = round(dollar, 2),
        clv_low           = clv_low,
        clv_high          = clv_high,
        segment           = segment,
        segment_key       = seg_key,
        log_prediction    = round(log_pred, 6),
        model_version     = registry.clv_bundle.get("version", "unknown"),
        features_used     = {k: round(float(v), 4)
                             for k, v in feature_row.iloc[0].to_dict().items()},
    )


@app.post("/segment-customer", response_model=SegmentResponse, tags=["Segmentation"])
async def segment_customer(customer: CustomerFeatures):
    """
    Assigns a customer to a behavioural segment using K-Means.

    Returns cluster ID, segment name, and the customer's feature profile
    relative to cluster centroids.
    """
    _require_bundle(registry.seg_bundle, "Segmentation")

    bundle    = registry.seg_bundle
    kmeans    = bundle["kmeans"]
    scaler    = bundle["scaler"]
    pca       = bundle.get("pca")
    cols_used = bundle["cols_used"]
    labels    = bundle.get("segment_labels", {})
    metrics   = bundle.get("metrics", {})

    # Build feature row using only cols_used (segmentation subset)
    row = FEATURE_DEFAULTS.copy()
    for field, val in customer.model_dump().items():
        if val is not None:
            row[field] = val

    input_df  = pd.DataFrame([row])
    available = [c for c in cols_used if c in input_df.columns]
    X_sub     = input_df[available].fillna(
        {c: FEATURE_DEFAULTS.get(c, 0.0) for c in available}
    ).values.astype(np.float32)

    try:
        X_scaled = scaler.transform(X_sub)
        if pca is not None:
            X_scaled = pca.transform(X_scaled)
        cluster_id = int(kmeans.predict(X_scaled)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")

    # Segment name from bundle labels or fallback
    seg_name = labels.get(cluster_id, f"Cluster {cluster_id}")

    return SegmentResponse(
        cluster_id      = cluster_id,
        segment_name    = seg_name,
        silhouette_score = round(float(metrics.get("silhouette", 0)), 4),
        optimal_k       = int(bundle.get("optimal_k", 2)),
        feature_profile = {k: round(float(row.get(k, 0)), 4) for k in available},
        model_version   = bundle.get("version", "unknown"),
    )


@app.post("/detect-anomaly", response_model=AnomalyResponse, tags=["Anomaly"])
async def detect_anomaly(customer: AnomalyFeatures):
    """
    Scores a customer for anomalous behaviour using Isolation Forest.

    Returns:
    - anomaly_score [0–1]: higher = more anomalous
    - risk_level: 🔴 High (≥0.65) | 🟡 Medium (≥0.40) | 🟢 Low (<0.40)
    - is_anomaly: True if score ≥ 0.50
    - is_high_return: True if Return_Rate > 0.30 AND score > 0.40
    """
    _require_bundle(registry.anomaly_bundle, "Anomaly")

    bundle   = registry.anomaly_bundle
    if_model = bundle["isolation_forest"]
    scaler   = bundle["scaler"]
    cols     = bundle["cols_used"]
    thresh   = bundle.get("thresholds", {})

    anomaly_threshold     = thresh.get("anomaly_score",   ANOMALY_THRESHOLD)
    return_threshold      = thresh.get("return_rate",     RETURN_RATE_THRESHOLD)
    high_return_score_min = thresh.get("high_return_min", 0.40)

    # Build feature row
    row = {c: FEATURE_DEFAULTS.get(c, 0.0) for c in cols}
    for field, val in customer.model_dump().items():
        if val is not None and field in row:
            row[field] = val

    X = pd.DataFrame([row])[cols].fillna(0).values.astype(np.float32)

    try:
        X_scaled = scaler.transform(X)
        raw_if   = float(-if_model.decision_function(X_scaled)[0])
        # Normalise using the same approach as training (~0.5 scale factor)
        if_score = float(np.clip(raw_if / 0.5, 0.0, 1.0))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly scoring failed: {e}")

    combined      = if_score   # AE stripped for deployment compat
    return_rate   = float(row.get("Return_Rate", 0.0))
    is_anomaly    = combined >= anomaly_threshold
    is_high_return = (return_rate > return_threshold) and (combined > high_return_score_min)

    if combined >= 0.65:
        risk = "🔴 High"
    elif combined >= 0.40:
        risk = "🟡 Medium"
    else:
        risk = "🟢 Low"

    return AnomalyResponse(
        anomaly_score = round(combined, 4),
        if_score      = round(if_score, 4),
        risk_level    = risk,
        is_anomaly    = is_anomaly,
        is_high_return = is_high_return,
        flags = {
            "is_anomaly":     is_anomaly,
            "is_high_return": is_high_return,
            "is_whale":       float(row.get("Monetary_Percentile", 0)) > 0.90,
            "is_suspicious":  is_high_return and is_anomaly and float(row.get("Recency", 999)) < 30,
        },
        model_version = bundle.get("version", "unknown"),
    )


# ===========================================================================
# Request logging middleware
# ===========================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({duration}ms)"
    )
    return response


# ===========================================================================
# Run directly
# ===========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)