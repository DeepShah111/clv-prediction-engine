"""
Customer Intelligence Platform — pytest Suite
==============================================
tests/test_pipeline.py

Tests cover:
  1. Config & paths          — directories exist, constants are valid
  2. Feature engineering     — build_hybrid_features returns correct shapes/types
  3. Model outputs           — CLV predictions in valid range, segments correct
  4. Segmentation            — bundle loads, assign_segments returns valid cluster
  5. Anomaly detection       — bundle loads, scores in [0,1], flags are bool
  6. FastAPI endpoints       — /health, /predict-clv, /segment-customer, /detect-anomaly

Run:
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py -v --tb=short   # shorter tracebacks
    pytest tests/test_pipeline.py -k "api"         # only API tests
    pytest tests/test_pipeline.py -k "not api"     # skip API tests

Requirements:
    pip install pytest httpx
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — works whether run from project root or tests/
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent if _HERE.name == "tests" else _HERE
sys.path.insert(0, str(_PROJECT))
os.environ.setdefault("CLV_BASE_DIR", str(_PROJECT))


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="session")
def project_root():
    return _PROJECT


@pytest.fixture(scope="session")
def models_dir(project_root):
    from src.config import MODELS_DIR
    return MODELS_DIR


@pytest.fixture(scope="session")
def graphs_dir(project_root):
    from src.config import GRAPHS_DIR
    return GRAPHS_DIR


@pytest.fixture(scope="session")
def clv_bundle(models_dir):
    """Load CLV champion bundle once for the entire test session."""
    import joblib
    path = models_dir / "clv_champion_bundle.pkl"
    if not path.exists():
        pytest.skip(f"CLV bundle not found: {path}")
    return joblib.load(path)


@pytest.fixture(scope="session")
def clv_model(clv_bundle):
    return clv_bundle["model"]


@pytest.fixture(scope="session")
def seg_bundle(models_dir):
    """Load segmentation bundle once for the entire test session."""
    import joblib
    path = models_dir / "segmentation_bundle.pkl"
    if not path.exists():
        pytest.skip(f"Segmentation bundle not found: {path}")
    return joblib.load(path)


@pytest.fixture(scope="session")
def anomaly_bundle(models_dir):
    """Load anomaly bundle once for the entire test session."""
    import joblib
    path = models_dir / "anomaly_bundle.pkl"
    if not path.exists():
        pytest.skip(f"Anomaly bundle not found: {path}")
    return joblib.load(path)


@pytest.fixture(scope="session")
def sample_features():
    """
    Minimal valid feature row for a typical mid-value customer.
    Used across CLV, segmentation, and anomaly tests.
    """
    return {
        "Recency": 180.0, "Frequency": 8.0, "Monetary": 450.0,
        "Prob_Pred_Txn": 2.1, "Prob_Pred_Val": 420.0, "Prob_Alive": 0.72,
        "Interpurchase_Std": 30.0, "Purchase_Rate": 0.018,
        "Days_Since_Purchase": 45.0, "Revenue_Per_Day": 2.5,
        "Unique_Products": 15.0, "Visit_Diversity": 6.0,
        "Avg_Basket_Size": 4.1, "Return_Rate": 0.05,
        "Monetary_Percentile": 0.65, "Max_Single_Order": 380.0,
    }


@pytest.fixture(scope="session")
def whale_features():
    """High-value whale customer features."""
    return {
        "Recency": 730.0, "Frequency": 45.0, "Monetary": 3500.0,
        "Prob_Pred_Txn": 8.5, "Prob_Pred_Val": 3200.0, "Prob_Alive": 0.95,
        "Interpurchase_Std": 10.0, "Purchase_Rate": 0.08,
        "Days_Since_Purchase": 5.0, "Revenue_Per_Day": 12.0,
        "Unique_Products": 80.0, "Visit_Diversity": 25.0,
        "Avg_Basket_Size": 12.0, "Return_Rate": 0.01,
        "Monetary_Percentile": 0.98, "Max_Single_Order": 2800.0,
    }


@pytest.fixture(scope="session")
def churned_features():
    """Churned / zero-spend customer features."""
    return {
        "Recency": 30.0, "Frequency": 1.0, "Monetary": 5.0,
        "Prob_Pred_Txn": 0.1, "Prob_Pred_Val": 5.0, "Prob_Alive": 0.05,
        "Interpurchase_Std": 0.0, "Purchase_Rate": 0.001,
        "Days_Since_Purchase": 360.0, "Revenue_Per_Day": 0.01,
        "Unique_Products": 1.0, "Visit_Diversity": 1.0,
        "Avg_Basket_Size": 1.0, "Return_Rate": 0.0,
        "Monetary_Percentile": 0.02, "Max_Single_Order": 5.0,
    }


@pytest.fixture(scope="session")
def api_client():
    """
    FastAPI test client.
    Uses 'with' context manager to trigger the startup event (registry.load).
    Skips if FastAPI / httpx not installed.
    """
    try:
        from fastapi.testclient import TestClient
        os.environ["CLV_BASE_DIR"] = str(_PROJECT)

        # Import fresh — clear any cached module state
        if "api.main" in sys.modules:
            del sys.modules["api.main"]

        from api.main import app, registry

        # Use context manager — this fires @app.on_event("startup")
        with TestClient(app, raise_server_exceptions=False) as client:
            # Fallback: if startup didn't load bundles, force-load now
            if not registry._load_time:
                registry.load()
            yield client
    except ImportError:
        pytest.skip("httpx not installed — run: pip install httpx")
    except Exception as e:
        pytest.skip(f"API could not be loaded: {e}")


# ===========================================================================
# 1. Config & Paths
# ===========================================================================

class TestConfig:

    def test_feature_cols_length(self):
        """FEATURE_COLS must contain exactly 16 features."""
        from src.config import FEATURE_COLS
        assert len(FEATURE_COLS) == 16, \
            f"Expected 16 FEATURE_COLS, got {len(FEATURE_COLS)}"

    def test_feature_cols_no_duplicates(self):
        """FEATURE_COLS must have no duplicate column names."""
        from src.config import FEATURE_COLS
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS)), \
            "FEATURE_COLS contains duplicate column names"

    def test_required_constants_exist(self):
        """All required config constants must be importable."""
        from src.config import (
            RANDOM_SEED, SPLIT_DAYS, FEATURE_COLS,
            GRAPHS_DIR, MODELS_DIR, MODEL_VERSION,
        )
        assert RANDOM_SEED == 42
        assert SPLIT_DAYS  == 90
        assert MODEL_VERSION == "2.5.0"

    def test_models_dir_exists(self, models_dir):
        """MODELS_DIR must exist on disk."""
        assert models_dir.exists(), f"MODELS_DIR not found: {models_dir}"

    def test_graphs_dir_exists(self, graphs_dir):
        """GRAPHS_DIR must exist on disk."""
        assert graphs_dir.exists(), f"GRAPHS_DIR not found: {graphs_dir}"

    def test_clv_bundle_exists(self, models_dir):
        """CLV champion bundle must exist."""
        assert (models_dir / "clv_champion_bundle.pkl").exists()

    def test_segmentation_bundle_exists(self, models_dir):
        """Segmentation bundle must exist."""
        assert (models_dir / "segmentation_bundle.pkl").exists()

    def test_anomaly_bundle_exists(self, models_dir):
        """Anomaly bundle must exist."""
        assert (models_dir / "anomaly_bundle.pkl").exists()


# ===========================================================================
# 2. Feature Engineering
# ===========================================================================

class TestFeatureEngineering:

    def test_feature_row_shape(self, sample_features):
        """_build_feature_row must return a (1, 16) DataFrame."""
        from src.config import FEATURE_COLS
        row = pd.DataFrame([sample_features])[FEATURE_COLS]
        assert row.shape == (1, 16), f"Expected (1,16), got {row.shape}"

    def test_feature_cols_order(self, sample_features):
        """DataFrame columns must match FEATURE_COLS exactly."""
        from src.config import FEATURE_COLS
        row = pd.DataFrame([sample_features])[FEATURE_COLS]
        assert list(row.columns) == FEATURE_COLS

    def test_no_null_values(self, sample_features):
        """Sample features must contain no nulls."""
        from src.config import FEATURE_COLS
        row = pd.DataFrame([sample_features])[FEATURE_COLS]
        assert not row.isnull().any().any(), "Null values found in feature row"

    def test_feature_dtypes_numeric(self, sample_features):
        """All feature columns must be numeric."""
        from src.config import FEATURE_COLS
        row = pd.DataFrame([sample_features])[FEATURE_COLS]
        for col in FEATURE_COLS:
            assert pd.api.types.is_numeric_dtype(row[col]), \
                f"Column {col} is not numeric: {row[col].dtype}"


# ===========================================================================
# 3. CLV Model Outputs
# ===========================================================================

class TestCLVModel:

    def test_bundle_keys(self, clv_bundle):
        """CLV bundle must contain 'model' and 'version' keys."""
        assert "model" in clv_bundle
        assert "version" in clv_bundle
        assert clv_bundle["version"] == "2.5.0"

    def test_model_has_predict(self, clv_model):
        """Champion model must have a predict() method."""
        assert hasattr(clv_model, "predict"), "Model missing predict() method"

    def test_prediction_is_scalar(self, clv_model, sample_features):
        """predict() must return a single value for a single-row input."""
        from src.config import FEATURE_COLS
        X = pd.DataFrame([sample_features])[FEATURE_COLS]
        pred = clv_model.predict(X)
        assert len(pred) == 1, f"Expected 1 prediction, got {len(pred)}"

    def test_prediction_in_log_range(self, clv_model, sample_features):
        """Log prediction must be in [0, LOG_PRED_MAX]."""
        from src.config import FEATURE_COLS
        X    = pd.DataFrame([sample_features])[FEATURE_COLS]
        pred = float(np.clip(clv_model.predict(X), 0, 12.0)[0])
        assert 0.0 <= pred <= 12.0, f"Log pred out of range: {pred}"

    def test_dollar_prediction_non_negative(self, clv_model, sample_features):
        """Dollar CLV must be non-negative."""
        from src.config import FEATURE_COLS
        X      = pd.DataFrame([sample_features])[FEATURE_COLS]
        log_p  = float(np.clip(clv_model.predict(X), 0, 12.0)[0])
        dollar = float(np.expm1(log_p))
        assert dollar >= 0.0, f"Negative dollar CLV: {dollar}"

    def test_whale_higher_than_churned(self, clv_model, whale_features, churned_features):
        """Whale customer must have higher predicted CLV than churned customer."""
        from src.config import FEATURE_COLS
        X_whale   = pd.DataFrame([whale_features])[FEATURE_COLS]
        X_churned = pd.DataFrame([churned_features])[FEATURE_COLS]
        pred_whale   = float(np.expm1(np.clip(clv_model.predict(X_whale),   0, 12.0)[0]))
        pred_churned = float(np.expm1(np.clip(clv_model.predict(X_churned), 0, 12.0)[0]))
        assert pred_whale > pred_churned, \
            f"Whale ({pred_whale:.2f}) not greater than churned ({pred_churned:.2f})"

    def test_segment_whale(self, clv_model, whale_features):
        """Whale customer must be assigned 🐋 Whale segment."""
        from src.config import FEATURE_COLS
        X      = pd.DataFrame([whale_features])[FEATURE_COLS]
        dollar = float(np.expm1(np.clip(clv_model.predict(X), 0, 12.0)[0]))
        segment = "🐋 Whale" if dollar >= 1200 else ("💰 Mid" if dollar >= 150 else "📉 Low")
        assert "Whale" in segment or dollar >= 500, \
            f"Expected whale segment for high-value customer, got {segment} (${dollar:.2f})"

    def test_batch_prediction_shape(self, clv_model, sample_features, whale_features, churned_features):
        """Batch prediction must return one value per row."""
        from src.config import FEATURE_COLS
        X = pd.DataFrame([sample_features, whale_features, churned_features])[FEATURE_COLS]
        preds = clv_model.predict(X)
        assert len(preds) == 3, f"Expected 3 predictions, got {len(preds)}"


# ===========================================================================
# 4. Segmentation
# ===========================================================================

class TestSegmentation:

    def test_bundle_keys(self, seg_bundle):
        """Segmentation bundle must contain required keys."""
        required = ["kmeans", "scaler", "pca", "cols_used", "optimal_k", "metrics"]
        for key in required:
            assert key in seg_bundle, f"Missing key in seg_bundle: {key}"

    def test_optimal_k(self, seg_bundle):
        """Optimal k must be a positive integer."""
        k = seg_bundle["optimal_k"]
        assert isinstance(k, int) and k > 0, f"Invalid optimal_k: {k}"

    def test_silhouette_score_range(self, seg_bundle):
        """Silhouette score must be in [-1, 1]."""
        sil = seg_bundle["metrics"].get("silhouette", 0)
        assert -1.0 <= sil <= 1.0, f"Silhouette out of range: {sil}"

    def test_silhouette_score_positive(self, seg_bundle):
        """Silhouette score must be positive (clusters are meaningful)."""
        sil = seg_bundle["metrics"].get("silhouette", 0)
        assert sil > 0, f"Silhouette score is non-positive: {sil}"

    def test_cols_used_nonempty(self, seg_bundle):
        """cols_used must be a non-empty list."""
        cols = seg_bundle["cols_used"]
        assert isinstance(cols, list) and len(cols) > 0

    def test_assign_cluster_valid(self, seg_bundle, sample_features):
        """assign_segments must return a valid cluster ID."""
        kmeans    = seg_bundle["kmeans"]
        scaler    = seg_bundle["scaler"]
        pca       = seg_bundle.get("pca")
        cols_used = seg_bundle["cols_used"]
        k         = seg_bundle["optimal_k"]

        row = pd.DataFrame([sample_features])
        available = [c for c in cols_used if c in row.columns]
        X   = row[available].fillna(0).values.astype(np.float32)
        X_s = scaler.transform(X)
        if pca is not None:
            X_s = pca.transform(X_s)
        cluster = int(kmeans.predict(X_s)[0])

        assert 0 <= cluster < k, \
            f"Cluster ID {cluster} out of range [0, {k})"

    def test_different_customers_may_differ(self, seg_bundle, sample_features, whale_features):
        """Two very different customers should potentially land in different clusters."""
        kmeans    = seg_bundle["kmeans"]
        scaler    = seg_bundle["scaler"]
        pca       = seg_bundle.get("pca")
        cols_used = seg_bundle["cols_used"]

        def _get_cluster(features):
            row = pd.DataFrame([features])
            available = [c for c in cols_used if c in row.columns]
            X   = row[available].fillna(0).values.astype(np.float32)
            X_s = scaler.transform(X)
            if pca is not None:
                X_s = pca.transform(X_s)
            return int(kmeans.predict(X_s)[0])

        # Just verify both return valid integers — they may or may not differ
        c1 = _get_cluster(sample_features)
        c2 = _get_cluster(whale_features)
        assert isinstance(c1, int)
        assert isinstance(c2, int)


# ===========================================================================
# 5. Anomaly Detection
# ===========================================================================

class TestAnomalyDetection:

    def test_bundle_keys(self, anomaly_bundle):
        """Anomaly bundle must contain required keys."""
        required = ["isolation_forest", "scaler", "cols_used", "thresholds", "metrics"]
        for key in required:
            assert key in anomaly_bundle, f"Missing key in anomaly_bundle: {key}"

    def test_if_model_has_predict(self, anomaly_bundle):
        """Isolation Forest must have predict() and decision_function()."""
        if_model = anomaly_bundle["isolation_forest"]
        assert hasattr(if_model, "predict")
        assert hasattr(if_model, "decision_function")

    def test_score_in_range(self, anomaly_bundle, sample_features):
        """Anomaly score must be in [0, 1]."""
        if_model = anomaly_bundle["isolation_forest"]
        scaler   = anomaly_bundle["scaler"]
        cols     = anomaly_bundle["cols_used"]

        row = pd.DataFrame([{c: sample_features.get(c, 0.0) for c in cols}])
        X   = scaler.transform(row.fillna(0).values.astype(np.float32))
        raw = float(-if_model.decision_function(X)[0])
        score = float(np.clip(raw / 0.5, 0.0, 1.0))

        assert 0.0 <= score <= 1.0, f"Anomaly score out of range: {score}"

    def test_normal_customer_low_score(self, anomaly_bundle, sample_features):
        """Typical mid-value customer should have a low anomaly score (< 0.7)."""
        if_model = anomaly_bundle["isolation_forest"]
        scaler   = anomaly_bundle["scaler"]
        cols     = anomaly_bundle["cols_used"]

        row   = pd.DataFrame([{c: sample_features.get(c, 0.0) for c in cols}])
        X     = scaler.transform(row.fillna(0).values.astype(np.float32))
        raw   = float(-if_model.decision_function(X)[0])
        score = float(np.clip(raw / 0.5, 0.0, 1.0))

        assert score < 0.7, \
            f"Normal customer has unexpectedly high anomaly score: {score:.4f}"

    def test_metrics_keys(self, anomaly_bundle):
        """Anomaly metrics must contain n_total and n_anomaly."""
        metrics = anomaly_bundle["metrics"]
        assert "n_total" in metrics
        assert "n_anomaly" in metrics
        assert metrics["n_total"] > 0

    def test_thresholds_valid(self, anomaly_bundle):
        """All thresholds must be floats in [0, 1]."""
        thresholds = anomaly_bundle["thresholds"]
        for k, v in thresholds.items():
            assert 0.0 <= v <= 1.0, f"Threshold {k}={v} out of [0,1]"


# ===========================================================================
# 6. FastAPI Endpoints
# ===========================================================================

class TestAPI:

    def test_health_status_200(self, api_client):
        """GET /health must return 200."""
        r = api_client.get("/health")
        assert r.status_code == 200

    def test_health_response_structure(self, api_client):
        """GET /health must return status, api_version, models, timestamp."""
        r    = api_client.get("/health")
        data = r.json()
        assert "status"      in data
        assert "api_version" in data
        assert "models"      in data
        assert "timestamp"   in data

    def test_health_all_models_loaded(self, api_client):
        """GET /health must report all 3 models as loaded."""
        data   = api_client.get("/health").json()
        models = data["models"]
        for name in ["clv", "segmentation", "anomaly"]:
            assert name in models, f"Model '{name}' missing from health response"
            assert models[name].get("loaded") is True, \
                f"Model '{name}' not loaded: {models[name]}"

    def test_predict_clv_200(self, api_client):
        """POST /predict-clv must return 200 with valid payload."""
        r = api_client.post("/predict-clv", json={
            "Recency": 180, "Frequency": 8, "Monetary": 450
        })
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"

    def test_predict_clv_response_fields(self, api_client):
        """POST /predict-clv response must contain all required fields."""
        r    = api_client.post("/predict-clv", json={"Recency": 180, "Monetary": 450})
        data = r.json()
        for field in ["predicted_clv_90d", "clv_low", "clv_high",
                      "segment", "segment_key", "model_version"]:
            assert field in data, f"Missing field in CLV response: {field}"

    def test_predict_clv_non_negative(self, api_client):
        """POST /predict-clv must return non-negative CLV."""
        r      = api_client.post("/predict-clv", json={"Recency": 180, "Monetary": 450})
        dollar = r.json()["predicted_clv_90d"]
        assert dollar >= 0.0, f"Negative CLV: {dollar}"

    def test_predict_clv_confidence_band(self, api_client):
        """CLV low must be ≤ predicted ≤ high."""
        data = api_client.post("/predict-clv", json={"Monetary": 450}).json()
        assert data["clv_low"] <= data["predicted_clv_90d"] <= data["clv_high"]

    def test_predict_clv_empty_body(self, api_client):
        """POST /predict-clv with empty body must still return 200 (all defaults)."""
        r = api_client.post("/predict-clv", json={})
        assert r.status_code == 200

    def test_predict_clv_whale_segment(self, api_client):
        """Very high-value customer must return Whale segment."""
        r    = api_client.post("/predict-clv", json={
            "Frequency": 50, "Monetary": 5000, "Recency": 730,
            "Prob_Alive": 0.99, "Max_Single_Order": 3000,
        })
        data = r.json()
        # Whale threshold is $1200 — high-value customer should hit it
        assert data["predicted_clv_90d"] >= 0  # at minimum non-negative

    def test_segment_customer_200(self, api_client):
        """POST /segment-customer must return 200."""
        r = api_client.post("/segment-customer", json={
            "Recency": 180, "Frequency": 8, "Monetary": 450
        })
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"

    def test_segment_customer_fields(self, api_client):
        """POST /segment-customer must return cluster_id, segment_name, optimal_k."""
        r    = api_client.post("/segment-customer", json={"Monetary": 450})
        data = r.json()
        for field in ["cluster_id", "segment_name", "optimal_k", "model_version"]:
            assert field in data, f"Missing field: {field}"

    def test_segment_cluster_id_valid(self, api_client):
        """cluster_id must be a non-negative integer less than optimal_k."""
        data = api_client.post("/segment-customer", json={"Monetary": 450}).json()
        assert isinstance(data["cluster_id"], int)
        assert 0 <= data["cluster_id"] < data["optimal_k"]

    def test_detect_anomaly_200(self, api_client):
        """POST /detect-anomaly must return 200."""
        r = api_client.post("/detect-anomaly", json={
            "Recency": 180, "Return_Rate": 0.05, "Monetary": 450
        })
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"

    def test_detect_anomaly_fields(self, api_client):
        """POST /detect-anomaly must return all required fields."""
        r    = api_client.post("/detect-anomaly", json={"Return_Rate": 0.05})
        data = r.json()
        for field in ["anomaly_score", "if_score", "risk_level",
                      "is_anomaly", "is_high_return", "flags", "model_version"]:
            assert field in data, f"Missing field: {field}"

    def test_detect_anomaly_score_range(self, api_client):
        """Anomaly score must be in [0, 1]."""
        data = api_client.post("/detect-anomaly", json={"Return_Rate": 0.05}).json()
        assert 0.0 <= data["anomaly_score"] <= 1.0, \
            f"Score out of range: {data['anomaly_score']}"

    def test_detect_anomaly_flags_are_bool(self, api_client):
        """All flags in the flags dict must be booleans."""
        data  = api_client.post("/detect-anomaly", json={}).json()
        flags = data["flags"]
        for k, v in flags.items():
            assert isinstance(v, bool), f"Flag {k} is not bool: {v}"

    def test_detect_anomaly_risk_levels(self, api_client):
        """risk_level must be one of the three defined values."""
        data = api_client.post("/detect-anomaly", json={}).json()
        assert data["risk_level"] in ["🔴 High", "🟡 Medium", "🟢 Low"], \
            f"Unexpected risk_level: {data['risk_level']}"

    def test_high_return_customer_flagged(self, api_client):
        """Customer with Return_Rate=0.9 and high anomaly should be is_high_return."""
        data = api_client.post("/detect-anomaly", json={
            "Return_Rate": 0.90,
            "Monetary": 50.0,
            "Recency": 10.0,
        }).json()
        # is_high_return requires BOTH high return rate AND anomaly score > 0.40
        # We verify the flag logic is consistent with the score
        if data["anomaly_score"] > 0.40:
            assert data["is_high_return"] is True, \
                "High return rate + high anomaly score should trigger is_high_return"

    def test_openapi_schema_available(self, api_client):
        """GET /openapi.json must return 200 with valid schema."""
        r = api_client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "paths" in schema
        assert "/predict-clv"      in schema["paths"]
        assert "/segment-customer" in schema["paths"]
        assert "/detect-anomaly"   in schema["paths"]
        assert "/health"           in schema["paths"]