"""
Customer Intelligence Platform — pytest Suite (v2)
==================================================
tests/test_pipeline.py

What these tests are
--------------------
Integration / contract tests of the SAVED artifacts (model bundles) and of the
FastAPI service, plus behavioural checks ("a whale is worth more than a churned
customer"). Unit tests of the data and feature code live in the other test files
(test_feature_engineering.py, test_cv_utils.py, test_anomaly_utils.py, ...).

Areas
  1. Config & paths          — directories exist, constants are valid
  2. Sample feature rows     — the 16-column input contract
  3. CLV model               — raw predictions are sane, behaviour is sensible
  4. Segmentation            — bundle contents, cluster assignment, separation
  5. Anomaly detection       — bundle contents, ranking of extreme customers
  6. FastAPI endpoints       — contract, validation (422), health (503), API key

Run:
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py -v --tb=short   # shorter tracebacks
    pytest tests/test_pipeline.py -k "TestAPI"    # only API tests
    pytest tests/test_pipeline.py -k "not TestAPI"

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


@pytest.fixture
def sample_features():
    """
    Typical mid-value customer (a fresh dictionary for every test).
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


@pytest.fixture
def whale_features():
    """High-value whale customer features (internally consistent)."""
    return {
        "Recency": 730.0, "Frequency": 45.0, "Monetary": 3500.0,
        "Prob_Pred_Txn": 8.5, "Prob_Pred_Val": 3200.0, "Prob_Alive": 0.95,
        "Interpurchase_Std": 10.0, "Purchase_Rate": 0.08,
        "Days_Since_Purchase": 5.0, "Revenue_Per_Day": 12.0,
        "Unique_Products": 80.0, "Visit_Diversity": 25.0,
        "Avg_Basket_Size": 12.0, "Return_Rate": 0.01,
        "Monetary_Percentile": 0.98, "Max_Single_Order": 2800.0,
    }


@pytest.fixture
def churned_features():
    """Churned / zero-spend customer features (internally consistent)."""
    return {
        "Recency": 30.0, "Frequency": 1.0, "Monetary": 5.0,
        "Prob_Pred_Txn": 0.1, "Prob_Pred_Val": 5.0, "Prob_Alive": 0.05,
        "Interpurchase_Std": 0.0, "Purchase_Rate": 0.001,
        "Days_Since_Purchase": 360.0, "Revenue_Per_Day": 0.01,
        "Unique_Products": 1.0, "Visit_Diversity": 1.0,
        "Avg_Basket_Size": 1.0, "Return_Rate": 0.0,
        "Monetary_Percentile": 0.02, "Max_Single_Order": 5.0,
    }


@pytest.fixture
def extreme_features():
    """A wildly unusual customer: every size feature far outside the normal range."""
    return {
        "Recency": 5000.0, "Frequency": 500.0, "Monetary": 50000.0,
        "Prob_Pred_Txn": 100.0, "Prob_Pred_Val": 50000.0, "Prob_Alive": 0.99,
        "Interpurchase_Std": 2000.0, "Purchase_Rate": 5.0,
        "Days_Since_Purchase": 3000.0, "Revenue_Per_Day": 5000.0,
        "Unique_Products": 3000.0, "Visit_Diversity": 400.0,
        "Avg_Basket_Size": 500.0, "Return_Rate": 0.95,
        "Monetary_Percentile": 1.0, "Max_Single_Order": 60000.0,
    }


@pytest.fixture(scope="session")
def api_client():
    """
    FastAPI test client.

    The 'with' block runs the app's startup (lifespan) code, which loads the three
    model bundles. We skip only when fastapi/httpx are genuinely not installed; any
    other problem (a bug in api/main.py, a missing library) makes the tests FAIL
    instead of being silently skipped.
    """
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    os.environ["CLV_BASE_DIR"] = str(_PROJECT)

    # Import fresh — clear any cached module state
    if "api.main" in sys.modules:
        del sys.modules["api.main"]

    from api.main import app, registry

    with TestClient(app, raise_server_exceptions=False) as client:
        if not registry.load_time:      # startup did not run for some reason
            registry.load()
        yield client


def _raw_if_score(bundle, features):
    """Raw Isolation Forest score (higher = more anomalous) for one customer."""
    cols = bundle["cols_used"]
    row  = pd.DataFrame([{c: features.get(c, 0.0) for c in cols}])
    X    = bundle["scaler"].transform(row.fillna(0))
    return float(-bundle["isolation_forest"].decision_function(X)[0])


def _cluster_of(bundle, features):
    """Cluster number of one customer, using the saved scaler, PCA and K-Means."""
    cols = bundle["cols_used"]
    row  = pd.DataFrame([features])
    X    = bundle["scaler"].transform(row[[c for c in cols if c in row.columns]].fillna(0))
    if bundle.get("pca") is not None:
        X = bundle["pca"].transform(X)
    return int(bundle["kmeans"].predict(X)[0])


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
        """All required config constants must be importable and valid."""
        from src.config import (
            RANDOM_SEED, SPLIT_DAYS, FEATURE_COLS,
            GRAPHS_DIR, MODELS_DIR, MODEL_VERSION,
        )
        assert RANDOM_SEED == 42
        assert SPLIT_DAYS  == 90
        assert isinstance(MODEL_VERSION, str) and MODEL_VERSION

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
# 2. Sample feature rows (the 16-column input contract)
# ===========================================================================

class TestFeatureRows:
    """These check the sample customers used by the other tests. The data-cleaning and
    feature-engineering CODE is tested in test_feature_engineering.py."""

    def test_feature_row_shape(self, sample_features):
        """A sample customer must fit the (1, 16) model input."""
        from src.config import FEATURE_COLS
        row = pd.DataFrame([sample_features])[FEATURE_COLS]
        assert row.shape == (1, 16), f"Expected (1,16), got {row.shape}"

    def test_feature_cols_order(self, sample_features):
        """DataFrame columns must match FEATURE_COLS exactly."""
        from src.config import FEATURE_COLS
        row = pd.DataFrame([sample_features])[FEATURE_COLS]
        assert list(row.columns) == FEATURE_COLS

    def test_no_null_values(self, sample_features, whale_features, churned_features):
        """Sample customers must contain no nulls."""
        from src.config import FEATURE_COLS
        rows = pd.DataFrame([sample_features, whale_features, churned_features])[FEATURE_COLS]
        assert not rows.isnull().any().any(), "Null values found in feature rows"

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
        """CLV bundle must contain 'model' and 'version', and the version must match config."""
        from src.config import MODEL_VERSION
        assert "model" in clv_bundle
        assert "version" in clv_bundle
        assert clv_bundle["version"] == MODEL_VERSION

    def test_model_has_predict(self, clv_model):
        """Champion model must have a predict() method."""
        assert hasattr(clv_model, "predict"), "Model missing predict() method"

    def test_prediction_is_scalar(self, clv_model, sample_features):
        """predict() must return a single value for a single-row input."""
        from src.config import FEATURE_COLS
        X = pd.DataFrame([sample_features])[FEATURE_COLS]
        pred = clv_model.predict(X)
        assert len(pred) == 1, f"Expected 1 prediction, got {len(pred)}"

    def test_raw_prediction_is_finite_and_in_log_range(self, clv_model, sample_features):
        """The RAW model output (before any clipping by the caller) must be finite and in [0, LOG_PRED_MAX]."""
        from src.config import FEATURE_COLS
        from src.modeling import LOG_PRED_MAX
        X    = pd.DataFrame([sample_features])[FEATURE_COLS]
        pred = float(clv_model.predict(X)[0])
        assert np.isfinite(pred), f"Non-finite prediction: {pred}"
        assert 0.0 <= pred <= LOG_PRED_MAX, f"Raw log prediction out of range: {pred}"

    def test_whale_higher_than_churned(self, clv_model, whale_features, churned_features):
        """Whale customer must have higher predicted CLV than churned customer."""
        from src.config import FEATURE_COLS
        X_whale   = pd.DataFrame([whale_features])[FEATURE_COLS]
        X_churned = pd.DataFrame([churned_features])[FEATURE_COLS]
        pred_whale   = float(np.expm1(clv_model.predict(X_whale)[0]))
        pred_churned = float(np.expm1(clv_model.predict(X_churned)[0]))
        assert pred_whale > pred_churned, \
            f"Whale ({pred_whale:.2f}) not greater than churned ({pred_churned:.2f})"

    def test_churned_customer_is_low_value(self, clv_model, churned_features):
        """A clearly lapsed customer must land in the Low or Churned tier (< $150)."""
        from src.config import FEATURE_COLS
        X      = pd.DataFrame([churned_features])[FEATURE_COLS]
        dollar = float(np.expm1(clv_model.predict(X)[0]))
        assert dollar < 150, f"Churned customer predicted too high: ${dollar:.2f}"

    def test_whale_customer_is_high_value(self, clv_model, whale_features):
        """A clearly high-value customer must be predicted well above the typical customer."""
        from src.config import FEATURE_COLS
        X      = pd.DataFrame([whale_features])[FEATURE_COLS]
        dollar = float(np.expm1(clv_model.predict(X)[0]))
        assert dollar >= 500, f"Whale customer predicted too low: ${dollar:.2f}"

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
        required = ["kmeans", "scaler", "pca", "cols_used", "optimal_k", "metrics", "segment_labels"]
        for key in required:
            assert key in seg_bundle, f"Missing key in seg_bundle: {key}"

    def test_optimal_k(self, seg_bundle):
        """Optimal k must be a positive integer."""
        k = seg_bundle["optimal_k"]
        assert isinstance(k, (int, np.integer)) and k > 0, f"Invalid optimal_k: {k}"

    def test_segment_names_cover_every_cluster(self, seg_bundle):
        """Every cluster number 0..k-1 must have a segment name in the bundle."""
        k      = int(seg_bundle["optimal_k"])
        labels = seg_bundle["segment_labels"]
        missing = [c for c in range(k) if c not in labels]
        assert not missing, f"No segment name for clusters {missing}"

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
        """The saved scaler, PCA and K-Means must turn a customer into a valid cluster number."""
        k       = int(seg_bundle["optimal_k"])
        cluster = _cluster_of(seg_bundle, sample_features)
        assert 0 <= cluster < k, f"Cluster ID {cluster} out of range [0, {k})"

    def test_whale_and_churned_land_in_different_clusters(self, seg_bundle, whale_features, churned_features):
        """Two completely different customers must not be put into the same segment."""
        c_whale   = _cluster_of(seg_bundle, whale_features)
        c_churned = _cluster_of(seg_bundle, churned_features)
        assert c_whale != c_churned, \
            f"Whale and churned customers share cluster {c_whale}; the segmentation does not separate them"


# ===========================================================================
# 5. Anomaly Detection
# ===========================================================================

class TestAnomalyDetection:

    def test_bundle_keys(self, anomaly_bundle):
        """Anomaly bundle must contain required keys."""
        required = ["isolation_forest", "scaler", "cols_used", "thresholds", "metrics"]
        for key in required:
            assert key in anomaly_bundle, f"Missing key in anomaly_bundle: {key}"

    @pytest.mark.xfail(
        reason="bundles trained before the normalisation fix have no 'norm'; passes after the retrain",
        strict=False,
    )
    def test_bundle_has_training_normalisation_values(self, anomaly_bundle):
        """The bundle must carry the min/max used to scale scores at training time."""
        norm = anomaly_bundle.get("norm") or {}
        assert norm.get("if_min") is not None and norm.get("if_max") is not None
        assert norm["if_max"] > norm["if_min"]

    def test_if_model_has_predict(self, anomaly_bundle):
        """Isolation Forest must have predict() and decision_function()."""
        if_model = anomaly_bundle["isolation_forest"]
        assert hasattr(if_model, "predict")
        assert hasattr(if_model, "decision_function")

    def test_raw_score_is_finite(self, anomaly_bundle, sample_features):
        """The raw Isolation Forest score of a normal customer must be a finite number."""
        raw = _raw_if_score(anomaly_bundle, sample_features)
        assert np.isfinite(raw), f"Non-finite anomaly score: {raw}"

    def test_extreme_customer_scores_higher_than_normal(self, anomaly_bundle, sample_features, extreme_features):
        """A wildly unusual customer must look MORE anomalous than a typical one."""
        raw_normal  = _raw_if_score(anomaly_bundle, sample_features)
        raw_extreme = _raw_if_score(anomaly_bundle, extreme_features)
        assert raw_extreme > raw_normal, \
            f"Extreme customer ({raw_extreme:.4f}) is not more anomalous than normal ({raw_normal:.4f})"

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

    # ---------------- health ----------------

    def test_health_status_200(self, api_client):
        """GET /health must return 200 when all models are loaded."""
        r = api_client.get("/health")
        assert r.status_code == 200, r.text

    def test_health_response_structure(self, api_client):
        """GET /health must return status, api_version, models, timestamp."""
        data = api_client.get("/health").json()
        for field in ("status", "api_version", "models", "timestamp"):
            assert field in data, f"Missing field: {field}"

    def test_health_all_models_loaded(self, api_client):
        """GET /health must report all 3 models as loaded."""
        models = api_client.get("/health").json()["models"]
        for name in ["clv", "segmentation", "anomaly"]:
            assert name in models, f"Model '{name}' missing from health response"
            assert models[name].get("loaded") is True, \
                f"Model '{name}' not loaded: {models[name]}"

    def test_health_returns_503_when_a_model_is_missing(self, api_client, monkeypatch):
        """A degraded service must answer 503, so Docker and load balancers notice."""
        import api.main as api_main
        monkeypatch.setitem(api_main.registry.status, "clv", {"loaded": False, "error": "simulated"})
        r = api_client.get("/health")
        assert r.status_code == 503
        assert r.json()["status"] == "degraded"

    # ---------------- /predict-clv ----------------

    def test_predict_clv_200(self, api_client):
        """POST /predict-clv must return 200 with valid payload."""
        r = api_client.post("/predict-clv", json={
            "Recency": 180, "Frequency": 8, "Monetary": 450
        })
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"

    def test_predict_clv_response_fields(self, api_client):
        """POST /predict-clv response must contain all required fields."""
        data = api_client.post("/predict-clv", json={"Recency": 180, "Monetary": 450}).json()
        for field in ["predicted_clv_90d", "clv_low", "clv_high",
                      "segment", "segment_key", "model_version", "features_used"]:
            assert field in data, f"Missing field in CLV response: {field}"

    def test_predict_clv_non_negative(self, api_client):
        """POST /predict-clv must return non-negative CLV."""
        dollar = api_client.post("/predict-clv", json={"Recency": 180, "Monetary": 450}).json()["predicted_clv_90d"]
        assert dollar >= 0.0, f"Negative CLV: {dollar}"

    def test_predict_clv_range_is_plus_minus_15_percent(self, api_client):
        """The range is a fixed +-15% band (documented as indicative, not a confidence interval)."""
        data = api_client.post("/predict-clv", json={"Monetary": 450}).json()
        pred = data["predicted_clv_90d"]
        assert data["clv_low"] <= pred <= data["clv_high"]
        assert data["clv_low"]  == pytest.approx(pred * 0.85, abs=0.02)
        assert data["clv_high"] == pytest.approx(pred * 1.15, abs=0.02)

    def test_predict_clv_empty_body(self, api_client):
        """POST /predict-clv with empty body must still return 200 (all defaults)."""
        r = api_client.post("/predict-clv", json={})
        assert r.status_code == 200

    def test_predict_clv_uses_supplied_values(self, api_client):
        """Values sent by the caller must be the values the model sees (and defaults fill the rest)."""
        data = api_client.post("/predict-clv", json={"Recency": 90}).json()
        assert data["features_used"]["Recency"] == 90.0
        assert data["features_used"]["Frequency"] == 4.0      # default

    def test_predict_clv_high_value_customer(self, api_client, whale_features):
        """A full high-value customer must be predicted as clearly valuable."""
        data = api_client.post("/predict-clv", json=whale_features).json()
        assert data["predicted_clv_90d"] >= 500, data

    @pytest.mark.parametrize("payload", [
        {"Return_Rate": 2},          # above its maximum of 1
        {"Recency": -1},             # negative
        {"Prob_Alive": 1.5},         # a probability above 1
        {"Monetary": "abc"},         # not a number
    ])
    def test_predict_clv_rejects_invalid_values(self, api_client, payload):
        """Invalid values must be rejected with HTTP 422, before any model code runs."""
        r = api_client.post("/predict-clv", json=payload)
        assert r.status_code == 422, f"{payload} -> {r.status_code}"

    def test_predict_clv_rejects_unknown_field(self, api_client):
        """A mistyped (e.g. lower-case) field name must return 422, not be silently ignored."""
        r = api_client.post("/predict-clv", json={"recency": 90})
        assert r.status_code == 422

    # ---------------- /segment-customer ----------------

    def test_segment_customer_200(self, api_client):
        """POST /segment-customer must return 200."""
        r = api_client.post("/segment-customer", json={
            "Recency": 180, "Frequency": 8, "Monetary": 450
        })
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"

    def test_segment_customer_fields(self, api_client):
        """POST /segment-customer must return cluster_id, segment_name, optimal_k."""
        data = api_client.post("/segment-customer", json={"Monetary": 450}).json()
        for field in ["cluster_id", "segment_name", "optimal_k", "model_version"]:
            assert field in data, f"Missing field: {field}"

    def test_segment_cluster_id_valid(self, api_client):
        """cluster_id must be a non-negative integer less than optimal_k."""
        data = api_client.post("/segment-customer", json={"Monetary": 450}).json()
        assert isinstance(data["cluster_id"], int)
        assert 0 <= data["cluster_id"] < data["optimal_k"]

    # ---------------- /detect-anomaly ----------------

    def test_detect_anomaly_200(self, api_client):
        """POST /detect-anomaly must return 200."""
        r = api_client.post("/detect-anomaly", json={
            "Recency": 180, "Return_Rate": 0.05, "Monetary": 450
        })
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"

    def test_detect_anomaly_fields(self, api_client):
        """POST /detect-anomaly must return all required fields."""
        data = api_client.post("/detect-anomaly", json={"Return_Rate": 0.05}).json()
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
        flags = api_client.post("/detect-anomaly", json={}).json()["flags"]
        for k, v in flags.items():
            assert isinstance(v, bool), f"Flag {k} is not bool: {v}"

    def test_detect_anomaly_risk_levels(self, api_client):
        """risk_level must be one of the three defined values."""
        data = api_client.post("/detect-anomaly", json={}).json()
        assert data["risk_level"] in ["🔴 High", "🟡 Medium", "🟢 Low"], \
            f"Unexpected risk_level: {data['risk_level']}"

    def test_whale_flag_uses_percentile_between_0_and_1(self, api_client):
        """is_whale must be True above the 90th percentile (0.90) and False below it."""
        high = api_client.post("/detect-anomaly", json={"Monetary_Percentile": 0.95}).json()
        low  = api_client.post("/detect-anomaly", json={"Monetary_Percentile": 0.50}).json()
        assert high["flags"]["is_whale"] is True
        assert low["flags"]["is_whale"] is False

    def test_high_return_flag_is_consistent_with_score(self, api_client):
        """is_high_return must equal (Return_Rate > 0.30 AND score > 0.40), whatever the score is."""
        data = api_client.post("/detect-anomaly", json={
            "Return_Rate": 0.90, "Monetary": 50.0, "Recency": 10.0,
        }).json()
        expected = bool(0.90 > 0.30 and data["anomaly_score"] > 0.40)
        assert data["is_high_return"] is expected

    def test_low_return_customer_is_never_high_return(self, api_client):
        """A customer with a low return rate can never be flagged as high-return."""
        data = api_client.post("/detect-anomaly", json={"Return_Rate": 0.01}).json()
        assert data["is_high_return"] is False

    def test_anomaly_score_equals_if_score_without_autoencoder(self, api_client):
        """Live scoring is Isolation-Forest only, so the two scores are identical."""
        data = api_client.post("/detect-anomaly", json={"Return_Rate": 0.05}).json()
        assert data["anomaly_score"] == data["if_score"]

    def test_detect_anomaly_rejects_unknown_field(self, api_client):
        """The anomaly endpoint must also reject unknown field names."""
        r = api_client.post("/detect-anomaly", json={"return_rate": 0.5})
        assert r.status_code == 422

    # ---------------- security & docs ----------------

    def test_api_key_is_enforced_when_configured(self, api_client, monkeypatch):
        """With an API key configured, POST endpoints need X-API-Key and /health stays open."""
        import api.main as api_main
        monkeypatch.setattr(api_main, "API_KEY", "secret-test-key")

        assert api_client.post("/predict-clv", json={}).status_code == 401
        assert api_client.post("/predict-clv", json={}, headers={"X-API-Key": "wrong"}).status_code == 401
        ok = api_client.post("/predict-clv", json={}, headers={"X-API-Key": "secret-test-key"})
        assert ok.status_code == 200
        assert api_client.get("/health").status_code == 200

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