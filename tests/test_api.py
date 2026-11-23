"""Tests for the FastAPI app in api/app.py (they need models/best_model_h1.joblib)."""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

import app as api_module  # noqa: E402

if api_module.bundle is None:
    pytest.skip("No saved model found, run src/train.py first", allow_module_level=True)

client = TestClient(api_module.app)
SAMPLE = json.loads((ROOT / "api" / "examples" / "sample_request.json").read_text())


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "OK"
    assert body["model_loaded"] is True


def test_features_match_the_model():
    body = client.get("/features").json()
    assert body["features"] == api_module.bundle["features"]
    assert body["n_features"] == len(body["features"])


def test_predict_returns_a_valid_class():
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in range(6)
    assert body["predicted_name"] in ["None", "D0", "D1", "D2", "D3", "D4"]
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=0.01)
    assert "X-Process-Time-ms" in response.headers


def test_missing_feature_gives_422():
    broken = {"features": dict(SAMPLE["features"])}
    del broken["features"]["score"]
    response = client.post("/predict", json=broken)
    assert response.status_code == 422
    assert "score" in response.json()["detail"]["missing"]


def test_unknown_feature_gives_422():
    broken = {"features": {**SAMPLE["features"], "not_a_feature": 1.0}}
    response = client.post("/predict", json=broken)
    assert response.status_code == 422
    assert "not_a_feature" in response.json()["detail"]["unknown"]


def test_text_instead_of_number_gives_422():
    broken = {"features": {**SAMPLE["features"], "score": "dry"}}
    assert client.post("/predict", json=broken).status_code == 422
