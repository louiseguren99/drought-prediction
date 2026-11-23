"""Drought prediction API (FastAPI).

Start it locally from the project root:

    uvicorn app:app --app-dir api --reload

Then open http://127.0.0.1:8000/docs to try it in the browser.

The API receives ONE ready-made row of features (the same columns the model was
trained on) and returns the predicted drought class `horizon` weeks ahead.
The features need up to 26 weeks of history, so they are computed before the
call (see src/features.py), not inside the API.
"""

import logging
import os
import time
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger("drought_api")
logging.basicConfig(level=logging.INFO)

CLASS_NAMES = ["None", "D0", "D1", "D2", "D3", "D4"]

# Where the saved model lives. Docker sets MODEL_PATH; locally we use models/ in the repo root.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model_h1.joblib"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))


def load_bundle():
    """Load {"model", "features", "horizon", "name"} saved by src/train.py."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        logger.error("Model file not found: %s", MODEL_PATH)
        return None


bundle = load_bundle()


# ---------- Input and output formats (validated by pydantic) ----------

class PredictionRequest(BaseModel):
    features: dict[str, Annotated[float, Field(allow_inf_nan=False)]] = Field(
        ...,
        description="Feature name -> value. Must contain exactly the columns listed by GET /features.",
    )


class PredictionResponse(BaseModel):
    predicted_class: int = Field(..., description="0 = None, 1 = D0, ... 5 = D4")
    predicted_name: str
    probabilities: dict[str, float] = Field(..., description="Probability of each drought class")
    horizon_weeks: int
    model_name: str


# ---------- The app ----------

app = FastAPI(title="Drought Risk Prediction API")


@app.middleware("http")
async def log_latency(request: Request, call_next):
    """Measure how long every request takes (basis for latency monitoring)."""
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-ms"] = f"{ms:.1f}"
    logger.info("%s %s -> %s in %.1f ms", request.method, request.url.path, response.status_code, ms)
    return response


@app.get("/")
def home():
    return {"message": "Drought Risk Prediction API. See /docs for how to use it."}


@app.get("/health")
def health():
    return {
        "status": "OK" if bundle is not None else "MODEL_MISSING",
        "model_loaded": bundle is not None,
        "model_name": bundle["name"] if bundle else None,
        "horizon_weeks": bundle["horizon"] if bundle else None,
    }


@app.get("/features")
def list_features():
    """The exact feature columns (in order) that /predict expects."""
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"n_features": len(bundle["features"]), "features": bundle["features"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    expected = bundle["features"]
    given = set(request.features)
    missing = [name for name in expected if name not in given]
    unknown = sorted(given - set(expected))
    if missing or unknown:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Features do not match what the model was trained on (see GET /features)",
                "missing": missing,
                "unknown": unknown,
            },
        )

    model = bundle["model"]
    row = pd.DataFrame([request.features])[expected]  # same column order as in training

    try:
        probabilities = model.predict_proba(row)[0]
    except Exception as error:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}")

    classes = [int(c) for c in model.classes_]
    best = classes[int(probabilities.argmax())]
    logger.info("prediction: class=%s (%s)", best, CLASS_NAMES[best])

    return PredictionResponse(
        predicted_class=best,
        predicted_name=CLASS_NAMES[best],
        probabilities={CLASS_NAMES[c]: round(float(p), 4) for c, p in zip(classes, probabilities)},
        horizon_weeks=bundle["horizon"],
        model_name=bundle["name"],
    )
