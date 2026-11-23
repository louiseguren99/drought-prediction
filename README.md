# Predicting Agricultural Drought Risk

Group 16, 61.501 Production-Ready Machine Learning (SUTD, 2026).

Forecasts the U.S. Drought Monitor (USDM) drought class of a county 1–2 weeks ahead from weather, soil and past drought data, using traditional machine learning only (no deep learning).

## Setup

Tested with Python 3.13.1 on macOS. A virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

Dataset: [Predict Droughts using Weather & Soil Data](https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data) (Kaggle, CC0).

1. Create a Kaggle API token (Kaggle → Settings → API) and save it to `~/.kaggle/access_token`.
2. Download into `data/raw/`:

```bash
python -m kaggle datasets download -d cdminix/us-drought-meteorological-data -p data/raw --unzip
```

Expected layout:

```
data/raw/
├── soil_data.csv
├── train_timeseries/train_timeseries.csv        # 2000–2016
├── validation_timeseries/validation_timeseries.csv  # 2017–2018
└── test_timeseries/test_timeseries.csv          # 2019–2020
```

## Run the pipeline

```bash
# Quick run on 300 random counties, forecasting 1 week ahead
python -m src.train --max-counties 300 --horizon 1

# All counties (slow, needs several GB of RAM)
python -m src.train --max-counties 0 --horizon 1

# Final evaluation on the test split (run once, at the end)
python -m src.train --max-counties 300 --horizon 1 --evaluate-test
```

The first run reads the raw CSV files and caches weekly data in `data/processed/`; later runs with the same county selection are fast. Outputs:

- `reports/validation_results_h<horizon>.csv`: metrics for every model
- `reports/figures/cm_<model>_h<horizon>.png`: confusion matrices
- `models/best_model_h<horizon>.joblib`: best model by validation macro F1

All randomness uses seed 42 (`--seed` to change).

## Prediction API

api/app.py serves a saved model with FastAPI. It needs a trained model in models/, so run `python -m src.train` first. By default it loads models/best_model_h1.joblib. Set the MODEL_PATH environment variable to use another file, for example the 2-week model.

The API has its own requirements file with pinned versions, because saved scikit-learn models depend on the library version.

```bash
python -m pip install -r api/requirements.txt
uvicorn app:app --app-dir api --reload
```

Open http://127.0.0.1:8000/docs to try it in the browser.

Endpoints:

- GET /health: shows if the model is loaded.
- GET /features: lists the feature columns that /predict expects.
- POST /predict: takes one row of features as JSON and returns the predicted drought class with class probabilities.

The row must be computed beforehand with src/features.py, because the features need up to 26 weeks of history. It is checked against the feature list saved with the model. Missing or unknown features give a 422 error that lists them.

An example request built from a real validation row is in api/examples/sample_request.json. It can be re-created with `python api/make_sample.py`, which needs the downloaded data in data/raw/.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @api/examples/sample_request.json
```

Every request is logged with its latency, which is also returned in the X-Process-Time-ms response header.

## Docker

The model file is not part of the image (models/ is in .gitignore). It is mounted when the container starts. Docker Desktop must be installed and running. From the project root:

```bash
docker build -f api/Dockerfile -t drought-api .
docker run -p 8000:8000 -v "$(pwd)/models:/models" drought-api
```

The API is then available at http://localhost:8000/docs. To serve the 2-week model, add `-e MODEL_PATH=/models/best_model_h2.joblib` to the docker run command.

## Tests

```bash
python -m pip install -r api/requirements.txt
python -m pytest tests -v
```

tests/test_features.py checks that src/features.py has no data leakage, using small synthetic data. tests/test_api.py tests the API and is skipped if no trained model is found in models/. It needs the packages from api/requirements.txt.

## Pipeline

1. **Weekly aggregation** (`src/data.py`): daily weather is grouped into USDM weeks (each week ends on the day the drought score is valid). Precipitation is summed, other variables averaged, max/min kept for extremes.
2. **Features** (`src/features.py`): current and lagged drought scores, 4/8/12/26-week rolling precipitation and temperature, seasonality, static soil data.
3. **Target**: drought score `horizon` weeks later, rounded to 6 classes (None, D0–D4).
4. **Models** (`src/train.py`): persistence baseline, logistic regression, random forest, histogram gradient boosting.
5. **Evaluation** (`src/evaluate.py`): macro F1 (main metric, robust to class imbalance), accuracy, weighted F1, MAE.

Splits are by time, as provided by the dataset. Features use only past data, and targets never cross a split boundary.

## Project structure

```
src/
├── config.py     # paths, column names, constants
├── data.py       # loading + weekly aggregation
├── features.py   # feature engineering + target
├── evaluate.py   # metrics, baseline, plots
└── train.py      # train/compare models (CLI)
notebooks/        # EDA and analysis
api/
├── app.py             # FastAPI app (/health, /features, /predict)
├── Dockerfile
├── requirements.txt   # pinned versions for the API
├── make_sample.py     # builds examples/sample_request.json
└── examples/          # sample_request.json
tests/
├── test_features.py   # leakage tests for features
└── test_api.py        # tests of the API
```
