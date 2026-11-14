# Predicting Agricultural Drought Risk

Group 16, 61.501 Production-Ready Machine Learning (SUTD, 2026).

Forecasts the U.S. Drought Monitor (USDM) drought class of a county 1–2 weeks ahead from weather, soil and past drought data, using traditional machine learning only (no deep learning).

## Setup

```bash
python -m pip install -r requirements.txt
```

On Windows, use `py` instead of `python` if `python` is not recognised.

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
api/              # FastAPI demo (to do)
```

## Team

| Name | Contribution |
| --- | --- |
| Louise Guren | |
| Alexander Gombor | |
| Clara Gomer | |
| Linéa Lejerkrans | |
| Joonas Sinerkari | |
