"""Train and compare models for drought forecasting.

Example (from the project root):

    python -m src.train --max-counties 300 --horizon 1

Models are tuned and compared on the validation split. The test split is only
scored when ``--evaluate-test`` is passed, which should happen once, at the end.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FIGURES_DIR, MODELS_DIR, RAW_DIR, REPORTS_DIR, SEED
from src.data import build_weekly_dataset, load_soil
from src.evaluate import compute_metrics, persistence_predict, plot_confusion_matrix
from src.features import add_features, split_xy


def get_models(seed: int = SEED) -> dict:
    """Candidate models. Hyperparameters here are starting points to tune."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed,
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=RAW_DIR, help="folder with the Kaggle files")
    p.add_argument("--max-counties", type=int, default=300,
                   help="random subset of counties for speed (use 0 for all 3,108)")
    p.add_argument("--horizon", type=int, default=1, help="weeks ahead to forecast")
    p.add_argument("--models", nargs="+", default=None,
                   help="subset of models to train (default: all)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-cache", action="store_true", help="re-read the raw CSV files")
    p.add_argument("--evaluate-test", action="store_true",
                   help="also score the test split (do this once, at the end)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    max_counties = None if args.max_counties == 0 else args.max_counties

    weekly = build_weekly_dataset(args.data_dir, max_counties, args.seed, use_cache=not args.no_cache)
    soil = load_soil(args.data_dir)
    data = add_features(weekly, soil, horizon=args.horizon)

    X_train, y_train, _ = split_xy(data, "train")
    X_val, y_val, _ = split_xy(data, "validation")
    X_test, y_test, _ = split_xy(data, "test")
    print(f"\nRows  train={len(X_train):,}  validation={len(X_val):,}  test={len(X_test):,}")
    print(f"Features: {X_train.shape[1]}  |  horizon: {args.horizon} week(s)")
    print("Train class distribution:", np.bincount(y_train, minlength=6))

    results, fitted = [], {}

    # Baseline
    val_pred = persistence_predict(X_val)
    results.append({"model": "persistence_baseline", "fit_seconds": 0.0, **compute_metrics(y_val, val_pred)})
    plot_confusion_matrix(y_val, val_pred, "Persistence baseline (validation)",
                          FIGURES_DIR / f"cm_persistence_h{args.horizon}.png")

    models = get_models(args.seed)
    if args.models:
        models = {name: models[name] for name in args.models}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        seconds = time.perf_counter() - start
        val_pred = model.predict(X_val)
        results.append({"model": name, "fit_seconds": round(seconds, 1), **compute_metrics(y_val, val_pred)})
        fitted[name] = model
        plot_confusion_matrix(y_val, val_pred, f"{name} (validation)",
                              FIGURES_DIR / f"cm_{name}_h{args.horizon}.png")

    table = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    print("\nValidation results (sorted by macro F1):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / f"validation_results_h{args.horizon}.csv", index=False)

    trained = table[table["model"].isin(fitted)]
    if trained.empty:
        return
    best_name = trained.iloc[0]["model"]
    best_model = fitted[best_name]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"best_model_h{args.horizon}.joblib"
    joblib.dump({"model": best_model, "features": list(X_train.columns),
                 "horizon": args.horizon, "name": best_name}, model_path)
    print(f"\nBest model: {best_name} -> saved to {model_path}")

    if args.evaluate_test:
        test_metrics = {
            "persistence_baseline": compute_metrics(y_test, persistence_predict(X_test)),
            best_name: compute_metrics(y_test, best_model.predict(X_test)),
        }
        test_table = pd.DataFrame(test_metrics).T
        print("\nTest results:")
        print(test_table.to_string(float_format=lambda v: f"{v:.4f}"))
        test_table.to_csv(REPORTS_DIR / f"test_results_h{args.horizon}.csv")


if __name__ == "__main__":
    main()
