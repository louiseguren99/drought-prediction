"""Feature engineering on the weekly data.

All features at week t use only information available at week t (the
current USDM map, past weather and static soil data). The target is the
drought class ``horizon`` weeks later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    ROLLING_MEAN_COLS,
    ROLLING_SUM_COLS,
    ROLLING_WINDOWS,
    SCORE_LAGS,
)

NON_FEATURE_COLS = {"fips", "date", "split", "target_score", "y"}


def _rolling(group: pd.core.groupby.SeriesGroupBy, window: int, how: str) -> pd.Series:
    rolled = group.rolling(window, min_periods=window)
    rolled = rolled.sum() if how == "sum" else rolled.mean()
    return rolled.reset_index(level=0, drop=True)


def add_features(weekly: pd.DataFrame, soil: pd.DataFrame | None = None,
                 horizon: int = 1) -> pd.DataFrame:
    """Return a modelling table with features, ``target_score`` and class ``y``.

    Lag and rolling features are computed over the full, time-ordered history of
    each county, so the first weeks of validation and test can use the end of the
    previous split as history. The target is shifted *within* each split so a
    training row never carries a label from the validation or test period.
    """
    df = weekly.sort_values(["fips", "date"]).reset_index(drop=True).copy()
    by_county = df.groupby("fips", sort=False)

    # Past drought levels (the current map, score at t, is already a column)
    for lag in SCORE_LAGS:
        df[f"score_lag{lag}"] = by_county["score"].shift(lag)
    df["score_change_1w"] = df["score"] - df["score_lag1"]
    df["score_change_4w"] = df["score"] - df["score_lag4"]

    # Multi-week weather history
    for window in ROLLING_WINDOWS:
        for col in ROLLING_SUM_COLS:
            df[f"{col}_sum_{window}w"] = _rolling(by_county[col], window, "sum")
        for col in ROLLING_MEAN_COLS:
            df[f"{col}_mean_{window}w"] = _rolling(by_county[col], window, "mean")

    # Seasonality
    week = df["date"].dt.isocalendar().week.astype(float)
    df["week_sin"] = np.sin(2 * np.pi * week / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * week / 52.0)

    # Static soil / land features
    if soil is not None:
        df = df.merge(soil, on="fips", how="left")

    # Target: drought score `horizon` weeks ahead, within the same split
    df["target_score"] = df.groupby(["fips", "split"], sort=False)["score"].shift(-horizon)
    df = df.dropna(subset=["target_score"])
    df["y"] = df["target_score"].round().clip(0, 5).astype(int)

    # Drop rows without enough history for the longest window
    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def split_xy(df: pd.DataFrame, split: str):
    part = df[df["split"] == split]
    cols = get_feature_columns(df)
    return part[cols], part["y"].to_numpy(), part
