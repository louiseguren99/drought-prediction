"""Simple leakage tests for src/features.py.

They use small made-up data, so no Kaggle download is needed.
Run from the project root with:  python -m pytest tests -v
"""

import numpy as np
import pandas as pd

from src.config import WEATHER_COLS
from src.features import add_features, get_feature_columns

WEEK = pd.Timedelta(weeks=1)


def make_data():
    """Two counties, 120 weeks each: weeks 0-69 train, 70-94 validation, 95-119 test."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-04", periods=120, freq="7D")
    split = ["train"] * 70 + ["validation"] * 25 + ["test"] * 25

    parts = []
    for fips in [1001, 1003]:
        df = pd.DataFrame({"fips": fips, "date": dates, "split": split})
        df["score"] = rng.integers(0, 6, 120).astype(float)
        for col in WEATHER_COLS:
            df[col] = rng.normal(10, 3, 120)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def test_target_is_next_weeks_score():
    data = make_data()
    out = add_features(data, horizon=1)

    # score of every (county, week), so we can look up next week
    score_at = {(r.fips, r.date): r.score for r in data.itertuples()}

    for row in out.itertuples():
        assert row.target_score == score_at[(row.fips, row.date + WEEK)]


def test_target_stays_in_same_split():
    data = make_data()
    out = add_features(data, horizon=1)

    split_at = {(r.fips, r.date): r.split for r in data.itertuples()}

    for row in out.itertuples():
        assert split_at[(row.fips, row.date + WEEK)] == row.split


def test_lag_is_last_weeks_score():
    data = make_data()
    out = add_features(data, horizon=1)

    score_at = {(r.fips, r.date): r.score for r in data.itertuples()}

    for row in out.itertuples():
        assert row.score_lag1 == score_at[(row.fips, row.date - WEEK)]


def test_no_answer_columns_in_features():
    out = add_features(make_data(), horizon=1)
    features = get_feature_columns(out)

    for bad in ["y", "target_score", "split", "date", "fips"]:
        assert bad not in features


def test_future_does_not_change_features():
    """Change everything after week 80. Features up to week 80 must stay the same."""
    data = make_data()
    cutoff = data["date"].iloc[80]

    changed = data.copy()
    later = changed["date"] > cutoff
    changed.loc[later, "score"] = 5 - changed.loc[later, "score"]
    for col in WEATHER_COLS:
        changed.loc[later, col] = changed.loc[later, col] + 100

    before = add_features(data, horizon=1)
    after = add_features(changed, horizon=1)

    before = before[before["date"] <= cutoff].reset_index(drop=True)
    after = after[after["date"] <= cutoff].reset_index(drop=True)

    cols = get_feature_columns(before)
    pd.testing.assert_frame_equal(before[cols], after[cols])
