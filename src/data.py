"""Loading the raw Kaggle CSV files and aggregating daily weather to weekly values.

The raw files hold one row per county (``fips``) per day. The USDM drought
``score`` is only filled in once a week (on the day the map is valid), so we
group every day into the week that ends on the next scored day.

The training file is about 2.2 GB, so it is read in chunks and only the
selected counties are kept in memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, SEED, SPLITS, WEATHER_COLS, WEEKLY_AGG

DTYPES = {col: "float32" for col in WEATHER_COLS}
DTYPES.update({"fips": "int32", "score": "float32"})


def find_split_file(data_dir: Path, split: str) -> Path:
    """Return the CSV path for a split.

    Kaggle's zip unpacks to ``<split>_timeseries/<split>_timeseries.csv``, but a
    flat ``<split>_timeseries.csv`` is accepted too.
    """
    data_dir = Path(data_dir)
    candidates = [
        data_dir / f"{split}_timeseries" / f"{split}_timeseries.csv",
        data_dir / f"{split}_timeseries.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find the {split} file in {data_dir}. "
        "Download the dataset first (see README)."
    )


def load_soil(data_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Static soil and land characteristics, one row per county."""
    return pd.read_csv(Path(data_dir) / "soil_data.csv")


def select_counties(data_dir: Path = RAW_DIR, max_counties: int | None = None,
                    seed: int = SEED) -> np.ndarray | None:
    """Pick a random, reproducible subset of counties (None = all counties)."""
    if max_counties is None:
        return None
    fips = load_soil(data_dir)["fips"].to_numpy()
    rng = np.random.default_rng(seed)
    size = min(max_counties, len(fips))
    return np.sort(rng.choice(fips, size=size, replace=False))


def aggregate_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily rows to one row per county and USDM week.

    Each day is assigned to the next date that has a drought score. Days after
    the last scored date of a county have no label and are dropped.
    """
    daily = daily.sort_values(["fips", "date"])
    week_end = daily["date"].where(daily["score"].notna())
    daily = daily.assign(week_end=week_end.groupby(daily["fips"]).bfill())
    daily = daily.dropna(subset=["week_end"])

    agg = dict(WEEKLY_AGG)
    agg["score"] = "last"          # the only non-missing score in the week
    weekly = (
        daily.groupby(["fips", "week_end"], sort=True)
        .agg(agg)
        .reset_index()
        .rename(columns={"week_end": "date"})
    )
    return weekly


def load_weekly_split(path: Path, fips_subset: np.ndarray | None = None,
                      chunksize: int = 1_000_000) -> pd.DataFrame:
    """Read one split file in chunks and return weekly aggregates.

    Rows are sorted by county, so the last county of each chunk is held back
    and joined with the next chunk to avoid splitting a county in two.
    """
    parts, carry = [], None
    reader = pd.read_csv(path, chunksize=chunksize, dtype=DTYPES, parse_dates=["date"])
    for chunk in reader:
        if fips_subset is not None:
            chunk = chunk[chunk["fips"].isin(fips_subset)]
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = None
        if chunk.empty:
            continue
        last_fips = chunk["fips"].iloc[-1]
        carry = chunk[chunk["fips"] == last_fips]
        done = chunk[chunk["fips"] != last_fips]
        if not done.empty:
            parts.append(aggregate_weekly(done))
    if carry is not None and not carry.empty:
        parts.append(aggregate_weekly(carry))
    return pd.concat(parts, ignore_index=True)


def build_weekly_dataset(data_dir: Path = RAW_DIR, max_counties: int | None = None,
                         seed: int = SEED, use_cache: bool = True) -> pd.DataFrame:
    """Weekly data for all three splits, with a ``split`` column.

    The result is cached in ``data/processed`` so the slow CSV pass only
    runs once per county selection.
    """
    tag = "all" if max_counties is None else f"{max_counties}c_seed{seed}"
    cache = PROCESSED_DIR / f"weekly_{tag}.pkl"
    if use_cache and cache.exists():
        print(f"Loading cached weekly data from {cache}")
        return pd.read_pickle(cache)

    fips_subset = select_counties(data_dir, max_counties, seed)
    frames = []
    for split in SPLITS:
        path = find_split_file(data_dir, split)
        print(f"Reading {path.name} ...")
        weekly = load_weekly_split(path, fips_subset)
        weekly["split"] = split
        frames.append(weekly)
        print(f"  {len(weekly):,} weekly rows, {weekly['fips'].nunique()} counties")

    data = pd.concat(frames, ignore_index=True).sort_values(["fips", "date"])
    data = data.reset_index(drop=True)

    if use_cache:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        data.to_pickle(cache)
        print(f"Cached weekly data to {cache}")
    return data
