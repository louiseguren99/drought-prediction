"""Central configuration: paths, column names and constants used across the pipeline."""

from pathlib import Path

# Project root = the folder that contains `src/`
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SEED = 42

SPLITS = ("train", "validation", "test")

# The 18 daily meteorological variables from NASA POWER
WEATHER_COLS = [
    "PRECTOT",      # precipitation (mm/day)
    "PS",           # surface pressure (kPa)
    "QV2M",         # specific humidity at 2 m (g/kg)
    "T2M",          # temperature at 2 m (C)
    "T2MDEW",       # dew/frost point at 2 m (C)
    "T2MWET",       # wet bulb temperature at 2 m (C)
    "T2M_MAX",      # max temperature at 2 m (C)
    "T2M_MIN",      # min temperature at 2 m (C)
    "T2M_RANGE",    # temperature range at 2 m (C)
    "TS",           # earth skin temperature (C)
    "WS10M",        # wind speed at 10 m (m/s)
    "WS10M_MAX",
    "WS10M_MIN",
    "WS10M_RANGE",
    "WS50M",        # wind speed at 50 m (m/s)
    "WS50M_MAX",
    "WS50M_MIN",
    "WS50M_RANGE",
]

# How each daily variable is summarised into one weekly value
WEEKLY_AGG = {col: "mean" for col in WEATHER_COLS}
WEEKLY_AGG.update({
    "PRECTOT": "sum",       # total weekly precipitation
    "T2M_MAX": "max",       # hottest day of the week
    "T2M_MIN": "min",       # coldest day of the week
    "WS10M_MAX": "max",
    "WS50M_MAX": "max",
})

# Variables used for rolling-window (multi-week) features
ROLLING_SUM_COLS = ["PRECTOT"]
ROLLING_MEAN_COLS = ["T2M", "T2M_MAX", "QV2M", "T2M_RANGE", "TS", "WS10M"]
ROLLING_WINDOWS = (4, 8, 12, 26)   # weeks
SCORE_LAGS = (1, 2, 3, 4, 8)       # weeks

# Drought classes after rounding the USDM score (0 = none, 1 = D0 ... 5 = D4)
CLASS_NAMES = ["None", "D0", "D1", "D2", "D3", "D4"]
