"""Download county-level crop yields from USDA NASS Quick Stats.

Used for the secondary analysis that links drought to crop yields.

1. Get a free API key: https://quickstats.nass.usda.gov/api (enter your email).
2. Set it as an environment variable, then run from the project root:

    PowerShell:  $env:NASS_API_KEY = "your-key"
                 py -m src.download_nass
    Mac/Linux:   export NASS_API_KEY=your-key
                 python -m src.download_nass

Output: data/raw/nass/crop_yields.csv with columns
    fips, year, crop, yield_bu_per_acre

The API returns at most 50,000 rows per request, so data are fetched one
crop and one year at a time.
"""

from __future__ import annotations

import argparse
import io
import os
import time

import pandas as pd
import requests

from src.config import RAW_DIR

API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
OUT_DIR = RAW_DIR / "nass"

CROPS = {
    "corn": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
    "soybeans": "SOYBEANS - YIELD, MEASURED IN BU / ACRE",
}


def fetch(api_key: str, short_desc: str, year: int) -> pd.DataFrame:
    params = {
        "key": api_key,
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "short_desc": short_desc,
        "domain_desc": "TOTAL",
        "agg_level_desc": "COUNTY",
        "freq_desc": "ANNUAL",
        "reference_period_desc": "YEAR",
        "year": year,
        "format": "CSV",
    }
    resp = requests.get(API_URL, params=params, timeout=120)
    if resp.status_code == 400 and "no data" in resp.text.lower():
        return pd.DataFrame()
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), dtype=str)


def tidy(raw: pd.DataFrame, crop: str) -> pd.DataFrame:
    """Keep real counties and build the 5-digit FIPS code used by the drought data."""
    df = raw[raw["county_code"].notna() & (raw["county_code"] != "998")].copy()  # 998 = combined counties
    df["fips"] = (df["state_fips_code"].str.zfill(2) + df["county_code"].str.zfill(3)).astype(int)
    # Values are text like "152.3" or "1,052.0"; "(D)" means withheld
    df["yield_bu_per_acre"] = pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce")
    df["year"] = df["year"].astype(int)
    df["crop"] = crop
    return df[["fips", "year", "crop", "yield_bu_per_acre"]].dropna()


def main() -> None:
    p = argparse.ArgumentParser(description="Download NASS county crop yields")
    p.add_argument("--start", type=int, default=2000)
    p.add_argument("--end", type=int, default=2020)
    args = p.parse_args()

    api_key = os.environ.get("NASS_API_KEY")
    if not api_key:
        raise SystemExit("Set the NASS_API_KEY environment variable first (see the docstring).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for crop, short_desc in CROPS.items():
        for year in range(args.start, args.end + 1):
            raw = fetch(api_key, short_desc, year)
            n = 0 if raw.empty else len(raw)
            print(f"{crop:9s} {year}: {n:,} rows")
            if n:
                frames.append(tidy(raw, crop))
            time.sleep(0.5)  # be polite to the API

    yields = pd.concat(frames, ignore_index=True).sort_values(["crop", "fips", "year"])
    out = OUT_DIR / "crop_yields.csv"
    yields.to_csv(out, index=False)
    print(f"\nSaved {len(yields):,} rows for {yields['fips'].nunique():,} counties to {out}")


if __name__ == "__main__":
    main()
