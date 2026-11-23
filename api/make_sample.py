"""Create api/examples/sample_request.json from one real validation row.

Run from the project root (needs data/ from the training step):
    python api/make_sample.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RAW_DIR, SEED  # noqa: E402
from src.data import build_weekly_dataset, load_soil  # noqa: E402
from src.features import add_features, get_feature_columns  # noqa: E402

weekly = build_weekly_dataset(RAW_DIR, 300, SEED, use_cache=True)
data = add_features(weekly, load_soil(RAW_DIR), horizon=1)

row = data[data["split"] == "validation"].iloc[1000]
features = {name: float(row[name]) for name in get_feature_columns(data)}

out = ROOT / "api" / "examples" / "sample_request.json"
out.write_text(json.dumps({"features": features}, indent=2))
print(f"Wrote {out}")
print(f"County {row['fips']}, week {row['date'].date()}, true class one week later: {row['y']}")
