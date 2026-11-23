"""Run the 7 checks on train.csv, print the findings, and write the cleaned file.

    python -m bakery.run_data_checks        (from the src/ folder, or with PYTHONPATH=src)
"""
import json
from pathlib import Path

import pandas as pd
import yaml

from bakery.checks import run_all
from bakery.clean import clean_train


def main(params_path="params.yaml"):
    params = yaml.safe_load(open(params_path))
    d = params["data"]
    train = pd.read_csv(d["train"])  # read-only: the raw file is never rewritten
    later = pd.read_csv(d["later"])
    holdout = pd.read_csv(d["holdout"])

    issues = run_all(train, later, holdout, params)
    print(json.dumps(issues, indent=2, default=str))

    cleaned = clean_train(train, issues, params)
    out = Path(params["checks"]["clean_data_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out, index=False)
    print(f"\ncleaned rows: {len(cleaned)}  ->  {out}")
    return issues, cleaned


if __name__ == "__main__":
    main()
