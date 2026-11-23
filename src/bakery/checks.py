"""The seven data checks. Each one looks at the data and returns a dict."""
import numpy as np
import pandas as pd

SENTINELS = [-1, -9, -99, -999, -9999, 99, 999, 9999, 99999]
KNOWN_FACTORS = [10, 12, 60, 100, 1000]


def check_leaking_column(train, holdout, target, later=None):
    """A column present in train but absent from holdout is filled in after the event."""
    candidates = [c for c in train.columns if c not in holdout.columns and c != target]
    if not candidates:
        return {"present": False, "column": None}
    col = candidates[0]
    typo_free = train[train[target] < train[target].median() * 3]
    corr = typo_free[col].corr(typo_free[target])
    return {"present": True, "column": col, "corr_with_target": round(float(corr), 4)}


def check_unit_mixup(train, numeric_cols, group_col, ratio_threshold=5):
    """One group whose column mean is far from the other groups' means."""
    for col in numeric_cols:
        means = train.groupby(group_col)[col].mean()
        typical = means.median()  # robust: one odd group cannot move it
        for group, m in means.items():
            ratio = m / typical
            if ratio > ratio_threshold or ratio < 1 / ratio_threshold:
                big = max(ratio, 1 / ratio)
                factor = min(KNOWN_FACTORS, key=lambda f: abs(f - big))
                if ratio < 1:
                    factor = 1 / factor
                return {"present": True, "column": col, "group_value": group,
                        "factor": factor, "raw_ratio": round(float(ratio), 2)}
    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_sentinel_value(train, numeric_cols):
    """A special number typed instead of a blank (-999, 9999 ...).

    It only counts if it is impossible: outside the range of the column's other values.
    """
    for col in numeric_cols:
        for s in SENTINELS:
            rest = train.loc[train[col] != s, col]
            n = int((train[col] == s).sum())
            if n >= 3 and (s < rest.min() or s > rest.max()):
                return {"present": True, "column": col, "value": s, "count": n}
    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicate_rows(train):
    """Extra copies of a record. Looks both at full rows and ignoring record_id."""
    full = int(train.duplicated().sum())
    no_id = int(train.drop(columns="record_id").duplicated().sum())
    n = max(full, no_id)
    return {"present": n > 0, "count": n if n > 0 else None}


def check_stuck_sensor(train, numeric_cols, date_col, target, min_run_rows=20):
    """The longest run of one repeated value, rows sorted by date."""
    df = train.sort_values(date_col, kind="stable").reset_index(drop=True)
    best = None
    for col in numeric_cols:
        if col == target:
            continue
        run_id = (df[col] != df[col].shift()).cumsum()
        runs = df.groupby(run_id).agg(n=(col, "size"), value=(col, "first"),
                                      start=(date_col, "min"), end=(date_col, "max"))
        top = runs.loc[runs["n"].idxmax()]
        if top["n"] >= min_run_rows and (best is None or top["n"] > best["rows"]):
            best = {"present": True, "column": col, "value": float(top["value"]),
                    "rows": int(top["n"]),
                    "start_date": str(pd.Timestamp(top["start"]).date()),
                    "end_date": str(pd.Timestamp(top["end"]).date())}
    return best or {"present": False, "column": None, "start_date": None, "end_date": None}


def check_target_typos(train, target, gap_ratio=3.0):
    """Targets typed 10 times too big: a cluster above a big empty gap."""
    s = np.sort(train[target].to_numpy(dtype=float))
    ratios = s[1:] / s[:-1]
    i = int(ratios.argmax())
    if ratios[i] < gap_ratio:
        return {"present": False, "count": None}
    threshold = float(s[i])
    return {"present": True, "count": int((train[target] > threshold).sum()),
            "threshold": threshold}


def check_spelling_variants(train, later, text_cols):
    """Raw levels that differ from the clean levels in later.csv."""
    for col in text_cols:
        clean_levels = set(later[col].unique())
        raw_levels = set(train[col].dropna().unique())
        if raw_levels != clean_levels:
            return {"present": True, "column": col, "true_levels": len(clean_levels),
                    "raw_levels": len(raw_levels)}
    return {"present": False, "column": None, "true_levels": None}


def run_all(train, later, holdout, params):
    d, c = params["data"], params["checks"]
    nums = d["numeric_features"] + ["loaves_in_closing_register"]
    return {
        "leaking_column": check_leaking_column(train, holdout, d["target"]),
        "unit_mixup": check_unit_mixup(train, d["numeric_features"], d["group_col"],
                                       c["unit_ratio_threshold"]),
        "sentinel_value": check_sentinel_value(train, nums),
        "duplicate_rows": check_duplicate_rows(train),
        "stuck_sensor": check_stuck_sensor(train, d["numeric_features"], d["date_col"],
                                           d["target"], c["stuck_min_run_rows"]),
        "target_typos": check_target_typos(train, d["target"], c["typo_gap_ratio"]),
        "spelling_variants": check_spelling_variants(train, later, d["categorical_features"]),
    }
