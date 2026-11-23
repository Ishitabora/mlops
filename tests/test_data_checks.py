import numpy as np
import pandas as pd

from bakery.checks import (check_duplicate_rows, check_leaking_column, check_sentinel_value,
                           check_stuck_sensor, check_target_typos, check_unit_mixup)
from bakery.clean import clean_train


def small_df():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "record_id": [f"R{i}" for i in range(n)],
        "date": pd.date_range("2026-01-01", periods=n).astype(str),
        "price_rs": rng.integers(30, 60, n).astype(float),
        "orders_previous_day": rng.integers(10, 500, n),
        "outlet": ["A", "B", "C"] * (n // 3),
        "loaves_sold": rng.integers(200, 600, n),
    })
    df["closing"] = df["loaves_sold"] + 3
    return df


def test_leaking_column_found_when_missing_from_holdout():
    df = small_df()
    r = check_leaking_column(df, df.drop(columns=["closing", "loaves_sold"]), "loaves_sold")
    assert r["present"] and r["column"] == "closing"


def test_unit_mixup_finds_group_and_factor():
    df = small_df()
    df.loc[df.outlet == "C", "price_rs"] *= 100
    r = check_unit_mixup(df, ["price_rs"], "outlet")
    assert r["group_value"] == "C" and r["factor"] == 100


def test_sentinel_must_be_outside_normal_range():
    df = small_df()
    assert not check_sentinel_value(df, ["price_rs"])["present"]
    df.loc[:4, "price_rs"] = -999
    assert check_sentinel_value(df, ["price_rs"])["count"] == 5


def test_duplicates_counted_as_extra_copies():
    df = small_df()
    df = pd.concat([df, df.iloc[:4]], ignore_index=True)
    assert check_duplicate_rows(df)["count"] == 4


def test_stuck_sensor_gives_first_and_last_date():
    df = small_df()
    df.loc[10:39, "orders_previous_day"] = 118
    r = check_stuck_sensor(df, ["orders_previous_day"], "date", "loaves_sold", 20)
    assert r["start_date"] == df.date[10] and r["end_date"] == df.date[39]


def test_target_typos_and_cleaning_divides_by_ten():
    df = small_df()
    df.loc[:2, "loaves_sold"] *= 10
    t = check_target_typos(df, "loaves_sold")
    assert t["count"] == 3
    params = {"data": {"group_col": "outlet", "date_col": "date", "target": "loaves_sold",
                       "numeric_features": ["price_rs", "orders_previous_day"]}}
    issues = {"duplicate_rows": {"present": False}, "leaking_column": {"present": True, "column": "closing"},
              "unit_mixup": {"present": False}, "sentinel_value": {"present": False},
              "stuck_sensor": {"present": False}, "target_typos": {"present": True, **t},
              "spelling_variants": {"present": False}}
    out = clean_train(df, issues, params)
    assert out["loaves_sold"].max() <= t["threshold"] and "closing" not in out.columns
