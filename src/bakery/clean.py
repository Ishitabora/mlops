"""Cleaning: turns the raw train.csv into a table that is safe to model on."""
import numpy as np
import pandas as pd


def clean_train(train, issues, params):
    """Fix every problem that `checks.run_all` found. Returns a new DataFrame."""
    d = params["data"]
    df = train.copy()

    if issues["duplicate_rows"]["present"]:
        df = df.drop_duplicates().reset_index(drop=True)

    if issues["leaking_column"]["present"]:
        df = df.drop(columns=issues["leaking_column"]["column"])

    u = issues["unit_mixup"]
    if u["present"]:
        rows = df[d["group_col"]] == u["group_value"]
        df.loc[rows, u["column"]] = df.loc[rows, u["column"]] / u["factor"]

    s = issues["sentinel_value"]
    if s["present"]:
        df.loc[df[s["column"]] == s["value"], s["column"]] = np.nan

    k = issues["stuck_sensor"]
    if k["present"]:
        in_run = (df[d["date_col"]] >= k["start_date"]) & (df[d["date_col"]] <= k["end_date"])
        df.loc[in_run & (df[k["column"]] == k["value"]), k["column"]] = np.nan

    t = issues["target_typos"]
    if t["present"]:
        big = df[d["target"]] > t["threshold"]
        df.loc[big, d["target"]] = df.loc[big, d["target"]] / 10

    sp = issues["spelling_variants"]
    if sp["present"]:
        df[sp["column"]] = df[sp["column"]].str.strip().str.lower()

    # blanks left by the sentinel / stuck-sensor fixes: fill with the column median
    for col in d["numeric_features"]:
        df[col] = df[col].fillna(df[col].median())
    return df
