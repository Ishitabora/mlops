"""Monitoring, drift detection (PSI), retraining, and holdout predictions (MP213).

Computes 10-bin PSI on all numeric features between cleaned train.csv and later.csv.
Evaluates v1 subgroup performance across outlets.
Retrains v2 on cleaned train.csv + first 70% of later.csv.
Applies promotion rule (>=12% improvement on last 30% check set).
Generates predictions/holdout_predictions.csv using the @champion model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
import yaml

from bakery.train import (
    build_pipeline,
    compute_file_sha256,
    create_candidate_model,
    load_and_clean_data,
)


def compute_psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    num_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute Population Stability Index (PSI) between expected and actual distributions."""
    exp_arr = np.asarray(expected, dtype=float)
    act_arr = np.asarray(actual, dtype=float)

    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(exp_arr, percentiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) <= 2:
        bin_edges = np.linspace(exp_arr.min(), exp_arr.max(), num_bins + 1)

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    exp_counts, _ = np.histogram(exp_arr, bins=bin_edges)
    act_counts, _ = np.histogram(act_arr, bins=bin_edges)

    exp_pct = exp_counts / len(exp_arr)
    act_pct = act_counts / len(act_arr)

    # Smooth zero bins to avoid division by zero or log(0)
    exp_pct = np.where(exp_pct == 0, epsilon, exp_pct)
    act_pct = np.where(act_pct == 0, epsilon, act_pct)

    exp_pct = exp_pct / exp_pct.sum()
    act_pct = act_pct / act_pct.sum()

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(round(float(psi_val), 4))


def compute_all_psi(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    numeric_features: list[str],
    num_bins: int = 10,
) -> Tuple[dict[str, float], str]:
    """Compute PSI for all numeric features and return the most drifted feature."""
    psi_dict: dict[str, float] = {}
    for col in numeric_features:
        psi = compute_psi(train_df[col], later_df[col], num_bins=num_bins)
        psi_dict[col] = psi

    most_drifted = max(psi_dict, key=lambda k: psi_dict[k])
    return psi_dict, most_drifted


def compute_group_mae(
    later_df: pd.DataFrame,
    model: Pipeline,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
) -> Tuple[dict[str, float], str]:
    """Compute MAE of model on later.csv separately for each group."""
    groups = list(later_df[group_col].unique())
    group_mae: dict[str, float] = {}

    for g in groups:
        sub = later_df[later_df[group_col] == g]
        y_true = sub[target_col]
        y_pred = model.predict(sub[feature_cols])
        mae = float(round(float(mean_absolute_error(y_true, y_pred)), 4))
        group_mae[str(g)] = mae

    worst_group = max(group_mae, key=lambda k: group_mae[k])
    return group_mae, worst_group


def evaluate_promotion(
    mae_v1: float,
    mae_v2: float,
    threshold: float = 0.12,
) -> Tuple[str, float]:
    """Apply the client's promotion rule: promote if v2 MAE is >= 12% lower than v1 MAE."""
    relative_improvement = (mae_v1 - mae_v2) / mae_v1
    decision = "promote" if relative_improvement >= threshold else "keep"
    return decision, relative_improvement


def run_monitor(params_path: str = "params.yaml") -> dict[str, Any]:
    """Run complete monitoring, retraining, promotion evaluation, and holdout prediction."""
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    data_cfg = params["data"]
    model_cfg = params["model"]
    mon_cfg = model_cfg.get("monitoring", {})

    numeric_features = data_cfg["numeric_features"]
    categorical_features = data_cfg["categorical_features"]
    feature_cols = numeric_features + categorical_features
    target_col = data_cfg["target"]
    group_col = data_cfg["group_col"]
    date_col = data_cfg["date_col"]

    tracking_uri = model_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    reg_name = model_cfg.get("registered_model_name", "bakery_bread_model")
    client = MlflowClient()

    # Load cleaned train and clean later data
    cleaned_train, data_sha256 = load_and_clean_data(params)
    later = pd.read_csv(data_cfg["later"])  # later is clean already; do not run clean_train

    # 1. Compute PSI for every numeric feature
    num_bins = mon_cfg.get("psi_bins", 10)
    psi_results, most_drifted = compute_all_psi(cleaned_train, later, numeric_features, num_bins=num_bins)
    print("\n--- Population Stability Index (PSI) ---")
    for feat, psi in psi_results.items():
        print(f"  {feat}: {psi:.4f}")
    print(f"Most drifted feature: {most_drifted}")

    # Load v1 model from models/model.joblib
    model_path = Path(model_cfg.get("model_path", "models/model.joblib"))
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Train v1 first.")
    v1_pipeline = joblib.load(model_path)

    # 2. Compute v1's MAE per outlet on later.csv
    group_mae, worst_group = compute_group_mae(later, v1_pipeline, feature_cols, target_col, group_col)
    print("\n--- v1 MAE by Outlet on later.csv ---")
    for g, mae_val in group_mae.items():
        print(f"  {g}: {mae_val:.4f}")
    print(f"Worst served group: {worst_group}")

    # 3. Retrain v2 on cleaned train + first 70% of later.csv
    later_sorted = later.sort_values(date_col).reset_index(drop=True)
    n_later = len(later_sorted)
    split_ratio = mon_cfg.get("later_train_split", 0.70)
    n_split = int(np.floor(split_ratio * n_later))

    later_train = later_sorted.iloc[:n_split]
    later_check = later_sorted.iloc[n_split:]
    print(f"\nSplit later.csv ({n_later} rows): first {len(later_train)} rows for train, last {len(later_check)} rows for check.")

    X_train_v2 = pd.concat([cleaned_train[feature_cols], later_train[feature_cols]], ignore_index=True)
    y_train_v2 = pd.concat([cleaned_train[target_col], later_train[target_col]], ignore_index=True)

    # Candidate configuration for v2: gradient boosting or best candidate
    cand_cfg = {"name": "gradient_boosting", "n_estimators": 100, "learning_rate": 0.05, "max_depth": 4}
    _, regressor_v2 = create_candidate_model(cand_cfg, random_state=model_cfg.get("random_state", 42))
    v2_pipeline = build_pipeline(regressor_v2, numeric_features, categorical_features)

    exp_name = model_cfg.get("experiment_name", "bakery_bread_sales")
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name="v2_retrain") as run:
        v2_run_id = run.info.run_id
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_version", "v2")
        mlflow.log_param("train_rows", len(X_train_v2))
        mlflow.log_params(cand_cfg)

        v2_pipeline.fit(X_train_v2, y_train_v2)

        check_preds_v1 = v1_pipeline.predict(later_check[feature_cols])
        check_preds_v2 = v2_pipeline.predict(later_check[feature_cols])

        mae_v1_on_check = float(round(float(mean_absolute_error(later_check[target_col], check_preds_v1)), 4))
        mae_v2_on_check = float(round(float(mean_absolute_error(later_check[target_col], check_preds_v2)), 4))

        mlflow.log_metric("mae_v1_on_check", mae_v1_on_check)
        mlflow.log_metric("mae_v2_on_check", mae_v2_on_check)
        mlflow.log_metric("val_mae", mae_v2_on_check)

        mlflow.sklearn.log_model(
            sk_model=v2_pipeline,
            artifact_path="model",
            serialization_format="cloudpickle",
        )

    # Register v2 in the model registry (Version 2)
    model_uri_v2 = f"runs:/{v2_run_id}/model"
    mv2 = mlflow.register_model(model_uri=model_uri_v2, name=reg_name)
    v2_version = str(mv2.version)
    print(f"Registered v2 model as version {v2_version}")

    # 4. Promotion decision: >= 12% improvement
    threshold = mon_cfg.get("promotion_threshold", 0.12)
    decision, rel_improvement = evaluate_promotion(mae_v1_on_check, mae_v2_on_check, threshold)
    print("\n--- Retraining & Promotion Evaluation ---")
    print(f"  v1 MAE on check (last 30%): {mae_v1_on_check:.4f}")
    print(f"  v2 MAE on check (last 30%): {mae_v2_on_check:.4f}")
    print(f"  Relative improvement: {rel_improvement * 100:.2f}% (Threshold: {threshold * 100:.1f}%)")
    print(f"  Decision: {decision.upper()}")

    if decision == "promote":
        champion_version = int(v2_version)
        champion_pipeline = v2_pipeline
        client.set_registered_model_alias(name=reg_name, alias="champion", version=v2_version)
        joblib.dump(champion_pipeline, model_path)
        print(f"Promoted v2! Alias @champion set to version {v2_version}. Updated {model_path}")
    else:
        champion_version = 1
        champion_pipeline = v1_pipeline
        client.set_registered_model_alias(name=reg_name, alias="champion", version="1")
        print(f"Kept v1! Alias @champion remains on version 1. {model_path} unchanged")

    # 5. Holdout predictions using @champion model
    holdout_path = data_cfg["holdout"]
    holdout_df = pd.read_csv(holdout_path)
    holdout_preds = champion_pipeline.predict(holdout_df[feature_cols])

    preds_out_path = Path(model_cfg.get("predictions_path", "predictions/holdout_predictions.csv"))
    preds_out_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df = pd.DataFrame({
        "record_id": holdout_df["record_id"],
        "prediction": np.round(holdout_preds, 4),
    })
    preds_df.to_csv(preds_out_path, index=False)
    print(f"\nGenerated holdout predictions for {len(preds_df)} records -> {preds_out_path}")

    return {
        "psi": psi_results,
        "most_drifted_feature": most_drifted,
        "mae_by_group": group_mae,
        "worst_served_group": worst_group,
        "mae_v1_on_check": mae_v1_on_check,
        "mae_v2_on_check": mae_v2_on_check,
        "decision": decision,
        "champion_version": champion_version,
    }


if __name__ == "__main__":
    run_monitor()
