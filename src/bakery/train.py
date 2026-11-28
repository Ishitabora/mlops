"""Training pipeline for Bakery Bread Sales (MP213).

Trains candidate regression models on cleaned train.csv, tracks experiments with MLflow,
registers the best performing model as Version 1, assigns the champion alias,
and persists models/model.joblib.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

from bakery.checks import run_all
from bakery.clean import clean_train


def compute_file_sha256(filepath: str | Path) -> str:
    """Compute the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_and_clean_data(params: dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    """Load raw training data, verify hash, run checks, and return clean dataframe."""
    d = params["data"]
    train_path = d["train"]
    data_sha256 = compute_file_sha256(train_path)

    train = pd.read_csv(train_path)
    later = pd.read_csv(d["later"])
    holdout = pd.read_csv(d["holdout"])

    issues = run_all(train, later, holdout, params)
    cleaned = clean_train(train, issues, params)
    return cleaned, data_sha256


def build_pipeline(
    model: Any,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Construct an end-to-end scikit-learn Pipeline with preprocessing."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])


def create_candidate_model(cand_cfg: dict[str, Any], random_state: int) -> Tuple[str, Any]:
    """Instantiate a regressor from candidate configuration."""
    name = cand_cfg["name"]
    if name == "ridge":
        model = Ridge(alpha=cand_cfg.get("alpha", 1.0), random_state=random_state)
    elif name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=cand_cfg.get("n_estimators", 100),
            max_depth=cand_cfg.get("max_depth", 8),
            random_state=random_state,
        )
    elif name == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=cand_cfg.get("n_estimators", 100),
            learning_rate=cand_cfg.get("learning_rate", 0.05),
            max_depth=cand_cfg.get("max_depth", 4),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown candidate model: {name}")
    return name, model


def train_v1(params_path: str = "params.yaml") -> dict[str, Any]:
    """Train candidate models, log to MLflow, register v1, and set @champion alias."""
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    data_cfg = params["data"]
    model_cfg = params["model"]

    cleaned_train, data_sha256 = load_and_clean_data(params)

    numeric_features = data_cfg["numeric_features"]
    categorical_features = data_cfg["categorical_features"]
    feature_cols = numeric_features + categorical_features
    target_col = data_cfg["target"]

    X = cleaned_train[feature_cols]
    y = cleaned_train[target_col]

    val_size = model_cfg.get("val_split_size", 0.2)
    random_state = model_cfg.get("random_state", 42)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state
    )

    tracking_uri = model_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    exp_name = model_cfg.get("experiment_name", "bakery_bread_sales")
    mlflow.set_experiment(exp_name)

    candidates = model_cfg.get("candidate_models", [])
    if not candidates:
        candidates = [
            {"name": "ridge", "alpha": 1.0},
            {"name": "random_forest", "n_estimators": 100, "max_depth": 8},
            {"name": "gradient_boosting", "n_estimators": 100, "learning_rate": 0.05, "max_depth": 4},
        ]

    runs_info: list[dict[str, Any]] = []

    for cand_cfg in candidates:
        name, regressor = create_candidate_model(cand_cfg, random_state)
        pipeline = build_pipeline(regressor, numeric_features, categorical_features)

        with mlflow.start_run(run_name=f"v1_{name}") as run:
            run_id = run.info.run_id
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_family", name)
            mlflow.log_params(cand_cfg)

            pipeline.fit(X_train, y_train)

            train_preds = pipeline.predict(X_train)
            val_preds = pipeline.predict(X_val)

            train_mae = float(mean_absolute_error(y_train, train_preds))
            val_mae = float(mean_absolute_error(y_val, val_preds))

            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("val_mae", val_mae)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            runs_info.append({
                "run_id": run_id,
                "name": name,
                "val_mae": val_mae,
                "train_mae": train_mae,
                "pipeline": pipeline,
            })
            print(f"Run {name} (ID: {run_id}): val_mae = {val_mae:.4f}")

    # Pick the model with minimum validation MAE
    best_candidate = min(runs_info, key=lambda r: r["val_mae"])
    best_run_id = best_candidate["run_id"]
    best_pipeline = best_candidate["pipeline"]
    print(f"\nBest candidate: {best_candidate['name']} with val_mae: {best_candidate['val_mae']:.4f}")

    # Register the best model as Version 1
    reg_name = model_cfg.get("registered_model_name", "bakery_bread_model")
    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=reg_name)
    v1_version = str(model_version.version)

    client = MlflowClient()
    client.set_registered_model_alias(name=reg_name, alias="champion", version=v1_version)
    print(f"Registered model '{reg_name}' version {v1_version} as @champion.")

    # Save to models/model.joblib
    model_out_path = Path(model_cfg.get("model_path", "models/model.joblib"))
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_out_path)
    print(f"Saved champion pipeline to {model_out_path}")

    return {
        "experiment_name": exp_name,
        "best_run_id": best_run_id,
        "registered_model_name": reg_name,
        "champion_version": int(v1_version),
        "val_mae": best_candidate["val_mae"],
        "data_sha256": data_sha256,
        "best_pipeline": best_pipeline,
    }


if __name__ == "__main__":
    train_v1()
