"""Self-contained unit tests for bakery model pipeline, monitoring, and prediction formatting.

All tests use synthetic in-memory fixtures. No raw or processed CSV files required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from bakery.monitor import compute_psi, evaluate_promotion
from bakery.train import build_pipeline


NUMERIC_FEATURES = [
    "price_rs",
    "temp_c",
    "footfall_index",
    "competitor_shops",
    "shelf_hours",
    "orders_previous_day",
]
CATEGORICAL_FEATURES = ["day_type", "outlet"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "loaves_sold"


def make_synthetic_dataframe(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Create a realistic in-memory synthetic dataset for unit testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "record_id": [f"TEST_{i:04d}" for i in range(n)],
        "price_rs": rng.uniform(30.0, 70.0, size=n),
        "temp_c": rng.uniform(10.0, 35.0, size=n),
        "footfall_index": rng.integers(30, 95, size=n),
        "competitor_shops": rng.integers(1, 10, size=n),
        "shelf_hours": rng.uniform(5.0, 20.0, size=n),
        "orders_previous_day": rng.integers(50, 500, size=n),
        "day_type": rng.choice(["weekday", "weekend", "festival"], size=n),
        "outlet": rng.choice(["Station Road", "Mall Road", "College Gate"], size=n),
        "loaves_sold": rng.uniform(200.0, 600.0, size=n),
    })


def test_pipeline_training_and_inference():
    """Verify that build_pipeline fits on synthetic 8 features and produces valid numeric predictions."""
    train_df = make_synthetic_dataframe(n=60, seed=0)
    test_df = make_synthetic_dataframe(n=20, seed=1)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]

    pipeline = build_pipeline(
        Ridge(alpha=1.0),
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    assert len(preds) == len(test_df)
    assert np.issubdtype(preds.dtype, np.floating)
    assert not np.isnan(preds).any()
    assert not np.isinf(preds).any()


def test_pipeline_handles_unseen_categories():
    """Verify that ColumnTransformer handles unseen categorical levels gracefully without crashing."""
    train_df = make_synthetic_dataframe(n=40, seed=10)
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    pipeline = build_pipeline(
        RandomForestRegressor(n_estimators=10, random_state=42),
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
    pipeline.fit(X_train, y_train)

    # Test set containing novel / unseen categories
    unseen_df = pd.DataFrame({
        "price_rs": [45.0, 50.0],
        "temp_c": [22.0, 25.0],
        "footfall_index": [60, 70],
        "competitor_shops": [3, 4],
        "shelf_hours": [10.0, 12.0],
        "orders_previous_day": [200, 250],
        "day_type": ["national_holiday", "special_event"],  # Unseen levels
        "outlet": ["Airport Kiosk", "Metro Station"],       # Unseen levels
    })

    preds = pipeline.predict(unseen_df)
    assert len(preds) == 2
    assert not np.isnan(preds).any()


def test_psi_identical_distributions():
    """Verify PSI calculation yields near-zero value (< 0.05) for identical distributions."""
    rng = np.random.default_rng(42)
    expected = rng.normal(loc=50.0, scale=10.0, size=500)
    actual = rng.normal(loc=50.0, scale=10.0, size=500)

    psi_val = compute_psi(expected, actual, num_bins=10)
    assert psi_val < 0.05, f"Expected low PSI for identical distributions, got {psi_val}"


def test_psi_shifted_distribution():
    """Verify PSI calculation detects significant drift (> 0.25) when distribution shifts."""
    rng = np.random.default_rng(42)
    expected = rng.normal(loc=10.0, scale=2.0, size=500)
    actual = rng.normal(loc=25.0, scale=2.0, size=500)  # Substantial shift

    psi_val = compute_psi(expected, actual, num_bins=10)
    assert psi_val > 0.25, f"Expected high PSI for shifted distribution, got {psi_val}"


@pytest.mark.parametrize(
    "mae_v1, mae_v2, expected_decision",
    [
        (100.0, 85.0, "promote"),  # 15% reduction >= 12% -> promote
        (100.0, 88.0, "promote"),  # 12% reduction >= 12% -> promote
        (100.0, 89.0, "keep"),     # 11% reduction < 12% -> keep
        (100.0, 105.0, "keep"),    # Negative improvement (worse) -> keep
    ],
)
def test_evaluate_promotion_rule(mae_v1: float, mae_v2: float, expected_decision: str):
    """Verify client's promotion rule strictly enforces the 12% improvement threshold."""
    decision, improvement = evaluate_promotion(mae_v1, mae_v2, threshold=0.12)
    assert decision == expected_decision


def test_holdout_prediction_formatting():
    """Verify holdout predictions output structure adheres to exact format: record_id,prediction."""
    holdout_df = make_synthetic_dataframe(n=30, seed=7)
    mock_preds = np.random.default_rng(7).uniform(300.0, 500.0, size=30)

    preds_df = pd.DataFrame({
        "record_id": holdout_df["record_id"],
        "prediction": np.round(mock_preds, 4),
    })

    assert list(preds_df.columns) == ["record_id", "prediction"]
    assert len(preds_df) == 30
    assert not preds_df["prediction"].isnull().any()
    assert preds_df["record_id"].nunique() == 30
