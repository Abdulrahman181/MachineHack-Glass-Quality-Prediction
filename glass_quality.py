"""Reusable, data-guarded baseline for the glass quality challenge."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

TARGET = "class"
RANDOM_STATE = 11
ENGINEERED_INPUTS = ("xmax", "xmin", "ymax", "ymin", "thickness", "log_area", "pixel_area")
EXCLUDED_FEATURES = {TARGET, "grade_A_Component_1", "y", "pixel_area", "diff_area"}
IDENTIFIER_NAMES = {"id", "index", "unnamed: 0"}


def _missing_message(path: Path) -> str:
    return (
        f"Required data file is missing: {path}. Obtain the authorized challenge "
        "Train.csv and Test.csv files and place them under data/; no data is bundled "
        "with this repository."
    )


def _validate_numeric_values(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    """Reject missing, non-numeric, and infinite values before model operations."""
    columns = list(columns)
    non_numeric = [
        column for column in columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(f"{context} must be numeric; non-numeric columns: {non_numeric}.")
    if frame[columns].isna().any().any():
        raise ValueError(f"{context} contains missing values; imputation is not silently applied.")
    values = frame[columns].to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{context} contains infinite or non-finite values.")


def _validate_features(frame: pd.DataFrame, features: list[str], context: str) -> None:
    """Validate a model feature matrix and give callers actionable errors."""
    if not features:
        raise ValueError("At least one model feature is required.")
    if len(features) != len(set(features)):
        raise ValueError("Model feature names must be unique.")
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing model features: {missing}.")
    _validate_numeric_values(frame, features, f"{context} model features")


def _validate_target(
    frame: pd.DataFrame, target: str, *, require_numeric: bool = False
) -> pd.Series:
    """Validate labels used by either cross-validation or final fitting."""
    if target not in frame.columns:
        raise ValueError(f"Training frame is missing target column {target!r}.")
    labels = frame[target]
    if labels.empty:
        raise ValueError("Training data must contain at least one row.")
    if labels.isna().any():
        raise ValueError(f"Training target {target!r} contains missing labels.")
    if require_numeric and not pd.api.types.is_numeric_dtype(labels):
        raise ValueError(f"Training target {target!r} must be numeric.")
    if pd.api.types.is_numeric_dtype(labels) and not np.isfinite(labels.to_numpy(dtype=float)).all():
        raise ValueError(f"Training target {target!r} contains infinite or non-finite labels.")
    if labels.nunique() < 2:
        raise ValueError("The target must contain at least two classes for classification.")
    return labels


def load_data(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the challenge train/test CSVs without fabricating fallbacks."""
    data_dir = Path(data_dir)
    train_path, test_path = data_dir / "Train.csv", data_dir / "Test.csv"
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(_missing_message(path))
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
    except Exception as exc:  # pragma: no cover - pandas errors depend on input files
        raise ValueError(f"Could not read challenge CSVs from {data_dir}: {exc}") from exc
    if train.empty or test.empty:
        raise ValueError("Train.csv and Test.csv must both contain at least one row.")
    if train.columns.duplicated().any() or test.columns.duplicated().any():
        raise ValueError("Train.csv and Test.csv must not contain duplicate column names.")
    if TARGET not in train.columns:
        raise ValueError(f"Train.csv must contain the target column {TARGET!r}.")
    train_features = [column for column in train.columns if column != TARGET]
    missing = [column for column in train_features if column not in test.columns]
    extra = [column for column in test.columns if column not in train_features]
    if missing or extra:
        raise ValueError(
            "Train/Test feature columns do not match. "
            f"Missing from Test.csv: {missing}; unexpected in Test.csv: {extra}."
        )
    _validate_target(train, TARGET, require_numeric=True)
    _validate_numeric_values(train, train_features, "Train.csv features")
    _validate_numeric_values(test, train_features, "Test.csv features")
    missing_engineering_inputs = [
        column for column in ENGINEERED_INPUTS if column not in train_features
    ]
    if missing_engineering_inputs:
        raise ValueError(
            "Train.csv/Test.csv are missing required geometry columns for this workflow: "
            f"{missing_engineering_inputs}."
        )
    return train, test


def create_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create row-local geometry features; this function never reads the target."""
    if frame.columns.duplicated().any():
        raise ValueError("Feature-engineering input must not contain duplicate column names.")
    missing = [column for column in ENGINEERED_INPUTS if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot create geometry features; missing columns: {missing}.")
    if frame.empty:
        raise ValueError("Cannot create geometry features from an empty frame.")
    _validate_numeric_values(frame, ENGINEERED_INPUTS, "Geometry inputs")
    result = frame.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        result["x"] = result["xmax"] - result["xmin"]
        result["y"] = result["ymax"] - result["ymin"]
        result["area"] = result["x"] * result["y"].abs()
        result["volume"] = result["area"] * result["thickness"]
        result["diff_area"] = result["log_area"] - result["pixel_area"]
    _validate_numeric_values(
        result, ["x", "y", "area", "volume", "diff_area"], "Derived geometry features"
    )
    return result


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Select features in stable input order and exclude target/derived leakage risks."""
    if frame.columns.duplicated().any():
        raise ValueError("Feature-selection input must not contain duplicate column names.")
    selected = []
    for column in frame.columns:
        normalized = str(column).strip().lower()
        if column in EXCLUDED_FEATURES or normalized in IDENTIFIER_NAMES:
            continue
        selected.append(column)
    if not selected:
        raise ValueError("No usable model features remain after exclusions.")
    return selected


def make_model() -> ExtraTreesClassifier:
    """Return the reproducible baseline estimator used by the notebook."""
    return ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)


def cross_validate_log_loss(
    model: ClassifierMixin,
    train: pd.DataFrame,
    features: Iterable[str],
    target: str = TARGET,
    n_splits: int = 5,
) -> list[float]:
    """Evaluate with shuffled stratified folds, fitting a fresh clone per fold."""
    features = list(features)
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2.")
    _validate_features(train, features, "Training frame")
    labels = _validate_target(train, target)
    counts = labels.value_counts()
    if counts.min() < n_splits:
        raise ValueError(
            f"Each class needs at least {n_splits} rows for stratified validation; "
            f"smallest class has {counts.min()}."
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("The estimator must provide predict_proba for log-loss validation.")
    X, y = train[features], labels
    for train_index, validation_index in splitter.split(X, y):
        fitted = clone(model)
        fitted.fit(X.iloc[train_index], y.iloc[train_index])
        probabilities = fitted.predict_proba(X.iloc[validation_index])
        scores.append(float(log_loss(y.iloc[validation_index], probabilities, labels=fitted.classes_)))
    return scores


def fit_predict(
    model: ClassifierMixin,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Iterable[str],
    target: str = TARGET,
) -> pd.DataFrame:
    """Fit on all labeled rows and return probabilities in estimator class order."""
    features = list(features)
    if train.empty or test.empty:
        raise ValueError("Training and test data must both contain at least one row.")
    _validate_features(train, features, "Training frame")
    _validate_features(test, features, "Test frame")
    _validate_target(train, target)
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("The estimator must provide predict_proba for probability output.")
    fitted = clone(model)
    fitted.fit(train[features], train[target])
    probabilities = fitted.predict_proba(test[features])
    return pd.DataFrame(probabilities, columns=fitted.classes_, index=test.index)
