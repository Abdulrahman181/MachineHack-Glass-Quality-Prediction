"""Reusable, data-guarded baseline for the glass quality challenge."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

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
    if train[TARGET].isna().any():
        raise ValueError("Train.csv contains missing target labels in the 'class' column.")
    if train[TARGET].nunique() < 2:
        raise ValueError("The target must contain at least two classes for classification.")
    non_numeric = [
        column
        for column in train_features
        if not pd.api.types.is_numeric_dtype(train[column])
        or not pd.api.types.is_numeric_dtype(test[column])
    ]
    if non_numeric:
        raise ValueError(f"All model features must be numeric; non-numeric columns: {non_numeric}.")
    if train[train_features].isna().any().any() or test[train_features].isna().any().any():
        raise ValueError("Train.csv/Test.csv contain missing feature values; imputation is not silently applied.")
    return train, test


def create_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create row-local geometry features; this function never reads the target."""
    missing = [column for column in ENGINEERED_INPUTS if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot create geometry features; missing columns: {missing}.")
    result = frame.copy()
    result["x"] = result["xmax"] - result["xmin"]
    result["y"] = result["ymax"] - result["ymin"]
    result["area"] = result["x"] * result["y"].abs()
    result["volume"] = result["area"] * result["thickness"]
    result["diff_area"] = result["log_area"] - result["pixel_area"]
    return result


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Select features in stable input order and exclude target/derived leakage risks."""
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
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if target not in train.columns:
        raise ValueError(f"Training frame is missing target column {target!r}.")
    counts = train[target].value_counts()
    if counts.min() < n_splits:
        raise ValueError(
            f"Each class needs at least {n_splits} rows for stratified validation; "
            f"smallest class has {counts.min()}."
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []
    X, y = train[features], train[target]
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
    missing = [column for column in features if column not in test.columns]
    if missing:
        raise ValueError(f"Test frame is missing model features: {missing}.")
    fitted = clone(model)
    fitted.fit(train[features], train[target])
    probabilities = fitted.predict_proba(test[features])
    return pd.DataFrame(probabilities, columns=fitted.classes_, index=test.index)
