from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier

from glass_quality import (
    ENGINEERED_INPUTS,
    create_features,
    cross_validate_log_loss,
    fit_predict,
    load_data,
    select_feature_columns,
)


def _frame(rows: int = 6) -> pd.DataFrame:
    values = np.arange(rows, dtype=float)
    frame = pd.DataFrame(
        {
            "xmax": values + 4,
            "xmin": values,
            "ymax": values + 6,
            "ymin": values + 1,
            "thickness": values + 1,
            "log_area": values + 10,
            "pixel_area": values + 2,
            "signal": values % 3,
        }
    )
    frame["class"] = np.tile([0, 1], rows // 2 + 1)[:rows]
    return frame


def test_load_data_enforces_required_geometry_and_finite_values(tmp_path):
    train = _frame()
    test = train.drop(columns="class")
    train.to_csv(tmp_path / "Train.csv", index=False)
    test.to_csv(tmp_path / "Test.csv", index=False)

    loaded_train, loaded_test = load_data(tmp_path)
    pd.testing.assert_frame_equal(loaded_train, train)
    pd.testing.assert_frame_equal(loaded_test, test)

    test.loc[0, "signal"] = np.inf
    test.to_csv(tmp_path / "Test.csv", index=False)
    with pytest.raises(ValueError, match="non-finite"):
        load_data(tmp_path)


def test_load_data_rejects_non_numeric_target_and_missing_geometry(tmp_path):
    train = _frame().drop(columns="log_area")
    test = train.drop(columns="class")
    train.to_csv(tmp_path / "Train.csv", index=False)
    test.to_csv(tmp_path / "Test.csv", index=False)

    with pytest.raises(ValueError, match="required geometry"):
        load_data(tmp_path)

    train = _frame()
    train["class"] = ["bad", "good", "bad", "good", "bad", "good"]
    train.to_csv(tmp_path / "Train.csv", index=False)
    train.drop(columns="class").to_csv(tmp_path / "Test.csv", index=False)
    with pytest.raises(ValueError, match="Train.csv features|target"):
        load_data(tmp_path)


def test_feature_engineering_is_row_local_and_excludes_known_redundancies():
    source = _frame()
    engineered = create_features(source)
    assert set(ENGINEERED_INPUTS).issubset(engineered.columns)
    assert engineered.loc[0, "area"] == 4 * 5
    assert "class" not in select_feature_columns(engineered)
    assert "pixel_area" not in select_feature_columns(engineered)
    assert "diff_area" not in select_feature_columns(engineered)
    assert source.columns.tolist() == _frame().columns.tolist()


def test_cross_validation_clones_model_and_fit_predict_preserves_class_order():
    train = create_features(_frame(10))
    test = train.drop(columns="class").iloc[[0, 1]].copy()
    features = select_feature_columns(train)
    model = ExtraTreesClassifier(n_estimators=8, random_state=11, n_jobs=1)

    scores = cross_validate_log_loss(model, train, features, n_splits=2)
    predictions = fit_predict(model, train, test, features)

    assert len(scores) == 2
    assert all(np.isfinite(scores))
    assert model.n_estimators == 8
    assert predictions.shape == (2, 2)
    assert predictions.columns.tolist() == [0, 1]
    np.testing.assert_allclose(predictions.sum(axis=1), 1.0)


def test_model_contracts_fail_before_training_on_invalid_inputs():
    train = create_features(_frame(6))
    features = select_feature_columns(train)
    with pytest.raises(ValueError, match="at least 2"):
        cross_validate_log_loss(ExtraTreesClassifier(), train, features, n_splits=1)
    with pytest.raises(ValueError, match="missing model features"):
        fit_predict(ExtraTreesClassifier(), train, train.drop(columns="signal"), features)
