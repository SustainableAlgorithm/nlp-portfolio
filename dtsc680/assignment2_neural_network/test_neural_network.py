import numpy as np
import pandas as pd
import pytest

from neural_network import (
    balanced_class_weights,
    evaluate_binary_predictions,
    prepare_binary_split,
)


@pytest.fixture
def sample_records():
    return pd.DataFrame({
        "Ind_ID": range(1, 21),
        "Test_date": ["2020-01-01"] * 20,
        "Cough_symptoms": [True, False] * 10,
        "Fever": [False, True, False, False] * 5,
        "Sex": ["female", "male", "female", "male"] * 5,
        "Known_contact": ["Abroad", "Other", "Contact", "Abroad"] * 5,
        "Corona": ["positive"] * 4 + ["negative"] * 16,
    })


def test_split_is_stratified_and_excludes_identifier_columns(sample_records):
    result = prepare_binary_split(sample_records, target_column="Corona")
    assert result.x_train.shape[1] == result.x_test.shape[1]
    assert result.x_test.shape[0] == 4
    assert set(np.unique(result.y_train)) == {0, 1}
    assert set(np.unique(result.y_test)) == {0, 1}
    assert all("Ind_ID" not in name and "Test_date" not in name for name in result.feature_names)
    assert np.isfinite(result.x_train).all()
    assert np.isfinite(result.x_test).all()


def test_missing_features_are_imputed(sample_records):
    sample_records.loc[0, "Cough_symptoms"] = np.nan
    result = prepare_binary_split(sample_records, target_column="Corona")
    assert np.isfinite(result.x_train).all()
    assert np.isfinite(result.x_test).all()


def test_excluded_target_labels_are_removed(sample_records):
    sample_records.loc[0, "Corona"] = "other"
    result = prepare_binary_split(sample_records, target_column="Corona")
    assert len(result.y_train) + len(result.y_test) == 19


def test_missing_target_column_is_rejected(sample_records):
    with pytest.raises(ValueError, match="target column"):
        prepare_binary_split(sample_records, target_column="missing")


def test_class_weights_follow_inverse_frequency_formula():
    weights = balanced_class_weights([0, 0, 0, 0, 1, 1])
    assert weights[0] == pytest.approx(0.75)
    assert weights[1] == pytest.approx(1.5)


def test_class_weights_reject_single_class():
    with pytest.raises(ValueError, match="at least two classes"):
        balanced_class_weights([0, 0, 0])


def test_metrics_expose_recall_precision_and_confusion_matrix():
    metrics = evaluate_binary_predictions([0, 0, 1, 1], [0.1, 0.8, 0.6, 0.9], threshold=0.5)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_metrics_validate_lengths_and_probability_range():
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_binary_predictions([0, 1], [0.4])
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_binary_predictions([0], [1.4])
