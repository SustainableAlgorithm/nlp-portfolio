"""Independent utilities for imbalanced binary-classification experiments.

This module deliberately does not load or bundle the original course dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class BinarySplit:
    """Preprocessed, stratified train/test split plus fitted preprocessing metadata."""
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    feature_names: list[str]


def prepare_binary_split(
    frame: pd.DataFrame,
    *,
    target_column: str,
    positive_label: str = "positive",
    negative_label: str = "negative",
    excluded_labels: Sequence[str] = ("other",),
    drop_columns: Sequence[str] = ("Ind_ID", "Test_date"),
    test_size: float = 0.2,
    random_state: int = 42,
) -> BinarySplit:
    """Clean labels, split stratifiably, then fit preprocessing on training data only.

    Numeric columns are median-imputed and standardized. Categorical columns are
    mode-imputed and one-hot encoded. Identifier/date columns are excluded by
    default. The positive class is encoded as 1 and the negative class as 0.
    """
    if target_column not in frame.columns:
        raise ValueError(f"target column {target_column!r} is missing")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    clean = frame.copy()
    clean = clean.loc[clean[target_column].notna()].copy()
    excluded = {str(label).strip().casefold() for label in excluded_labels}
    target_text = clean[target_column].astype(str).str.strip().str.casefold()
    clean = clean.loc[~target_text.isin(excluded)].copy()
    target_text = clean[target_column].astype(str).str.strip().str.casefold()

    positive = str(positive_label).strip().casefold()
    negative = str(negative_label).strip().casefold()
    present = set(target_text.unique())
    unexpected = present - {positive, negative}
    if unexpected:
        raise ValueError(f"unexpected target labels: {sorted(unexpected)}")
    if present != {positive, negative}:
        raise ValueError("both positive and negative target classes are required")

    y = (target_text == positive).astype("int64").to_numpy()
    x = clean.drop(columns=[target_column, *[c for c in drop_columns if c in clean.columns]])
    if x.shape[1] == 0:
        raise ValueError("no feature columns remain after dropping target and excluded columns")

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    numeric_columns = x_train_raw.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [c for c in x_train_raw.columns if c not in numeric_columns]
    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    x_train = np.asarray(preprocessor.fit_transform(x_train_raw), dtype=np.float32)
    x_test = np.asarray(preprocessor.transform(x_test_raw), dtype=np.float32)
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    return BinarySplit(x_train, x_test, y_train, y_test, preprocessor, names)


def balanced_class_weights(labels: Iterable[int]) -> dict[int, float]:
    """Compute inverse-frequency class weights: n / (k * class_count)."""
    y = np.asarray(list(labels))
    if y.size == 0:
        raise ValueError("labels must not be empty")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("at least two classes are required")
    n_samples = len(y)
    n_classes = len(classes)
    return {int(label): float(n_samples / (n_classes * count)) for label, count in zip(classes, counts)}


def build_keras_binary_classifier(n_features: int, *, learning_rate: float = 0.001):
    """Build the single-sigmoid-output Keras network used for binary classification.

    TensorFlow is imported lazily so preprocessing and metric tests do not require
    TensorFlow to be installed.
    """
    if n_features < 1:
        raise ValueError("n_features must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to build the Keras model. Install the optional "
            "dependency described in this project's README."
        ) from exc

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_binary_predictions(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Return thresholded binary metrics and a 2x2 confusion matrix."""
    truth = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(list(probabilities), dtype=float)
    if truth.size == 0 or scores.size == 0:
        raise ValueError("truth and probabilities must not be empty")
    if truth.shape[0] != scores.shape[0]:
        raise ValueError("truth and probabilities must have equal lengths")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must be finite values between 0 and 1")

    predicted = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
        "threshold": float(threshold),
    }
