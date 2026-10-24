"""Independent dimensionality-reduction workflow for tabular data."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class EmbeddingResults:
    labels: pd.Series
    feature_names: list[str]
    pca_coordinates: np.ndarray
    tsne_coordinates: np.ndarray
    pca_then_tsne_coordinates: np.ndarray
    explained_variance_ratio: np.ndarray

def prepare_features(frame: pd.DataFrame, target_column: str = "species") -> tuple[np.ndarray, pd.Series, list[str]]:
    """Separate labels and preprocess features without encoding the target as input."""
    if target_column not in frame.columns:
        raise ValueError(f"target column {target_column!r} is missing")
    clean = frame.dropna().copy()
    if clean.empty:
        raise ValueError("no rows remain after removing missing values")
    labels = clean[target_column].copy()
    features = clean.drop(columns=[target_column])
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=np.number).columns.tolist()
    if not numeric_columns and not categorical_columns:
        raise ValueError("no input features found")
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    transformer = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])
    matrix = transformer.fit_transform(features)
    names = [str(name) for name in transformer.get_feature_names_out()]
    return np.asarray(matrix, dtype=float), labels.reset_index(drop=True), names

def reduce_dimensions(frame: pd.DataFrame, *, target_column: str = "species", random_state: int = 0, perplexity: float = 30.0) -> EmbeddingResults:
    """Create PCA, t-SNE, and PCA-then-t-SNE 2-D embeddings."""
    matrix, labels, feature_names = prepare_features(frame, target_column=target_column)
    if matrix.shape[0] < 4:
        raise ValueError("at least four complete rows are required for this workflow")
    if not 0 < perplexity < matrix.shape[0]:
        raise ValueError("perplexity must be greater than 0 and smaller than row count")
    pca = PCA(n_components=2, random_state=random_state)
    pca_coordinates = pca.fit_transform(matrix)
    tsne_coordinates = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(matrix)
    pca_then_tsne = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(pca_coordinates)
    return EmbeddingResults(labels, feature_names, pca_coordinates, tsne_coordinates, pca_then_tsne, pca.explained_variance_ratio_)

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a UTF-8 CSV file."""
    return pd.read_csv(path, encoding="utf-8")
