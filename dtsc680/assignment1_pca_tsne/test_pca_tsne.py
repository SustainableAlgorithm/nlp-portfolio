import numpy as np
import pandas as pd
import pytest
from pca_tsne import prepare_features, reduce_dimensions

@pytest.fixture
def sample_penguins():
    return pd.DataFrame({
        "species": ["Adelie", "Chinstrap", "Gentoo", "Adelie", "Gentoo", "Chinstrap"],
        "island": ["A", "B", "C", "A", "C", "B"],
        "bill_length_mm": [39.1, 46.5, 50.0, 38.5, 49.0, 47.1],
        "body_mass_g": [3750, 3800, 5000, 3600, 5100, 3900],
        "sex": ["male", "female", "male", "female", "male", "female"],
    })

def test_prepare_features_excludes_target_and_encodes_categories(sample_penguins):
    matrix, labels, names = prepare_features(sample_penguins)
    assert matrix.shape == (6, 7)
    assert len(labels) == 6
    assert all("species" not in name for name in names)
    assert np.isfinite(matrix).all()

def test_prepare_features_drops_rows_with_missing_values(sample_penguins):
    sample_penguins.loc[0, "body_mass_g"] = np.nan
    matrix, labels, _ = prepare_features(sample_penguins)
    assert matrix.shape[0] == 5
    assert len(labels) == 5

def test_missing_target_column_is_rejected(sample_penguins):
    with pytest.raises(ValueError, match="target column"):
        prepare_features(sample_penguins, target_column="missing")

def test_reduce_dimensions_returns_three_two_dimensional_embeddings(sample_penguins):
    result = reduce_dimensions(sample_penguins, perplexity=2)
    assert result.pca_coordinates.shape == (6, 2)
    assert result.tsne_coordinates.shape == (6, 2)
    assert result.pca_then_tsne_coordinates.shape == (6, 2)
    assert result.explained_variance_ratio.shape == (2,)
    assert np.isfinite(result.pca_coordinates).all()

def test_invalid_perplexity_is_rejected(sample_penguins):
    with pytest.raises(ValueError, match="perplexity"):
        reduce_dimensions(sample_penguins, perplexity=6)
