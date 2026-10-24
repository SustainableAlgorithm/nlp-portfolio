# Dimensionality Reduction: PCA and t-SNE

An independent Python implementation exploring three 2-D visualizations of tabular data: PCA, t-SNE, and t-SNE applied after PCA.

## Design choices
- Drops rows with missing values for this demonstration.
- Separates the target label from input features to avoid target leakage.
- Standardizes numeric features and one-hot encodes categorical features.
- Uses a fixed random seed and explicit t-SNE settings for repeatability.
- Keeps source data and generated embeddings outside version control.

This is an original portfolio implementation of general methods, not a copy of the DTSC 680 assignment notebook. No course notebook, pickle output, or dataset is included.

## Run
Install dependencies listed in `requirements.txt`, then run `python -m pytest -q`.

To use a licensed CSV with a `species` target column:
```python
from pca_tsne import load_csv, reduce_dimensions
frame = load_csv("path/to/your-data.csv")
results = reduce_dimensions(frame, target_column="species")
print(results.explained_variance_ratio)
```

t-SNE coordinates are intended for visualization, not as interpretable feature values. Visual cluster separation depends on preprocessing and hyperparameters; it is not proof of predictive performance.
