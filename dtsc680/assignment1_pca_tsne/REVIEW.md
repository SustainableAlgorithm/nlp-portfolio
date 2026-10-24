# DTSC 680 Assignment 1 — Portfolio Review

Review date: 2026-10-24

## Materials inspected

The supplied ZIP contains:
- The original Jupyter notebook for DTSC 680 Assignment 1: Principal Component Analysis (PCA).
- `penguins.csv`.
- Four serialized pickle outputs, including cleaned data and dimensionality-reduction arrays.
- Two environment YAML files.
- A Jupyter checkpoint copy of the notebook.

The original notebook, course dataset, pickle files, checkpoint, and environment exports have **not** been committed to GitHub. The portfolio implementation is newly written and keeps data files out of version control.

## Original assignment scope

The notebook asks for data cleaning, categorical encoding, PCA, t-SNE, t-SNE after PCA, explained-variance extraction, and visual comparison of the resulting embeddings.

## Independent implementation

The private portfolio implementation:
- Separates the species label from the input features.
- Drops rows with missing values for this demonstration.
- Standardizes numerical columns and one-hot encodes categorical columns.
- Produces 2-D PCA, t-SNE, and PCA-then-t-SNE embeddings.
- Uses explicit t-SNE parameters and a fixed random seed.
- Validates the target column, row count, and perplexity.

This implementation is intentionally not a line-for-line copy of the coursework notebook. It changes preprocessing choices, notably replacing label encoding of nominal feature categories with one-hot encoding and standardizing numeric features before dimensionality reduction.

## Local reproduction on supplied dataset

Using the local `penguins.csv` only (not uploaded), the independent workflow found:
- 344 rows in the source CSV; 333 complete rows after dropping rows with missing values.
- 10 processed features after numeric scaling and one-hot encoding.
- PCA explained-variance ratio for the first two components: 0.4833 and 0.1813, totaling 0.6646 (66.46%).
- Species silhouette score on the 2-D PCA embedding: 0.3227.
- Species silhouette score on the 2-D t-SNE embedding: 0.4429.
- Species silhouette score on the t-SNE embedding generated from the 2-D PCA coordinates: 0.3039.

Silhouette scores describe label separation in these specific embeddings. They are not predictive accuracy, and t-SNE can change apparent cluster distances and shapes. These values should not be treated as universal or definitive evidence of cluster structure.

## Automated tests

**5 tests passed** in a temporary Python workspace for the independent implementation:
- Feature preprocessing excludes the target and encodes categorical features.
- Missing rows are dropped.
- Missing target-column errors are raised.
- The three embeddings have the expected shapes.
- Invalid t-SNE perplexity is rejected.

## Limitations and next steps

- The test suite uses a small synthetic fixture; the supplied dataset was used for a separate local run.
- t-SNE is stochastic and sensitive to settings; fixed settings aid repeatability but do not remove modeling choices.
- One-hot encoding and standardization differ from the original assignment's label-encoding approach, so its plots/results should not be claimed as identical.
- A future revision can add a reproducible plotting function, parameter sensitivity comparison, and a clearer discussion of PCA's linear variance objective versus t-SNE's neighborhood-preservation objective.

## Privacy and publication

No source CSV, pickle outputs, original notebook, or environment files are included in this repository. The public portfolio page reports summary methods and clearly qualified exploratory results only.
