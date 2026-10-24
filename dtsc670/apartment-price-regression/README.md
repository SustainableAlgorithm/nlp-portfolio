# Apartment Price Regression

Independent regression workflow for apartment listings. Uses a local CSV and scikit-learn preprocessing pipeline; no course notebook or dataset is included.

Expected target column: `price`. Candidate numeric/categorical features are inferred from the local file, with identifier-like columns dropped by a configurable rule.

Run: `python apartment_regression.py --data path/to/listings.csv --target price`

The script reports MAE, RMSE, and R² on a held-out test split. Confirm dataset permissions before use and inspect inferred feature columns before interpreting results.