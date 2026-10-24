# Ridge Regression and Multicollinearity

Independent comparison of ordinary least squares, scaled SGD regression, and Ridge regression with cross-validated alpha selection. Supply a local CSV and target column; no course data is included.

Run: `python ridge_compare.py --data path/to/data.csv --target target`. Reports held-out MAE, RMSE, and R². Scaling and model fitting occur inside pipelines to avoid leakage.