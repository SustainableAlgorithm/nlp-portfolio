# Census Income Classification

Independent classification pipeline for estimating whether income exceeds a threshold from tabular census-style data. The CSV is not included. This example demonstrates missing-value handling, custom feature engineering, column transformers, and held-out evaluation.

Run: `python census_classifier.py --data path/to/census.csv --target income`

Expected labels should be normalized to two classes. The model is for education only: census attributes can encode structural inequities, and performance does not establish fairness or justify decisions about individuals.