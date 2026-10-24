import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from dtsc670.apartment_price_regression.apartment_regression import evaluate as apartment_evaluate
from dtsc670.census_income_classification.census_classifier import CleanCategories
from dtsc670.classification_baselines.classification import compare_models
from dtsc670.ridge_regression.ridge_compare import compare as ridge_compare
from dtsc670.polynomial_regression.polynomial_compare import compare as polynomial_compare


def test_custom_category_cleaner_strips_whitespace():
    result = CleanCategories().fit_transform(pd.DataFrame({"x": [" yes ", " no "]}))
    assert result["x"].tolist() == ["yes", "no"]


def test_digit_classifier_comparison_returns_two_models():
    result = compare_models()
    assert set(result) == {"logistic_regression", "rbf_svm"}
    assert all(0 <= item["accuracy"] <= 1 for item in result.values())


def test_apartment_regression_on_synthetic_data():
    frame = pd.DataFrame({"size": np.arange(30), "neighborhood": ["A", "B"] * 15,
                          "price": 1000 + 25 * np.arange(30)})
    result = apartment_evaluate(frame, "price")
    assert {"mae", "rmse", "r2"} <= result.keys()
    assert result["rows"] == 30


def test_ridge_comparison_on_correlated_features():
    x = np.arange(60, dtype=float)
    frame = pd.DataFrame({"x1": x, "x2": x * 1.01 + np.sin(x) * .01,
                          "target": 3 * x + np.cos(x)})
    result = ridge_compare(frame, "target")
    assert {"ols", "sgd", "ridge_cv", "best_ridge_alpha"} <= result.keys()


def test_polynomial_comparison_on_nonlinear_data():
    x = np.linspace(-2, 2, 80)
    frame = pd.DataFrame({"feature_1": x, "feature_2": np.sin(x),
                          "target": x ** 2 + .5 * np.sin(x)})
    result = polynomial_compare(frame, "target")
    assert {"linear", "polynomial_degree_2", "polynomial_svr", "best_svr_params"} <= result.keys()
