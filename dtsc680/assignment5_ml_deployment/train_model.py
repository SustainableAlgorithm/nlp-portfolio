"""Train and locally save a Titanic survival probability model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["Age", "Sex", "Pclass"]
TARGET = "Survived"


def build_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, ["Age"]),
        ("categorical", categorical, ["Sex", "Pclass"]),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def train(data_path: str | Path, output_path: str | Path) -> Path:
    """Train on the supplied local CSV and save the fitted pipeline."""
    data_path, output_path = Path(data_path), Path(output_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    data = pd.read_csv(data_path)
    missing = set(FEATURES + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    data = data[FEATURES + [TARGET]].copy()
    data["Age"] = pd.to_numeric(data["Age"], errors="coerce")
    data["Sex"] = data["Sex"].astype("string").str.strip().str.lower()
    data["Pclass"] = data["Pclass"].astype("string").str.strip()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    data = data[data[TARGET].isin([0, 1])]
    if data[TARGET].nunique() != 2:
        raise ValueError("Target column 'Survived' must contain both classes 0 and 1")
    if data.empty:
        raise ValueError("No usable rows found in dataset")
    model = build_pipeline()
    model.fit(data[FEATURES], data[TARGET].astype(int))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to a local Titanic CSV")
    parser.add_argument("--output", default="artifacts/titanic_model.joblib", help="Local model artifact path")
    args = parser.parse_args()
    saved = train(args.data, args.output)
    print(f"Saved trained pipeline to {saved}")


if __name__ == "__main__":
    main()
