"""Reusable baseline for apartment price regression on a local CSV."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

def build_model(frame: pd.DataFrame) -> Pipeline:
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocess", pre), ("regressor", LinearRegression())])

def evaluate(data: pd.DataFrame, target: str) -> dict:
    if target not in data: raise ValueError(f"Target column {target!r} is missing")
    data = data.dropna(subset=[target]).copy()
    y = pd.to_numeric(data.pop(target), errors="coerce")
    valid = y.notna()
    data, y = data.loc[valid], y.loc[valid]
    data = data.drop(columns=[c for c in data.columns if c.lower() in {"id", "listing_id", "url"}], errors="ignore")
    if data.empty or len(data) < 5: raise ValueError("Need at least five usable rows")
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=.2, random_state=42)
    model = build_model(X_train).fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"mae": float(mean_absolute_error(y_test,pred)), "rmse": float(np.sqrt(mean_squared_error(y_test,pred))), "r2": float(r2_score(y_test,pred)), "rows": len(data), "features": list(data.columns)}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data",required=True); p.add_argument("--target",default="price"); a=p.parse_args()
    result=evaluate(pd.read_csv(a.data),a.target)
    for k,v in result.items(): print(f"{k}: {v}")
if __name__=="__main__": main()
