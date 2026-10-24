"""Census-style binary classification with a custom transformer and preprocessing pipeline."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class CleanCategories(BaseEstimator, TransformerMixin):
    """Normalize whitespace and replace common placeholder missing values."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        out = X.copy()
        for col in out.columns:
            out[col] = out[col].astype("string").str.strip().replace({"?": pd.NA, "": pd.NA})
        return out

def build_model(X: pd.DataFrame) -> Pipeline:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("clean", CleanCategories()), ("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("prep", prep), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])

def evaluate(frame: pd.DataFrame, target: str) -> dict:
    if target not in frame: raise ValueError(f"Missing target column: {target}")
    y = frame.pop(target).astype("string").str.strip().str.rstrip(".")
    valid = y.notna()
    frame, y = frame.loc[valid], y.loc[valid]
    if y.nunique() != 2: raise ValueError("Target must contain exactly two classes")
    X_train,X_test,y_train,y_test=train_test_split(frame,y,test_size=.2,random_state=42,stratify=y)
    model=build_model(X_train).fit(X_train,y_train)
    pred=model.predict(X_test)
    precision,recall,f1,_=precision_recall_fscore_support(y_test,pred,average="binary",pos_label=sorted(y.unique())[-1],zero_division=0)
    out={"accuracy":float(accuracy_score(y_test,pred)),"precision":float(precision),"recall":float(recall),"f1":float(f1),"rows":len(frame)}
    if hasattr(model,"predict_proba"):
        out["roc_auc"]=float(roc_auc_score((y_test==sorted(y.unique())[-1]).astype(int),model.predict_proba(X_test)[:,list(model.classes_).index(sorted(y.unique())[-1])]))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data",required=True);p.add_argument("--target",default="income");a=p.parse_args()
    print(evaluate(pd.read_csv(a.data,na_values=["?"]),a.target))
if __name__=="__main__":main()
