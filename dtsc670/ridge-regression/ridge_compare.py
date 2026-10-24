"""Compare OLS, SGD, and cross-validated Ridge regression on a local CSV."""
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def compare(frame: pd.DataFrame, target: str) -> dict:
    if target not in frame: raise ValueError(f"Missing target column: {target}")
    frame=frame.dropna(subset=[target]); y=pd.to_numeric(frame.pop(target),errors="coerce")
    good=y.notna(); frame,y=frame.loc[good],y.loc[good]
    X=pd.get_dummies(frame,drop_first=False,dummy_na=True)
    X=X.replace([np.inf,-np.inf],np.nan).fillna(X.median(numeric_only=True)).fillna(0)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
    candidates={
      "ols":make_pipeline(StandardScaler(),LinearRegression()),
      "sgd":make_pipeline(StandardScaler(),SGDRegressor(max_iter=10000,tol=1e-4,random_state=42)),
    }
    ridge=make_pipeline(StandardScaler(),Ridge())
    search=GridSearchCV(ridge,{"ridge__alpha":[.01,.1,1,10,100]},cv=5,scoring="neg_root_mean_squared_error")
    search.fit(X_train,y_train);candidates["ridge_cv"]=search.best_estimator_
    results={}
    for name,model in candidates.items():
        if name!="ridge_cv":model.fit(X_train,y_train)
        pred=model.predict(X_test)
        results[name]={"mae":float(mean_absolute_error(y_test,pred)),"rmse":float(np.sqrt(mean_squared_error(y_test,pred))),"r2":float(r2_score(y_test,pred))}
    results["best_ridge_alpha"]=float(search.best_params_["ridge__alpha"])
    return results
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data",required=True);p.add_argument("--target",default="target");a=p.parse_args()
    print(compare(pd.read_csv(a.data),a.target))
if __name__=="__main__":main()
