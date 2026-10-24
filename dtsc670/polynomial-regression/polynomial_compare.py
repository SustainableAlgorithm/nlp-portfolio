"""Compare linear, polynomial, and polynomial-kernel SVR models."""
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

def compare(frame: pd.DataFrame, target: str) -> dict:
    if target not in frame: raise ValueError(f"Missing target column: {target}")
    frame=frame.dropna(subset=[target]);y=pd.to_numeric(frame.pop(target),errors="coerce")
    good=y.notna();frame,y=frame.loc[good],y.loc[good]
    X=pd.get_dummies(frame,drop_first=False,dummy_na=True).replace([np.inf,-np.inf],np.nan)
    X=X.fillna(X.median(numeric_only=True)).fillna(0)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
    models={
      "linear":make_pipeline(StandardScaler(),LinearRegression()),
      "polynomial_degree_2":make_pipeline(StandardScaler(),PolynomialFeatures(degree=2,include_bias=False),LinearRegression()),
    }
    search=GridSearchCV(make_pipeline(StandardScaler(),SVR(kernel="poly")),{"svr__degree":[2,3,4],"svr__C":[.1,1,10],"svr__epsilon":[.05,.1,.5]},cv=5,scoring="neg_root_mean_squared_error")
    search.fit(X_train,y_train);models["polynomial_svr"]=search.best_estimator_
    results={}
    for name,model in models.items():
        if name!="polynomial_svr":model.fit(X_train,y_train)
        pred=model.predict(X_test)
        results[name]={"mae":float(mean_absolute_error(y_test,pred)),"rmse":float(np.sqrt(mean_squared_error(y_test,pred))),"r2":float(r2_score(y_test,pred))}
    results["best_svr_params"]=search.best_params_
    return results
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data",required=True);p.add_argument("--target",default="target");a=p.parse_args()
    print(compare(pd.read_csv(a.data),a.target))
if __name__=="__main__":main()
