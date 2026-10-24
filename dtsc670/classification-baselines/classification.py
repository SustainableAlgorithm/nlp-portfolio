"""Compare simple classifiers on scikit-learn's built-in handwritten digits dataset."""
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def compare_models():
    data=load_digits()
    X_train,X_test,y_train,y_test=train_test_split(data.data,data.target,test_size=.2,random_state=42,stratify=data.target)
    models={
        "logistic_regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=42)),
        "rbf_svm":make_pipeline(StandardScaler(),SVC(kernel="rbf",gamma="scale")),
    }
    results={}
    for name,model in models.items():
        model.fit(X_train,y_train);pred=model.predict(X_test)
        results[name]={"accuracy":float(accuracy_score(y_test,pred)),"report":classification_report(y_test,pred,output_dict=True,zero_division=0)}
    return results
if __name__=="__main__":
    for name,result in compare_models().items(): print(name, "accuracy:",round(result["accuracy"],4))
