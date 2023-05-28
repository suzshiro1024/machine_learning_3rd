from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from gen_bcw_data import gen_bcw_data


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = gen_bcw_data()

    pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
    pipe_svc.fit(X_train, y_train)
    y_pred = pipe_svc.predict(X_test)

    print(f"Precision: {precision_score(y_true=y_test, y_pred=y_pred)}")
    print(f"Recall: {recall_score(y_true=y_test,y_pred=y_pred)}")
    print(f"F1: {f1_score(y_true=y_test,y_pred=y_pred)}")
