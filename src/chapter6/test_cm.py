from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from gen_bcw_data import gen_bcw_data

import matplotlib.pyplot as plt

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = gen_bcw_data()

    pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
    pipe_svc.fit(X_train, y_train)

    y_pred = pipe_svc.predict(X_test)

    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

    print(f"confusion matrix:\n {confmat}")

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va="center", ha="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    plt.savefig("../../figure/test_cm.png")
