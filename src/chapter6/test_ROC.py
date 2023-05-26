from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from numpy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import make_pipeline
from gen_bcw_data import gen_bcw_data

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = gen_bcw_data()

    pipe_lr = make_pipeline(
        StandardScaler(),
        PCA(n_components=2),
        LogisticRegression(penalty="l2", random_state=1, solver="lbfgs", C=100.0),
    )

    X_train2 = X_train[:, [4, 14]]

    cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))
    fig = plt.figure(figsize=(7, 5))

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(
            X_train2[test]
        )

        fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label="ROC fold %d (area = %0.2f)" % (i + 1, roc_auc))

    plt.plot(
        [0, 1], [0, 1], linestyle="--", color=(0.6, 0.6, 0.6), label="Random Guessing"
    )
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(
        mean_fpr, mean_tpr, "k--", label="Mean ROC (area = %0.2f)" % mean_auc, lw=2
    )
    plt.plot(
        [0, 0, 1], [0, 1, 1], linestyle=":", color="black", label="Perfect Performance"
    )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig("../../figure/test_ROC.png")
