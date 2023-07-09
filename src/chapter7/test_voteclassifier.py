from majority_vote_classifier import MajorityVoteClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt

if __name__ == "__main__":
    iris = datasets.load_iris()

    x_data, y_data = iris.data[50:, [1, 2]], iris.target[50:]

    encoder = LabelEncoder()
    y_data = encoder.fit_transform(y_data)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.5, random_state=1, stratify=y_data
    )

    classifier1 = LogisticRegression(
        penalty="l2", C=0.001, solver="lbfgs", random_state=1
    )
    classifier2 = DecisionTreeClassifier(
        max_depth=1, criterion="entropy", random_state=0
    )
    classifier3 = KNeighborsClassifier(n_neighbors=1, p=2, metric="minkowski")

    pipe1 = Pipeline(
        [
            ["sc", StandardScaler()],
            ["clf", classifier1],
        ]
    )

    pipe3 = Pipeline(
        [
            ["sc", StandardScaler()],
            ["clf", classifier3],
        ]
    )

    classifier_labels = ["Logistic Regression", "Decision Tree", "KNN"]

    classifier = MajorityVoteClassifier(classifiers=[pipe1, classifier2, pipe3])
    classifier_labels += ["Majority Voting"]
    all_classifier = [pipe1, classifier2, pipe3, classifier]

    for clf, label in zip(all_classifier, classifier_labels):
        scores = cross_val_score(
            estimator=clf,
            X=x_train,
            y=y_train,
            cv=10,
            scoring="roc_auc",
        )
        print(f"ROC AUC: {scores.mean():.2f} (+/- {scores.std():.2f}) [{label}]")

    colors = ["black", "orange", "blue", "green"]
    linestyles = [":", "--", "-.", "-"]

    for clf, label, clr, ls in zip(
        all_classifier, classifier_labels, colors, linestyles
    ):
        # 陽性クラスのラベルが1であることが理想

        y_predict = clf.fit(x_train, y_train).predict_proba(x_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_predict)
        roc_auc = auc(x=fpr, y=tpr)
        plt.plot(
            fpr, tpr, color=clr, linestyle=ls, label=f"{label} (auc = {roc_auc:.2f})"
        )

        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.grid(alpha=0.5)
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")

        plt.savefig("../../figure/test_voteclassifier.png")
