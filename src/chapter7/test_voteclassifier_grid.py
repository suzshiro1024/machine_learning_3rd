from itertools import product
from majority_vote_classifier import MajorityVoteClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    iris = datasets.load_iris()

    x_data, y_data = iris.data[50:, [1, 2]], iris.target[50:]

    encoder = LabelEncoder()
    scaler = StandardScaler()
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
            ["sc", scaler],
            ["clf", classifier1],
        ]
    )

    pipe3 = Pipeline(
        [
            ["sc", scaler],
            ["clf", classifier3],
        ]
    )

    classifier_labels = ["Logistic Regression", "Decision Tree", "KNN"]

    classifier = MajorityVoteClassifier(classifiers=[pipe1, classifier2, pipe3])
    classifier_labels += ["Majority Voting"]
    all_classifier = [pipe1, classifier2, pipe3, classifier]

    x_train_std = scaler.fit_transform(x_train)

    x_min = x_train_std[:, 0].min() - 1
    x_max = x_train_std[:, 0].max() + 1
    y_min = x_train_std[:, 1].min() - 1
    y_max = x_train_std[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1),
    )

    f, axarr = plt.subplots(
        nrows=2,
        ncols=2,
        sharex="col",
        sharey="row",
        figsize=(7, 5),
    )

    for idx, clf, tt in zip(product([0, 1], [0, 1]), all_classifier, classifier_labels):
        clf.fit(x_train_std, y_train)

        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, z, alpha=0.3)
        axarr[idx[0], idx[1]].scatter(
            x_train_std[y_train == 0, 0],
            x_train_std[y_train == 0, 1],
            c="blue",
            marker="^",
            s=50,
        )
        axarr[idx[0], idx[1]].scatter(
            x_train_std[y_train == 1, 0],
            x_train_std[y_train == 1, 1],
            c="green",
            marker="o",
            s=50,
        )
        axarr[idx[0], idx[1]].set_title(tt)

    plt.text(
        -3.5,
        -5,
        s="Sepal width [standardized]",
        ha="center",
        va="center",
        fontsize=12,
    )
    plt.text(
        -12.5,
        4.5,
        s="Petal width [standardized]",
        ha="center",
        va="center",
        fontsize=12,
        rotation=90,
    )
    plt.savefig("../../figure/test_voteclassifier_grid.png")
