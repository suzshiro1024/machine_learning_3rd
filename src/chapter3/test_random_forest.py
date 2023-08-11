from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from region_plot import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import matplotlib.pyplot as plt


def main():
    iris = datasets.load_iris()

    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )

    forest = RandomForestClassifier(
        criterion="gini", n_estimators=25, random_state=1, n_jobs=2
    )

    forest.fit(X_train, y_train)

    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(
        X_combined, y_combined, classifier=forest, test_idx=range(105, 150)
    )

    plt.xlabel("petal length [cm]")
    plt.ylabel("petal width [cm]")
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig("../../figure/test_random_forest.png")


if __name__ == "__main__":
    main()
