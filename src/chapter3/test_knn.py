from data_gen import data_gen
from region_plot import plot_decision_regions
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import matplotlib.pyplot as plt


def main():
    X_train_std, X_test_std, y_train, y_test = data_gen()

    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")
    knn.fit(X_train_std, y_train)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(
        X_combined_std, y_combined, classifier=knn, test_idx=range(105, 150)
    )

    plt.xlabel("petal length [standardized]")
    plt.ylabel("petal width [standardized]")
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig("../../figure/test_knn.png")


if __name__ == "__main__":
    main()
