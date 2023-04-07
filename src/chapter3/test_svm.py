from data_gen import data_gen
from region_plot import plot_decision_regions
from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X_train_std, X_test_std, y_train, y_test = data_gen()

    svm = SVC(kernel="linear", C=1.0, random_state=1)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    svm.fit(X_train_std, y_train)
    plot_decision_regions(
        X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150)
    )

    plt.xlabel("petal length [standardized]")
    plt.ylabel("petal width [standardized]")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("../../figure/test_svm.png")
