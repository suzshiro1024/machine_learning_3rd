from xor_data_gen import xor_data_gen
from data_gen import data_gen
from region_plot import plot_decision_regions
from sklearn.svm import SVC


import numpy as np
import matplotlib.pyplot as plt


def main():
    X_xor, y_xor = xor_data_gen()

    svm = SVC(kernel="rbf", random_state=1, gamma=0.10, C=10.0)
    svm.fit(X_xor, y_xor)

    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("../../figure/test_svm_rbf.png")

    X_train_std, X_test_std, y_train, y_test = data_gen()

    svm2 = SVC(kernel="rbf", random_state=1, gamma=0.2, C=1.0)
    svm2.fit(X_train_std, y_train)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    plt.figure()

    plot_decision_regions(
        X_combined_std, y_combined, classifier=svm2, test_idx=range(105, 150)
    )

    plt.xlabel("petal length [standardized]")
    plt.ylabel("petal width [standardized]")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("../../figure/test_svm_rbf_iris1.png")

    svm3 = SVC(kernel="rbf", random_state=1, gamma=100.0, C=1.0)
    svm3.fit(X_train_std, y_train)

    plt.figure()

    plot_decision_regions(
        X_combined_std, y_combined, classifier=svm3, test_idx=range(105, 150)
    )

    plt.xlabel("petal length [standardized]")
    plt.ylabel("petal width [standardized]")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("../../figure/test_svm_rbf_iris2.png")


if __name__ == "__main__":
    main()
