from data_gen import data_gen
from region_plot import plot_decision_regions
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt


def main():
    X_train_std, X_test_std, y_train, y_test = data_gen()

    lr = LogisticRegression(C=100.0, random_state=1, solver="lbfgs", multi_class="ovr")
    lr.fit(X_train_std, y_train)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(
        X=X_combined_std, y=y_combined, classifier=lr, test_idx=range(105, 150)
    )

    plt.xlabel("petal length [standardized]")
    plt.ylabel("petal width [standardized]")
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig("../../figure/test_lr_sklearn.png")


if __name__ == "__main__":
    main()
