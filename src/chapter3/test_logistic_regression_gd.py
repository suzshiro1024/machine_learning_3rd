import os
import sys

sys.path.append(os.pardir)

from data_gen import data_gen
from chapter2.region_plot import plot_decision_regions
from logistic_regression_gd import LogisticRegressionGD

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    X_train_std, X_test_std, y_train, y_test = data_gen()

    X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

    lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd.fit(X_train_01_subset, y_train_01_subset)

    plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=lrgd)
    plt.xlabel("petal length [standardized]")
    plt.ylabel("petal width [standardized]")
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig("../../figure/test_lrgd.png")
