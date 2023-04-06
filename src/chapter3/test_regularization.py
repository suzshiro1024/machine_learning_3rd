from data_gen import data_gen
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    X_train_std, X_test_std, y_train, y_test = data_gen()

    weights, params = [], []

    for c in np.arange(-5, 5):
        lr = LogisticRegression(
            C=10.0 ** c, random_state=1, solver="lbfgs", multi_class="ovr"
        )
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(10.0 ** c)

    weights = np.array(weights)

    plt.plot(params, weights[:, 0], label="petal length")
    plt.plot(params, weights[:, 1], linestyle="--", label="petal width")

    plt.ylabel("weight coefficient")
    plt.xlabel("C")
    plt.legend(loc="upper left")
    plt.xscale("log")

    plt.savefig("../../figure/test_refularization.png")
