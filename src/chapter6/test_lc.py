from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from gen_bcw_data import gen_bcw_data

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = gen_bcw_data()

    pipe_lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2", random_state=1, solver="lbfgs", max_iter=10000
        ),
    )

    train_sizes, train_scores, test_scores = learning_curve(
        estimator=pipe_lr,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=10,
        n_jobs=1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(
        train_sizes,
        train_mean,
        color="blue",
        marker="o",
        markersize=5,
        label="Training Accuracy",
    )
    plt.fill_between(
        train_sizes,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color="blue",
    )

    plt.plot(
        train_sizes,
        test_mean,
        color="green",
        linestyle="--",
        marker="s",
        markersize=5,
        label="Validation Accuracy",
    )
    plt.fill_between(
        train_sizes,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color="green",
    )

    plt.grid()
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.ylim([0.8, 1.03])
    plt.tight_layout()

    plt.savefig("../../figure/test_lc.png")
