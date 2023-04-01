from data_gen import data_gen
from sklearn.linear_model import Perceptron
from region_plot import plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X_train_std, X_test_std, y_train, y_test = data_gen()

    ppn = Perceptron(eta0=0.01, random_state=1)
    ppn.fit(X_train_std, y_train)

    y_pred = ppn.predict(X_test_std)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(
        X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150)
    )

    plt.xlabel("petal length [standardized]")
    plt.ylabel("petal width [standardized]")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("../../figure/region_plot_test_sklearn.png")
