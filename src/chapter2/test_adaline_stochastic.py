from adaline_stochastic import AdalineSGD
from region_plot import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("../../data/iris.data", header=None)

    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", -1, 1)

    X = df.iloc[0:100, [0, 2]].values

    X_std = np.copy(X)

    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada_sgd.fit(X_std, y)

    plot_decision_regions(X_std, y, classifier=ada_sgd)

    plt.title("Adaline - Stochastic Gradient Descent")
    plt.xlabel("sepal length [standardized]")
    plt.ylabel("petal length [standardized]")
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig("../../figure/test_Adaline_SGD_region.png")

    plt.figure()

    plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_, marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Average Cost")
    plt.tight_layout()

    plt.savefig("../../figure/test_Adaline_SGD_cost.png")


if __name__ == "__main__":
    main()
