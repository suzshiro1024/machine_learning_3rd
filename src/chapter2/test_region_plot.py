from region_plot import plot_decision_regions
from perceptron import Perceptron

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("../../data/iris.data", header=None)

    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", -1, 1)

    X = df.iloc[0:100, [0, 2]].values

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    plot_decision_regions(X, y, classifier=ppn)

    plt.xlabel("sepal length [cm]")
    plt.ylabel("petal length [cm]")

    plt.legend(loc="upper left")

    plt.savefig("../../figure/region_plot_test.png")
