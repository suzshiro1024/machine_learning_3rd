import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perceptron import Perceptron


if __name__ == "__main__":
    df = pd.read_csv("../../data/iris.data", header=None)

    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", -1, 1)

    X = df.iloc[0:100, [0, 2]].values

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Number of Update")

    plt.savefig("../../figure/iris_fitting.png")
