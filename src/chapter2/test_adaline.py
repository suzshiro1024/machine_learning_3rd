from adaline import AdalineGD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("../../data/iris.data", header=None)

    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", -1, 1)

    X = df.iloc[0:100, [0, 2]].values

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)

    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker="o")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("log(Sum-squared-error)")
    ax[0].set_title("Adalines - Learning rate 0.01")

    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)

    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker="o")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Sum-squared-error")
    ax[1].set_title("Adalines - Learning rate 0.01")

    plt.savefig("../../figure/adaline.png")


if __name__ == "__main__":
    main()
