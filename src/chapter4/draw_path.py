from wine_gen import wine_data_gen
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    df_wine = pd.read_csv("../../data/wine.data", header=None)

    X_train, X_test, y_train, y_test = wine_data_gen()

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    fig = plt.figure()
    ax = plt.subplot(111)

    colors = [
        "blue",
        "green",
        "red",
        "cyan",
        "magenta",
        "yellow",
        "black",
        "pink",
        "lightgreen",
        "lightblue",
        "gray",
        "indigo",
        "orange",
    ]

    weights, params = [], []

    for c in np.arange(-4.0, 6.0):
        lr = LogisticRegression(
            penalty="l1",
            C=10.0**c,
            solver="liblinear",
            multi_class="ovr",
            random_state=0,
        )

        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(10**c)

    weights = np.array(weights)

    for column, color in zip(range(weights.shape[1]), colors):
        plt.plot(
            params, weights[:, column], label=df_wine.columns[column + 1], color=color
        )

    plt.axhline(0, color="black", linestyle="--", linewidth=3)
    plt.xlim([10 ** (-5), 10**5])

    plt.ylabel("weight coefficient")
    plt.xlabel("C")

    plt.xscale("log")
    plt.legend(loc="upper left")
    ax.legend(loc="upper center", bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)

    plt.savefig("../../figure/L1_path.png")


if __name__ == "__main__":
    main()
