from sklearn.preprocessing import StandardScaler
from linear_regression_gd import LinearRegressionGD

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lin_regplot(x, y, model):
    plt.scatter(x, y, c="steelblue", edgecolor="white", s=70)
    plt.plot(x, model.predict(x), color="black", lw=2)
    return None


if __name__ == "__main__":
    df = pd.read_csv("../../data/housing.data.txt", header=None, sep="\s+")

    df.columns = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
        "MEDV",
    ]

    cols = ["LSTAT", "INDUS", "NOX", "RM", "MEDV"]

    x = df[["RM"]].values
    y = df["MEDV"].values
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x_std = sc_x.fit_transform(x)
    y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
    lr = LinearRegressionGD()
    lr.fit(x_std, y_std)

    lin_regplot(x_std, y_std, lr)
    plt.xlabel("Average number of rooms [RM] (standardized)")
    plt.ylabel("Price in $1000s [MEDV] (standardized)")
    plt.savefig("../../figure/test_linear_regression_gd_regplot.png")
