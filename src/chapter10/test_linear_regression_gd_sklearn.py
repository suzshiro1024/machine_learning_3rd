from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

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

    slr = LinearRegression()
    slr.fit(x, y)
    y_pred = slr.predict(x)

    print(f"Slope: {slr.coef_[0]:.3f}")
    print(f"Intercept: {slr.intercept_:.3f}")

    lin_regplot(x, y, slr)
    plt.xlabel("Average number of room [RM]")
    plt.ylabel("Price in $1000s [MEDV]")
    plt.savefig("../../figure/test_linear_regression_gd_sklearn.png")
