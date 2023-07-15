from mlxtend.plotting import scatterplotmatrix
import pandas as pd
import matplotlib.pyplot as plt

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

    scatterplotmatrix(df[cols].values, figsize=(10, 8), names=cols, alpha=0.5)
    plt.tight_layout()
    plt.savefig("../../figure/test_housing_eda.png")
