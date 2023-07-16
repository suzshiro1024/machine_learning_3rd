from mlxtend.plotting import heatmap
import pandas as pd
import numpy as np
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

    cm = np.corrcoef(df[cols].values.T)
    hm = heatmap(cm, row_names=cols, column_names=cols)

    plt.savefig("../../figure/test_housing_heatmap.png")
