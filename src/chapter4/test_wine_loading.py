import pandas as pd
import numpy as np

if __name__ == "__main__":
    df_wine = pd.read_csv("../../data/wine.data", header=None)

    df_wine.columns = [
        "Class label",
        "Alcohol",
        "Malic acid",
        "Ash",
        "Alcalinity of ash",
        "Magnesium",
        "Total phenols",
        "Flavanoids",
        "Nonflavanoid phenols",
        "Proanthocyanins",
        "Color intensity",
        "Hue",
        "OD280/OD315 of diluted wines",
        "Proline",
    ]

    c_label = np.unique(df_wine["Class label"])
    print(f"Class labels\n{c_label}")
    print(f"df\n{df_wine.head()}")
