from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def wine_data_gen():
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

    x_data, y_data = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=0, stratify=y_data
    )

    return x_train, x_test, y_train, y_test, df_wine
