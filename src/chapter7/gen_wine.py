from sklearn.model_selection import train_test_split
import pandas as pd


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

    df_wine = df_wine[df_wine["Class label"] != 1]

    x_data, y_data = (
        df_wine[["Alcohol", "OD280/OD315 of diluted wines"]].values,
        df_wine["Class label"].values,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=1, stratify=y_data
    )

    return x_train, x_test, y_train, y_test, df_wine
