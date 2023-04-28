from wine_gen import wine_data_gen
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import numpy as np
import pandas as pd

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

    X_train, X_test, y_train, y_test = wine_data_gen()
    feat_labels = df_wine.columns[1:]

    forest = RandomForestClassifier(n_estimators=500, random_state=1)
    forest.fit(X_train, y_train)

    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    sfm = SelectFromModel(forest, threshold=0.1, prefit=True)

    X_selected = sfm.transform(X_train)
    print(
        f"Number of features that meet this threshold criterion: {X_selected.shape[1]}"
    )

    for f in range(X_selected.shape[1]):
        print(
            "%2d) %-*s %f"
            % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]])
        )
