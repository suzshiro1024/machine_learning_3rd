from wine_gen import wine_data_gen
from sbs import SBS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pandas as pd


def main():
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

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)

    sbs = SBS(knn, k_features=1)

    sbs.fit(X_train_std, y_train)

    k_feat = [len(k) for k in sbs.subsets_]

    plt.plot(k_feat, sbs.scores_, marker="o")
    plt.ylim([0.7, 1.02])
    plt.ylabel("Accuracy")
    plt.xlabel("Number of features")
    plt.grid()
    plt.tight_layout()
    plt.savefig("../../figure/test_sbs.png")

    k3 = list(sbs.subsets_[10])
    print(f"{df_wine.columns[1:][k3]}")

    print()

    knn.fit(X_train_std, y_train)

    print(f"Training Accuracy:{knn.score(X_train_std, y_train)}")
    print(f"Test Accuracy:{knn.score(X_test_std,y_test)}")

    print()

    knn.fit(X_train_std[:, k3], y_train)
    print(f"Training Accuracy:{knn.score(X_train_std[:,k3], y_train)}")
    print(f"Test Accuracy:{knn.score(X_test_std[:,k3],y_test)}")


if __name__ == "__main__":
    main()
