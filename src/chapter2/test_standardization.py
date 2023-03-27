from adaline import AdalineGD
from region_plot import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv("../../data/iris.data", header=None)

    # 1-100行目の目的変数の抽出
    y = df.iloc[0:100, 4].values

    # Iris-setosaを-1, Iris-versicolorを1に変換
    y = np.where(y == "Iris-setosa", -1, 1)

    # 1-100行目の1,3列目の抽出
    X = df.iloc[0:100, [0, 2]].values

    X_std = np.copy(X)

    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    ada_gd = AdalineGD(n_iter=15, eta=0.01)
    ada_gd.fit(X_std, y)

    plot_decision_regions(X_std, y, classifier=ada_gd)

    plt.title("Adaline - Gradient Descent")

    plt.xlabel("sepal length [standardized]")
    plt.ylabel("petal length [standardized]")

    plt.legend(loc="upper left")

    plt.tight_layout()

    plt.savefig("../../figure/Adaline_std_region.png")

    plt.figure()

    plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker="o")

    plt.xlabel("Epochs")
    plt.ylabel("Sum-squared-error")

    plt.tight_layout()
    plt.savefig("../../figure/Adaline_std_cost.png")
