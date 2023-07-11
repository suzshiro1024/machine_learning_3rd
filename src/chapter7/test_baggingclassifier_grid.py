from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from gen_wine import wine_data_gen

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x_train, x_test, y_train, y_test, df_wine = wine_data_gen()

    tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=None,
        random_state=1,
    )

    bag = BaggingClassifier(
        base_estimator=tree,
        n_estimators=500,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        n_jobs=1,
        random_state=1,
    )

    tree = tree.fit(x_train, y_train)

    x_min = x_train[:, 0].min() - 1
    x_max = x_train[:, 0].max() + 1
    y_min = x_train[:, 1].min() - 1
    y_max = x_train[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1),
    )
    f, axarr = plt.subplots(
        nrows=1,
        ncols=2,
        sharex="col",
        sharey="row",
        figsize=(8, 3),
    )
    for idx, clf, tt in zip([0, 1], [tree, bag], ["Decision tree", "Bagging"]):
        clf.fit(x_train, y_train)
        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        axarr[idx].contourf(xx, yy, z, alpha=0.3)
        axarr[idx].scatter(
            x_train[y_train == 0, 0], x_train[y_train == 0, 1], c="blue", marker="^"
        )
        axarr[idx].scatter(
            x_train[y_train == 1, 0], x_train[y_train == 1, 1], c="green", marker="o"
        )
        axarr[idx].set_title(tt)

    axarr[0].set_ylabel("Alcohol", fontsize=12)
    plt.tight_layout()
    plt.text(
        0,
        -0.2,
        s="OD280/OD315 of diluted wines",
        ha="center",
        va="center",
        fontsize=12,
        transform=axarr[1].transAxes,
    )

    plt.savefig("../../figure/test_baggingclassifier_grid.png")
