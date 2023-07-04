from region_plot import plot_decision_regions
from gen_wine import wine_data_gen
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df_wine = wine_data_gen()

    stdsc = StandardScaler()

    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    pca = PCA(n_components=2)
    lr = LogisticRegression(multi_class="ovr", random_state=1, solver="lbfgs")

    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    lr.fit(X_train_pca, y_train)

    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(loc="lower left")
    plt.tight_layout()

    plt.savefig("../../figure/test_pca_region_plot_train.png")

    plt.figure()

    plot_decision_regions(X_test_pca, y_test, classifier=lr)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(loc="lower left")
    plt.tight_layout()

    plt.savefig("../../figure/test_pca_region_plot_test.png")
