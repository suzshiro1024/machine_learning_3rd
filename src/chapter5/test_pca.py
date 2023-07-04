from gen_wine import wine_data_gen
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df_wine = wine_data_gen()

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

    print(f"Eigenvalues\n{eigen_vals}")

    tot = sum(eigen_vals)

    ver_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_ver_exp = np.cumsum(ver_exp)

    plt.bar(
        range(1, 14),
        ver_exp,
        alpha=0.5,
        align="center",
        label="Individual Explained Variance",
    )
    plt.step(
        range(1, 14), cum_ver_exp, where="mid", label="Cumulative Explained Variance"
    )
    plt.ylabel("Explained Variance Ratio")
    plt.xlabel("Principal Component Index")
    plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig("../../figure/test_ver.png")
    plt.figure()

    eigen_pairs = [
        (np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))
    ]

    eigen_pairs.sort(key=lambda k: k[0], reverse=True)

    W = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    print(f"Matrix W\n{W}")

    X_train_pca = X_train_std.dot(W)
    print(f"X_train_pca\n{X_train_pca}")

    colors = ["r", "b", "g"]
    markers = ["s", "x", "o"]

    for n, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(
            X_train_pca[y_train == n, 0],
            X_train_pca[y_train == n, 1],
            c=c,
            label=1,
            marker=m,
        )

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("../../figure/test_pca.png")
