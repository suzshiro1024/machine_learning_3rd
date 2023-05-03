from wine_gen import wine_data_gen
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df_wine = wine_data_gen()

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    np.set_printoptions(precision=4)

    mean_vecs = []
    for label in range(1, 4):
        mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
        print(f"MV: {label}: {mean_vecs[label-1]}")

    dim = 13
    S_W = np.zeros([dim, dim])

    for label, mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.zeros([dim, dim])
        # for row in X_train_std[y_train == label]:
        #     row, mv = row.reshape(dim, 1), mv.reshape(dim, 1)
        #     class_scatter += (row - mv).dot((row - mv).T)
        class_scatter = np.cov(X_train_std[y_train == label].T)
        S_W += class_scatter

    print(f"Within-class scatter matrix: {S_W.shape[0]}x{S_W.shape[1]}")
    print(f"Class label distribution: {np.bincount(y_train)[1:]}")

    mean_overall = np.mean(X_train_std, axis=0)
    S_B = np.zeros([dim, dim])

    for i, mean_vec in enumerate(mean_vecs):
        n = X_train_std[y_train == i + 1, :].shape[0]
        mean_vec = mean_vec.reshape(dim, 1)
        mean_overall = mean_overall.reshape(dim, 1)
        S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

    print(f"Between-class scatter matrix: {S_B.shape[0]}x{S_B.shape[1]}")

    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    eigen_pairs = [
        (np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))
    ]

    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

    print("Eigenvalues in descending order:\n")
    for eigen_val in eigen_pairs:
        print(eigen_val[0])

    tot = sum(eigen_vals.real)

    discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
    cum_discr = np.cumsum(discr)

    plt.bar(
        range(1, 14),
        discr,
        alpha=0.5,
        align="center",
        label="Individual 'discriminability'",
    )
    plt.step(
        range(1, 14), cum_discr, where="mid", label="Cumulative 'Discriminability'"
    )
    plt.ylabel("'Discriminability' Ratio")
    plt.xlabel("Linear Discriminants")
    plt.ylim([-0.1, 1.1])
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("../../figure/test_discriminability.png")

    plt.figure()
    W = np.hstack(
        (eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real)
    )
    print(f"Matrix W:\n{W}")

    X_train_lda = X_train_std.dot(W)
    colors = ["r", "b", "g"]
    markers = ["s", "x", "o"]
    for la, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(
            X_train_lda[y_train == la, 0],
            X_train_lda[y_train == la, 1] * (-1),
            c=c,
            label=la,
            marker=m,
        )

    plt.xlabel("LD 1")
    plt.ylabel("LD 2")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("../../figure/test_lda.png")
