from sklearn.datasets import make_circles
from kernel_pca import rbf_kernel_pca
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

    X_spca, _ = rbf_kernel_pca(X, gamma=15, n_components=2)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
    ax[0].scatter(
        X_spca[y == 0, 0], X_spca[y == 0, 1], color="red", marker="^", alpha=0.5
    )
    ax[0].scatter(
        X_spca[y == 1, 0], X_spca[y == 1, 1], color="blue", marker="o", alpha=0.5
    )

    ax[1].scatter(
        X_spca[y == 0, 0], np.zeros((500, 1)) + 0.02, color="red", marker="^", alpha=0.5
    )
    ax[1].scatter(
        X_spca[y == 1, 0],
        np.zeros((500, 1)) - 0.02,
        color="blue",
        marker="o",
        alpha=0.5,
    )

    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel("PC1")

    plt.tight_layout()
    plt.savefig("../../figure/test_kernel_pca2.png")
