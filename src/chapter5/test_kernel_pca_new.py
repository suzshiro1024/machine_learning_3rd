from sklearn.datasets import make_moons
from kernel_pca import rbf_kernel_pca
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import matplotlib.pyplot as plt


def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row) ** 2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)


if __name__ == "__main__":
    X, y = make_moons(n_samples=100, random_state=123)
    alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)

    x_new = X[25]
    print(f"x_new\n{x_new}")

    x_proj = alphas[25]
    print(f"x_proj\n{x_proj}")

    x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
    print(f"x_reproj\n{x_reproj}")
    plt.scatter(alphas[y == 0, 0], np.zeros((50)), color="red", marker="^", alpha=0.5)
    plt.scatter(alphas[y == 1, 0], np.zeros((50)), color="blue", marker="o", alpha=0.5)
    plt.scatter(
        x_proj,
        0,
        color="black",
        label="Original projection of point X[25]",
        marker="^",
        s=100,
    )
    plt.scatter(
        x_reproj, 0, color="green", label="Remapped point X[25]", marker="x", s=500
    )
    plt.yticks([], [])
    plt.legend(scatterpoints=1)
    plt.tight_layout()

    plt.savefig("../../figure/test_kernel_pca_new.png")
