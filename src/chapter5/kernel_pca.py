from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh

import numpy as np


def rbf_kernel_pca(X, gamma, n_components):
    """
    RBFカーネルPCAの実装\n

    Args:\n
    -----\n
    X (Numpy ndarray), shape = [n_examples, n_features]\n
    gamma (float) : RBFカーネルのチューニングパラメータ\n
    n_components (int) : 返される成分の個数\n

    Attributes:\n
    -----------\n
    alphas (Numpy ndarray), shape = [n_examples, n_features] : 射影されたデータセット\n
    lamdas (list) : 固有値
    """

    # M × N次元のデータセットでペアごとのユークリッド距離の2乗を計算
    sq_dists = pdist(X, "sqeuclidean")

    # ペアごとの距離を正方行列に変換(pdistは対称行列である距離行列の上三角要素をflattenして返すため、squareformで正方行列に戻す)
    mat_sq_dists = squareform(sq_dists)

    # 対称カーネル行列を計算
    K = exp(-gamma * mat_sq_dists)

    # カーネル行列を中心化
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # 中心化されたカーネル行列から固有対を取得する(eighは昇順で固有対を出力する)
    eigenvals, eigenvecs = eigh(K)
    eigenvals, eigenvecs = eigenvals[::-1], eigenvecs[:, ::-1]

    # 上位k個の固有ベクトルを収集
    alphas = np.column_stack((eigenvecs[:, i] for i in range(n_components)))

    # 対応する固有値も収集
    lambdas = [eigenvals[i] for i in range(n_components)]

    return alphas, lambdas
