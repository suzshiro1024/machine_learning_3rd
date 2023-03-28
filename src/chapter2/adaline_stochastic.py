import numpy as np
from numpy.random import seed


class AdalineSGD(object):
    """
    ADAptive LInear NEuron分類器(ADALINE)\n

    Args:\n
    -----\n
    eta (float) : 学習率(0.0 < eta < 1.0)
    n_iter (int) : 訓練データの訓練回数
    w_initialized (bool) : 各エポックで訓練データをシャッフルするかどうかのフラグ
    shuffle (bool) : Trueの場合は、循環回避のためにエポックごとに訓練データをシャッフルする(デフォルト : True)
    random_state (int) : 重みを初期化するための乱数シード

    Attributes:\n
    -----------\n
    w_ (一次元配列) : 適合後の重み
    cost_ (リスト) : 各エポックでの誤差平方和のコスト関数
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        """
        訓練データへの適合を行う関数\n

        Args:\n
        -----\n
        X (配列に類似したデータ構造), shape = [n_examples, n_features]: 訓練データ.n_examplesは訓練データの個数、n_featuresは特徴量の個数\n
        y (配列に類似したデータ構造), shape = [n_examples]: 目的変数\n

        Return Value:\n
        -------------\n
        self(object)
        """

        # 特徴量分の次元を持つ重みベクトルの生成
        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            # シャッフルをすることになっている場合、訓練データをシャッフルする
            if self.shuffle:
                X, y = self._shuffle(X, y)

            # 各訓練データごとにコストをそれぞれ計算するので、リストを用意する
            cost = []

            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))

            # 平均コストを計算しそれを格納する
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):
        """
        重みを再初期化することなく訓練データへ適合させる関数\n
        """
        # 初期化前の場合は初期化
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])

        # 目的変数yの要素数が2以上の場合は、各訓練データの特徴量xiと目的変数targetで重みを更新する
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        # 目的変数yの要素数が1の場合は、訓練データ全体の特徴量Xと目的変数yで重みを更新する
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """
        訓練データをシャッフルする関数\n
        """
        # 長さyで0からy-1までのすべての整数を含むベクトルを生成し、それをインデックスにすることでシャッフルする
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """
        重みを小さな乱数に初期化する関数\n
        """
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """
        ADALINEの学習規則を用いて重みを更新する関数\n
        """
        output = self.activation(self.net_input(xi))
        error = target - output

        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
