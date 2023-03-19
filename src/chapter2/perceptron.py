import numpy as np


class Perceptron(object):
    """
    パーセプトロン分類器\n

    Args:\n
    -----\n
    eta (float) : 学習率(0.0 < eta < 1.0)
    n_iter (int) : 訓練データの訓練回数
    random_state (int) : 重みを初期化するための乱数シード

    Attributes:\n
    -----------\n
    w_ (一次元配列) : 適合後の重み
    errors_ (リスト) : 各エポックでの誤分類の回数
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
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

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # 重みw_1, ... , w_mの更新
                # Δw_j = η(y - y^)x_j (j = 1, ... ,m)
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi

                # 重みw_0の更新
                # Δw_0 = η(y - y^)
                self.w_[0] += update

                # 重みの更新が0出ない場合は誤分類
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        総入力を計算する
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        1ステップ後のクラスラベルを返す
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
