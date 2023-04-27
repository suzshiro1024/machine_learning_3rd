from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from itertools import combinations

import numpy as np


class SBS:
    """
    逐次後退選択 (sequential backward selection) を実行するクラス
    """

    def __init__(
        self,
        estimator,
        k_features,
        scoring=accuracy_score,
        test_size=0.25,
        random_state=1,
    ):
        self.scoreing = scoring  # 特徴量の評価指標
        self.estimator = clone(estimator)  # 推定器
        self.k_features = k_features  # 選択する特徴量の個数
        self.test_size = test_size  # テストデータの割合
        self.random_state = random_state  # 乱数シードの固定

    def fit(self, X, y):
        # 訓練データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # すべての特徴量の個数，列インデックス
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]

        # すべての特徴量を用いてスコアを算出する
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        # 特徴量の部分集合を表す列インデックスの組み合わせごとに処理を反復する
        while dim > self.k_features:
            scores = []
            subsets = []

            # 特徴量の部分集合を表す列インデックスの組み合わせごとに処理を反復
            for p in combinations(self.indices_, r=dim - 1):
                # スコアを算出して格納
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)

                # 特徴量の部分集合を表す列インデックスのリストを格納
                subsets.append(p)

            # 最良のスコアのインデックスを抽出
            best = np.argmax(scores)

            # 最良のスコアとなる列インデックスを抽出して格納
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)

            # 特徴量の個数を１つ減じて次のステップに入る
            dim -= 1

            # スコアを格納
            self.scores_.append(scores[best])

        # 最後に格納したスコア
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        # 抽出した特徴量を返す
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        # 指定された列番号indicesの特徴量を抽出してモデルを適合
        self.estimator.fit(X_train[:, indices], y_train)

        # テストデータを用いたクラスラベルの予測と真のクラスラベルからスコアを算出
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoreing(y_test, y_pred)
        return score
