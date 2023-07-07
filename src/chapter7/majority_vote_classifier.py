from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """多数決アンサンブル分類器

    パラメータ
    ----------
    classifiers : array-like, shape = [n_classifiers]
        アンサンブルの様々な分類器

    vote : str, {'classlabel', 'probability'} (default : 'classlabel')
        'classlabel'の場合、クラスラベルの予測はクラスラベルのargmaxに基づく
        'probability'の場合、クラスラベルの予測はクラスの所属確率のargmaxに基づく(分類器が調整済みであることを推奨)

    weights : array-like, shape = [n_classifiers] (optimal, default=None)
        'int'または'float'型の値リストが提供された場合、分類器は重要度で重みづけされる
        'weight=None'の場合は均一な重みを使用
    """

    def __init__(self, classifiers, vote="classlabel", weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {
            key: value for key, value in _name_estimators(classifiers)
        }

        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """分類器を学習する

        パラメータ
        ----------
        X : {array-like, sparse matrix}, shape = [n_examples, n_features]
            訓練データから成る行列

        y : array-like, shape = [n_examples]
            クラスラベルのベクトル

        戻り値
        ----------
        self : object
        """
        if self.vote not in ("probability", "classlabel"):
            raise ValueError(
                f"vote mast be 'probability' or 'classlabel'; got {self.vote}"
            )

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(
                f"Number of classifiers and weights must be equal; got {len(self.weights)} weights, {len(self.classifiers)} classifiers"
            )

        # LabelEncoderを使ってクラスラベルが0から始まるようにエンコードする
        # self.predictのnp.argmax呼び出しで重要
        self.labelenc_ = LabelEncoder()
        self.labelenc_.fit(y)
        self.classes_ = self.labelenc_.classes_
        self.classifiers_ = []

        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.labelenc_.transform(y))
            self.classifiers_.append(fitted_clf)

        return self

    def predict(self, X):
        """Xのクラスラベルを予測する

        パラメータ
        ----------
        X : {array-like, sparse matrix}, shape = [n_examples, n_features]
            訓練データからなる行列

        戻り値
        ----------
        maj_vote : array-like, shape = [n_examples]
            予測されたクラスラベル
        """
        if self.vote == "probability":
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  #'classlabel'での多数決
            # clf.predict呼び出しの結果を実装
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T

            # 各データ点のクラス確率に重みをつけて足し合わせた値が最大となる、列番号を配列にして返す
            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=predictions,
            )

        # 各データ点に確率の最大値を与えるクラスラベルを抽出
        maj_vote = self.labelenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """Xのクラス確率を予測する

        パラメータ
        ----------
        X : {array-like, sparse matrix}, shape = [n_examples, n_features]
            訓練ベクトル：n_examplesはデータの個数、n_featuresは特徴量の個数

        戻り値
        ----------
        avg_proba : array-like, shape = [n_examples, n_classes]
            各データ点に対する各クラスで重みをつけた平均確率
        """
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """GridSearchの実行時に分類器のパラメータ名を取得"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            # キーを"分類器の名前__パラメータ名",値をパラメータの値とするディクショナリを作成
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out[f"{name}__{key}"] = value

            return out
