from data_gen import data_gen
from sklearn.linear_model import LogisticRegression
import numpy as np


def main():
    X_train_std, X_test_std, y_train, y_test = data_gen()

    lr = LogisticRegression(C=100.0, random_state=1, solver="lbfgs", multi_class="ovr")
    lr.fit(X_train_std, y_train)

    print(f"prob:\n {lr.predict_proba(X_test_std[:3, :])}")

    # print(lr.predict_proba(X_test_std[:3, :]).sum(axis=1))

    print(f"predict:\n {lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)}")

    print(f"predict:\n {lr.predict(X_test_std[:3, :])}")

    # bad
    # print(f"predict:\n {lr.predict(X_test_std[0,:])}")

    # good (配列の次元がpredictで扱える二次元に自動で変更される)
    print(f"predict:\n {lr.predict(X_test_std[0,:].reshape(1,-1))}")


if __name__ == "__main__":
