from gen_bcw_data import gen_bcw_data

import numpy as np


def gen_imbalance_bcw_data():
    X_train, _, y_train, _ = gen_bcw_data()

    X_imb = np.vstack((X_train[y_train == 0], X_train[y_train == 1][:40]))
    y_imb = np.hstack((y_train[y_train == 0], y_train[y_train == 1][:40]))

    return X_imb, y_imb
