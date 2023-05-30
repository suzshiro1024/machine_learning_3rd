from gen_imbalance_bcw_data import gen_imbalance_bcw_data
from sklearn.utils import resample

import numpy as np

if __name__ == "__main__":
    X_imb, y_imb = gen_imbalance_bcw_data()

    print(f"Number of class 1 examples before: {X_imb[y_imb == 1].shape[0]}")

    X_upsampled, y_upsampled = resample(
        X_imb[y_imb == 1],
        y_imb[y_imb == 1],
        replace=True,
        n_samples=X_imb[y_imb == 0].shape[0],
        random_state=123,
    )

    print(f"Number of class 1 example after: {X_upsampled.shape[0]}")

    X_bal = np.vstack((X_imb[y_imb == 0], X_upsampled))
    y_bal = np.hstack((y_imb[y_imb == 0], y_upsampled))

    y_pred = np.zeros(y_bal.shape[0])
    print(f"Accuracy: {np.mean(y_pred==y_bal)*100}")
