from gen_imbalance_bcw_data import gen_imbalance_bcw_data

import numpy as np

if __name__ == "__main__":
    X_imb, y_imb = gen_imbalance_bcw_data()

    y_pred = np.zeros(y_imb.shape[0])
    print(f"Accuracy: {np.mean(y_pred == y_imb)*100}")
