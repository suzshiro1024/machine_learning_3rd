from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from gen_bcw_data import gen_bcw_data

import numpy as np
import pandas as pd

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = gen_bcw_data()

    pipe_lr = make_pipeline(
        StandardScaler(),
        PCA(n_components=2),
        LogisticRegression(random_state=1, solver="lbfgs"),
    )

    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)

    print(f"Test Accuracy : {pipe_lr.score(X_test, y_test)}")
