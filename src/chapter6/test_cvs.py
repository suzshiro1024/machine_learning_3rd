from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from gen_bcw_data import gen_bcw_data

import numpy as np


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = gen_bcw_data()

    pipe_lr = make_pipeline(
        StandardScaler(),
        PCA(n_components=2),
        LogisticRegression(random_state=1, solver="lbfgs"),
    )

    scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=1)

    print(f"CV accuracy scores: {scores}")
    print(f"CV accuracy: {np.mean(scores)} +/- {np.std(scores)}")
