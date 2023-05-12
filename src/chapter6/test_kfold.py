from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from gen_bcw_data import gen_bcw_data

import numpy as np

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = gen_bcw_data()

    kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
    scores = []

    pipe_lr = make_pipeline(
        StandardScaler(),
        PCA(n_components=2),
        LogisticRegression(random_state=1, solver="lbfgs"),
    )

    for k, (train, test) in enumerate(kfold):
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)
        print(f"Fold: {k+1}, Class dist.: {np.bincount(y_train[train])}, Acc: {score}")

    print(f"CV accuracy: {np.mean(scores)} +/- {np.std(scores)}")
