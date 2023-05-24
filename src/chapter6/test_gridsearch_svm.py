from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from gen_bcw_data import gen_bcw_data

import numpy as np

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = gen_bcw_data()

    pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    param_grid = [
        {"svc__C": param_range, "svc__kernel": ["linear"]},
        {"svc__C": param_range, "svc__gamma": param_range, "svc__kernel": ["rbf"]},
    ]

    gs = GridSearchCV(
        estimator=pipe_svc,
        param_grid=param_grid,
        scoring="accuracy",
        cv=10,
        refit=True,
        n_jobs=-1,
    )

    gs.fit(X_train, y_train)

    print(f"Best score: {gs.best_score_}")
    print(f"Best params: {gs.best_params_}")

    clf = gs.best_estimator_
    clf.fit(X_train, y_train)

    print(f"Test Accuracy: {clf.score(X_test, y_test)}")

    scores = cross_val_score(gs, X_train, y_train, scoring="accuracy", cv=5)

    print(f"CV Accuracy: {np.mean(scores)} +/- {np.std(scores)}")
