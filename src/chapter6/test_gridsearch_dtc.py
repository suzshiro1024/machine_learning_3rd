from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from gen_bcw_data import gen_bcw_data

import numpy as np

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = gen_bcw_data()

    gs = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=0),
        param_grid=[{"max_depth": [1, 2, 3, 4, 5, 6, 7, None]}],
        scoring="accuracy",
        cv=2,
    )

    scores = cross_val_score(gs, X_train, y_train, scoring="accuracy", cv=5)

    print(f"CV Accuracy: {np.mean(scores)} +/- {np.std(scores)}")