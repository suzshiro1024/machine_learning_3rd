from majority_vote_classifier import MajorityVoteClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    iris = datasets.load_iris()

    x_data, y_data = iris.data[50:, [1, 2]], iris.target[50:]

    encoder = LabelEncoder()
    scaler = StandardScaler()
    y_data = encoder.fit_transform(y_data)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.5, random_state=1, stratify=y_data
    )

    classifier1 = LogisticRegression(
        penalty="l2", C=0.001, solver="lbfgs", random_state=1
    )
    classifier2 = DecisionTreeClassifier(
        max_depth=1, criterion="entropy", random_state=0
    )
    classifier3 = KNeighborsClassifier(n_neighbors=1, p=2, metric="minkowski")

    pipe1 = Pipeline(
        [
            ["sc", scaler],
            ["clf", classifier1],
        ]
    )

    pipe3 = Pipeline(
        [
            ["sc", scaler],
            ["clf", classifier3],
        ]
    )

    classifier = MajorityVoteClassifier(classifiers=[pipe1, classifier2, pipe3])

    params = {
        "decisiontreeclassifier__max_depth": [1, 2],
        "pipeline-1__clf__C": [0.001, 0.1, 100.0],
    }

    grid = GridSearchCV(
        estimator=classifier,
        param_grid=params,
        cv=10,
        scoring="roc_auc",
    )
    grid.fit(x_train, y_train)

    for r, _ in enumerate(grid.cv_results_["mean_test_score"]):
        print(
            f"{grid.cv_results_['mean_test_score'][r]:.2f} +/- {grid.cv_results_['std_test_score'][r]/2.0:.2f} {grid.cv_results_['params'][r]}"
        )

    print(f"Best parameters: {grid.best_params_}")
    print(f"Accuracy: {grid.best_score_:.2f}")
