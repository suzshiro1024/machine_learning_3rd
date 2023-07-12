from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


if __name__ == "__main__":
    iris = datasets.load_iris()

    x_data, y_data = iris.data[50:, [1, 2]], iris.target[50:]

    encoder = LabelEncoder()
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
            ["sc", StandardScaler()],
            ["clf", classifier1],
        ]
    )

    pipe3 = Pipeline(
        [
            ["sc", StandardScaler()],
            ["clf", classifier3],
        ]
    )

    classifier_labels = ["Logistic Regression", "Decision Tree", "KNN"]
    print("10-fold cross validation\n")

    for clf, label in zip([pipe1, classifier2, pipe3], classifier_labels):
        scores = cross_val_score(
            estimator=clf,
            X=x_train,
            y=y_train,
            cv=10,
            scoring="roc_auc",
        )

        print(f"ROC AUC: {scores.mean():.2f} (+/- {scores.std():.2f}) [{label}]")
