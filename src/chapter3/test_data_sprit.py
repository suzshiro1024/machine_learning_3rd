from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np


def main():
    iris = datasets.load_iris()

    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )

    print(f"Label counts in y: {np.bincount(y)}")
    print(f"Label counts in y_train: {np.bincount(y_train)}")
    print(f"Label counts in y_test: {np.bincount(y_test)}")


if __name__ == "__main__":
    main()
