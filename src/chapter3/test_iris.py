from sklearn import datasets
import numpy as np


def main():
    iris = datasets.load_iris()

    X = iris.data[:, [2, 3]]
    y = iris.target

    print(f"Class labels: {np.unique(y)}")


if __name__ == "__main__":
    main()
