from sklearn.model_selection import train_test_split
from sklearn import datasets
from region_plot import plot_decision_regions
from pydotplus import graph_from_dot_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    iris = datasets.load_iris()

    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )

    tree_model = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=1)
    tree_model.fit(X_train, y_train)

    dot_data = export_graphviz(
        tree_model,
        filled=True,
        rounded=True,
        class_names=["Setosa", "Versicolor", "Virginica"],
        feature_names=["petal length", "petal width"],
        out_file=None,
    )

    graph = graph_from_dot_data(dot_data)
    graph.write_png("../../figure/test_tree_graphviz.png")
