from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from gen_wine import wine_data_gen

if __name__ == "__main__":
    x_train, x_test, y_train, y_test, df_wine = wine_data_gen()

    tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=None,
        random_state=1,
    )

    bag = BaggingClassifier(
        base_estimator=tree,
        n_estimators=500,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        n_jobs=1,
        random_state=1,
    )

    tree = tree.fit(x_train, y_train)
    y_train_predict = tree.predict(x_train)
    y_test_predict = tree.predict(x_test)

    tree_train = accuracy_score(y_train, y_train_predict)
    tree_test = accuracy_score(y_test, y_test_predict)
    print(f"Decision tree train/test accuracies {tree_train:.3f}/{tree_test:.3f}")

    bag = bag.fit(x_train, y_train)
    y_train_predict = bag.predict(x_train)
    y_test_predict = bag.predict(x_test)

    bag_train = accuracy_score(y_train, y_train_predict)
    bag_test = accuracy_score(y_test, y_test_predict)

    print(f"Bagging train/test accuracies {bag_train:.3f}/{bag_test:.3f}")
