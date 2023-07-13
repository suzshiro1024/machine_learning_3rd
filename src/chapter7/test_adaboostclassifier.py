from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from gen_wine import wine_data_gen

if __name__ == "__main__":
    x_train, x_test, y_train, y_test, df_wine = wine_data_gen()

    tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=1,
        random_state=1,
    )
    ada = AdaBoostClassifier(
        base_estimator=tree,
        n_estimators=500,
        learning_rate=0.1,
        random_state=1,
    )
    tree = tree.fit(x_train, y_train)
    y_train_predict = tree.predict(x_train)
    y_test_predict = tree.predict(x_test)

    tree_train = accuracy_score(y_train, y_train_predict)
    tree_test = accuracy_score(y_test, y_test_predict)
    print(f"Decision tree train/test accuracies {tree_train:.3f}/{tree_test:.3f}")

    ada = ada.fit(x_train, y_train)
    y_train_predict = ada.predict(x_train)
    y_test_predict = ada.predict(x_test)

    ada_train = accuracy_score(y_train, y_train_predict)
    ada_test = accuracy_score(y_test, y_test_predict)
    print(f"AdaBoost train/test accuracies: {ada_train:.3f}/{ada_test:.3f}")
