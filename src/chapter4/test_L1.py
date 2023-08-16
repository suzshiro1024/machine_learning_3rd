from wine_gen import wine_data_gen
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def main():
    X_train, X_test, y_train, y_test = wine_data_gen()

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    lr = LogisticRegression(penalty="l1", C=1.0, solver="liblinear", multi_class="ovr")
    lr.fit(X_train_std, y_train)

    print(f"Training Accuracy: {lr.score(X_train_std, y_train)}")
    print(f"Test Accuracy: {lr.score(X_test_std, y_test)}")


if __name__ == "__main__":
    main()
