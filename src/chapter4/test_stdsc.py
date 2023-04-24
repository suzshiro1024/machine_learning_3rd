from wine_gen import wine_data_gen
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = wine_data_gen()

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    print(f"X_train\n{X_train_std[:10]}")
    print(f"X_test\n{X_test_std[:10]}")
