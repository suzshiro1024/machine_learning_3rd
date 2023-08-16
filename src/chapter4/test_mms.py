from wine_gen import wine_data_gen
from sklearn.preprocessing import MinMaxScaler


def main():
    X_train, X_test, y_train, y_test = wine_data_gen()

    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.transform(X_test)

    print(f"X_train\n{X_train_norm[:10]}")
    print(f"X_test\n{X_test_norm[:10]}")


if __name__ == "__main__":
    main()
