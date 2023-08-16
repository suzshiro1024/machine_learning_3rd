from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from category_gen import category_gen


def main():
    df = category_gen()

    size_mapping = {"XL": 3, "L": 2, "M": 1}
    df["size"] = df["size"].map(size_mapping)

    X = df[["color", "size", "price"]].values
    c_transf = ColumnTransformer(
        [("onehot", OneHotEncoder(), [0]), ("nothing", "passthrough", [1, 2])]
    )

    X_ohe = c_transf.fit_transform(X).astype(float)

    print(f"X\n{X_ohe}")


if __name__ == "__main__":
    main()
