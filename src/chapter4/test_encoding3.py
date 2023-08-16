from sklearn.preprocessing import LabelEncoder
from category_gen import category_gen


def main():
    df = category_gen()

    X = df[["color", "size", "price"]].values

    color_le = LabelEncoder()
    X[:, 0] = color_le.fit_transform(X[:, 0])

    print(f"X\n{X}")


if __name__ == "__main__":
    main()
