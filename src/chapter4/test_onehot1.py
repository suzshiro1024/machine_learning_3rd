from sklearn.preprocessing import OneHotEncoder
from category_gen import category_gen

if __name__ == "__main__":
    df = category_gen()
    X = df[["color", "size", "price"]].values

    color_ohe = OneHotEncoder()

    X_ohe = color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()

    print(f"X\n{X_ohe}")
