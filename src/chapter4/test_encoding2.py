from sklearn.preprocessing import LabelEncoder
from category_gen import category_gen
import numpy as np


def main():
    df = category_gen()

    class_le = LabelEncoder()

    y = class_le.fit_transform(df["classlabel"].values)
    print(f"y\n{y}")

    inv = class_le.inverse_transform(y)
    print(f"inv\n{inv}")


if __name__ == "__main__":
    main()
