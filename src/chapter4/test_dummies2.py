from category_gen import category_gen
import pandas as pd


def main():
    df = category_gen()

    df_dummies = pd.get_dummies(df[["price", "color", "size"]], drop_first=True)

    print(f"df\n{df_dummies}")


if __name__ == "__main__":
    main()
