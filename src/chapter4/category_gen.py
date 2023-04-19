import pandas as pd


def category_gen():
    df = pd.DataFrame(
        [
            ["green", "M", 10.1, "class2"],
            ["red", "L", 13.5, "class1"],
            ["blue", "XL", 15.3, "class2"],
        ]
    )

    df.columns = ["color", "size", "price", "classlabel"]

    return df


if __name__ == "__main__":
    print(f"df\n{category_gen()}")
