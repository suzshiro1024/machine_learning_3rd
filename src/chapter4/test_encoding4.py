from category_gen import category_gen

if __name__ == "__main__":
    df = category_gen()

    df["x > M"] = df["size"].apply(lambda x: 1 if x in {"L", "XL"} else 0)
    df["x > L"] = df["size"].apply(lambda x: 1 if x == "XL" else 0)

    del df["size"]

    print(f"df\n{df}")
