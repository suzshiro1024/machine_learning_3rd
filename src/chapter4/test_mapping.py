from category_gen import category_gen


def main():
    df = category_gen()

    size_mapping = {"XL": 3, "L": 2, "M": 1}
    inv_size_mapping = {v: k for k, v in size_mapping.items()}

    df["size"] = df["size"].map(size_mapping)
    print(f"df\n{df}")

    df["size"] = df["size"].map(inv_size_mapping)
    print(f"df\n{df}")


if __name__ == "__main__":
    main()
