from nan_gen import nan_gen


def main():
    df = nan_gen()
    print(df.isnull().sum())


if __name__ == "__main__":
    main()
