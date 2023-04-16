from nan_gen import nan_gen

if __name__ == "__main__":
    df = nan_gen()
    print(df.isnull().sum())
