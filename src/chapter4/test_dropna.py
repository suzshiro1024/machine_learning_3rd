from nan_gen import nan_gen

if __name__ == "__main__":
    df = nan_gen()
    print(f"dropna()\n{df.dropna()}")

    print(f"dropna(axis=1)\n{df.dropna(axis=1)}")

    drop_all = df.dropna(how="all")
    print(f"dropna(how=all)\n{drop_all}")

    drop_any = df.dropna(how="any")
    print(f"dropna(how=any)\n{drop_any}")

    print(f"dropna(thresh=4)\n{df.dropna(thresh=4)}")

    df_subset = df.dropna(subset=["C"])
    print(f"dropna(subset=[C])\n{df_subset}")
