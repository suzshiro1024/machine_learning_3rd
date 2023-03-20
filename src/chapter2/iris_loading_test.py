import os
import pandas as pd


if __name__ == "__main__":
    path = os.path.join(
        "https://archive.ics.uci.edu",
        "ml",
        "machine-learning-databases",
        "iris",
        "iris.data",
    )

    print(f"URL: {path}")

    df = pd.read_csv(path, header=None, encoding="utf-8")
    df.tail()
