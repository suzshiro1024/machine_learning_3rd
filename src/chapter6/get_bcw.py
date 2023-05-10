import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        header=None,
    )

    df.to_csv("../../data/iris.data")
