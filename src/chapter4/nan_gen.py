import pandas as pd
from io import StringIO


def nan_gen():
    csv_data = """A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,
    """

    df = pd.read_csv(StringIO(csv_data))
    return df


def main():
    print(nan_gen())


if __name__ == "__main__":
    main()
