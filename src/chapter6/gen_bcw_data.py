from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd


def gen_bcw_data():
    df = pd.read_csv("../../data/wdbc.data", header=None)

    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()

    y = le.fit_transform(y)
    print(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=1
    )

    return X_train, X_test, y_train, y_test
