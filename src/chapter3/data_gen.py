from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets


def data_gen():
    """
    return values: X_train_std, X_test_std, y_train, y_test
    """
    scaler = StandardScaler()

    iris = datasets.load_iris()

    x_data = iris.data[:, [2, 3]]
    y_data = iris.target

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=1, stratify=y_data
    )

    scaler.fit(x_train)
    x_train_std = scaler.transform(x_train)
    x_test_std = scaler.transform(x_test)

    return x_train_std, x_test_std, y_train, y_test
