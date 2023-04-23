from sklearn.model_selection import train_test_split
from wine_gen import wine_data_gen

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = wine_data_gen()

    print(f"X_train\n{X_train[:5,:]}")
