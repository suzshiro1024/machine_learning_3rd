from wine_gen import wine_data_gen
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df_wine = wine_data_gen()

    stdsc = StandardScaler()

    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    pca = PCA(n_components=None)
    lr = LogisticRegression(multi_class="ovr", random_state=1, solver="lbfgs")

    X_train_pca = pca.fit_transform(X_train_std)

    print(f"evr\n{pca.explained_variance_ratio_}")
