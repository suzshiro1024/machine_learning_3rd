from gen_wine import wine_data_gen
from region_plot import plot_decision_regions
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import matplotlib.pyplot as plt

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df = wine_data_gen()

    stdsc = StandardScaler()

    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train_std, y_train)

    lr = LogisticRegression(multi_class="ovr", random_state=1, solver="lbfgs")
    lr = lr.fit(X_train_lda, y_train)

    plot_decision_regions(X_train_lda, y_train, classifier=lr)
    plt.xlabel("LD 1")
    plt.ylabel("LD 2")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("../../figure/test_lda_sklearn_train.png")

    plt.figure()

    X_test_lda = lda.transform(X_test_std)
    plot_decision_regions(X_test_lda, y_test, classifier=lr)
    plt.xlabel("LD 1")
    plt.ylabel("LD 2")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("../../figure/test_lda_sklearn_test.png")
