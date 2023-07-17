import numpy as np


class LinearRegressionGD(object):
    def __init__(self, learning_rate=0.001, n_iter=20):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, x, y):
        self.w_ = np.zeros(1 + x.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(x)
            errors = y - output
            self.w_[1:] += self.learning_rate * x.T.dot(errors)
            self.w_[0] += self.learning_rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return self.net_input(x)
