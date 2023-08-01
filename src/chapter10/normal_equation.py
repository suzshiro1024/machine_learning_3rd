import numpy as np


def normal_equation(x, y):
    """
    resolve `w = (x.Tx)^-1 x.Ty`
    """

    xb = np.hstack((np.ones((x.shape[0], 1)), x))
    w = np.zeros(x.shape[1])
    z = np.linalg.inv(np.dot(xb.T, xb))
    w = np.dot(z, np.dot(xb.T, y))
    print(f"Slope: {w[1]:.3f}")
    print(f"Intercept: {w[0]:.3f}")

    return w


if __name__ == "__main__":
    x_test = np.random.rand(3, 2)
    y_test = np.random.rand(3)

    w = normal_equation(x_test, y_test)
    print(w)
