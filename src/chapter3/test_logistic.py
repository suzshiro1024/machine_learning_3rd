from logistic import cost_0
from logistic import cost_1
from sigmoid import sigmoid
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    z = np.arange(-10, 10, 0.1)
    phi_z = sigmoid(z)

    c1 = [cost_1(x) for x in z]
    plt.plot(phi_z, c1, label="J(w) if y=1")

    c0 = [cost_0(x) for x in z]
    plt.plot(phi_z, c0, linestyle="--", label="J(w) if y=0")

    plt.ylim(0.0, 5.1)
    plt.xlim([0, 1])
    plt.xlabel("phi(z)")
    plt.ylabel("J(w)")
    plt.legend(loc="upper center")
    plt.tight_layout()

    plt.savefig("../../figure/test_logistic.png")
