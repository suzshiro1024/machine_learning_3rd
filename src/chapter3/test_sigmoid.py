from sigmoid import sigmoid
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    z = np.arange(-7, 7, 0.1)
    phi_z = sigmoid(z)

    plt.plot(z, phi_z)
    plt.axvline(0.0, color="k")
    plt.ylim(-0.1, 1.1)

    plt.xlabel("z")
    plt.ylabel("sigmoid(z)")

    plt.yticks([0.0, 0.5, 1.0])

    ax = plt.gca()
    ax.yaxis.grid(True)
    plt.tight_layout()

    plt.savefig("../../figure/sigmoid_sample.png")
