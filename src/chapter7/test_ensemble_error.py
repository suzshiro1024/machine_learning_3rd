import sys
import os

sys.path.append(os.pardir)

from ensemble_error import ensemble_error
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    error_range = np.arange(0.0, 1.01, 0.01)

    ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]

    plt.plot(error_range, ens_errors, label="Ensemble Error", linewidth=2)
    plt.plot(error_range, error_range, linestyle="--", label="Base Error", linewidth=2)
    plt.xlabel("Base Error")
    plt.ylabel("Base/Ensemble Error")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.5)
    plt.savefig("../../figure/test_ensemble_error.png")
