import numpy as np


def error(p):
    return 1 - np.max([p, 1 - p])
