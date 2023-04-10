import numpy as np


def gini(p):
    return (p) * (1 - (p)) + (1 - p) * (1 - (1 - p))
