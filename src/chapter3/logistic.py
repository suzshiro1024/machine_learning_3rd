from sigmoid import sigmoid
import numpy as np


# y=1のコストを計算する関数
def cost_1(z):
    return -np.log(sigmoid(z))


# y=0のコストを計算する関数
def cost_0(z):
    return -np.log(1 - sigmoid(z))
