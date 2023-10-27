import numpy as np


def beta_0(x, y, k=5):
    n = np.shape(x)[0]
    z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            z[i][j] = 1 / (1 + np.exp(k * (max(abs(x[i][j]), abs(y[i][j])) - 2)))
    return z


def beta_1(x, y, k=5):
    return 1 - beta_0(x, y, k)
