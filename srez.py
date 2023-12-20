import numpy as np
import matplotlib.pyplot as plt

EPS = 0.5


def get2d_srez(z):
    # return z[:, N // 2 - 1]
    return z.diagonal()


def get2d_circle_srez(x, y, z, r):
    srez = np.zeros(10 * N)
    ind = 0

    for i in range(np.shape(x)[0]):
        for j in range(np.shape(y)[0]):
            if abs(x[i][j] ** 2 + y[i][j] ** 2 - r ** 2) < EPS:
                srez[ind] = z[i][j]
                ind += 1
    return srez[:ind]


spiral_flag = 1

if spiral_flag:
    z_approx = z_approx[: N // 2, : N // 2]

'''za = get2d_srez(z_approx)
offs = (za[0] + za[N - 1]) / 2 * (spiral_flag - 1)

plt.plot(get2d_srez(X), get2d_srez(z4), label='Исходный фронт')
plt.plot(get2d_srez(X), get2d_srez(z_approx) - offs, label='Восстановленный фронт')
plt.plot(get2d_srez(X), get2d_srez(fm), linestyle='-.', label='Зашумленный фронт')
plt.legend()
plt.show()'''

zc = get2d_circle_srez(Xd2, Yd2, z_approx, 2)
os = np.arange(np.shape(zc)[0])
print(np.shape(os))
print(np.shape(zc))
print(np.size(zc))
plt.plot(os, zc, label='Исходный фронт')
plt.legend()
plt.show()
