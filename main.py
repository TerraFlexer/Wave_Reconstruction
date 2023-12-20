import numpy as np
import matplotlib.pyplot as plt
import finit_module as fmd
from method import method_v

N = 64
Edge = np.pi


def add_noise(z, perc, N):
    ampl = abs(z.max() - z.min()) * perc
    return z + ampl * np.random.randn(N, N)


x = np.linspace(-Edge, Edge, N, endpoint=False)
y = np.linspace(-Edge, Edge, N, endpoint=False)
Y, X = np.meshgrid(x, y)


def multifocal(Rs, Zs):
    n = len(Rs)
    Z = np.zeros((N, N))

    for k in range(n):
        R = Rs[k]
        zz = Zs[k]
        for i in range(N):
            for j in range(N):
                if R * R - X[i][j] ** 2 - Y[i][j] ** 2 > 0:
                    if Z[i][j] < np.sqrt(R * R - X[i][j] ** 2 - Y[i][j] ** 2) + zz:
                        Z[i][j] = np.sqrt(R * R - X[i][j] ** 2 - Y[i][j] ** 2) + zz

    return Z


def multifocal_razr(Rs, Zs, R2s):
    n = len(Rs)
    Z = np.zeros((N, N))

    for k in range(n):
        R = Rs[k]
        zz = Zs[k]
        for i in range(N):
            for j in range(N):
                if R2s[k] ** 2 <= X[i][j] ** 2 + Y[i][j] ** 2 < R2s[k + 1] ** 2:
                    if np.sqrt(R * R - X[i][j] ** 2 - Y[i][j] ** 2) + zz >= 0:
                        Z[i][j] = np.sqrt(R * R - X[i][j] ** 2 - Y[i][j] ** 2) + zz

    return Z


def to_polar_teta(x, y):
    if x > 0 and y >= 0:
        return np.arctan(y / x)
    if x > 0 > y:
        return np.arctan(y / x) + 2 * np.pi
    if x < 0:
        return np.arctan(y / x) + np.pi
    if x == 0 and y > 0:
        return np.pi / 2
    if x == 0 and y < 0:
        return 3 * np.pi / 2
    if x == 0 and y == 0:
        return 0


def to_polar_r(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def spiral(f0, f1, x, y):
    N1 = x.shape[0]
    r = np.zeros((N1, N1))
    teta = np.zeros((N1, N1))
    for i in range(N1):
        for j in range(N1):
            r[i][j] = to_polar_r(x[i][j], y[i][j]) * 0.8
            teta[i][j] = to_polar_teta(x[i][j], y[i][j])

    ff = f1 * (teta - np.pi) / np.pi
    u = r ** 2 / (2 * f0 + 2 * ff)
    return u


def generate_random_multifocal():
    n = np.random.randint(2, 6)

    hr = 3

    rs = np.zeros(n)
    zs = np.zeros(n)

    for i in range(n):
        if i == 0:
            rs[i] = np.random.uniform(2, hr)
        else:
            rs[i] = np.random.uniform(0.7, hr)

        hr = rs[i]
        if i == 0:
            zs[i] = np.random.uniform(-rs[i], 0)
        else:
            rz = np.sqrt(rs[i - 1] ** 2 - rs[i] ** 2) + zs[i - 1]
            # print("z: ", zs[i], "left: ", rs[i - 1] + zs[i - 1] - rs[i], "right: ", rz)
            zs[i] = np.random.uniform(rs[i - 1] + zs[i - 1] - rs[i], rz)
            # print("z: ", zs[i], "left: ", rs[i - 1] + zs[i - 1] - rs[i], "right: ", rz)

    return multifocal(rs[::-1], zs[::-1])


def generate_random_multifocal_razr():
    n = np.random.randint(2, 6)

    hr = 3

    rs = np.zeros(n)
    zs = np.zeros(n)
    r2s = np.zeros(n + 1)

    for i in range(n):
        if i == 0:
            rs[i] = np.random.uniform(2, hr)
            r2s[i] = np.random.uniform(2, rs[i])
        else:
            rs[i] = np.random.uniform(0.7, hr)
            r2s[i] = np.random.uniform(0.7, rs[i])

        hr = r2s[i]

        if i == 0:
            zs[i] = np.random.uniform(-rs[i], 0)
        else:
            rz = np.sqrt(rs[i - 1] ** 2 - rs[i] ** 2) + zs[i - 1]
            # print("z: ", zs[i], "left: ", rs[i - 1] + zs[i - 1] - rs[i], "right: ", rz)
            zs[i] = np.random.uniform(rs[i - 1] + zs[i - 1] - rs[i], rz)
            # print("z: ", zs[i], "left: ", rs[i - 1] + zs[i - 1] - rs[i], "right: ", rz)

    r2s[n] = 0

    return multifocal_razr(rs[::-1], zs[::-1], r2s[::-1])


def offset(z1, z2, pnt, sprl_flg):
    if sprl_flg != 1:
        v_offset = (np.average(z1[0, :]) + np.average(z1[pnt - 1, :]) +
                    np.average(z1[:, 0]) + np.average(z1[:, pnt - 1])) / 4
    else:
        v_offset = np.min(z1) - np.min(z2)

    return v_offset


def make_zeros(z, pnt=4):
    sz = np.shape(z)[0]
    z[:pnt, :] = 0
    z[sz - pnt:, :] = 0
    z[:, :pnt] = 0
    z[:, sz - pnt:] = 0
    return z


def cut_zeros(z, pnt=4):
    sz = np.shape(z)[0]
    return z[pnt: sz - pnt, pnt: sz - pnt]


def gauss(x, y):
    z = np.sin(x)
    return z


# z = multifocal([1, 3, 12], [0.8, -1.5, -11])
# z = multifocal([1, 3], [0.8, -1.5])
# z = multifocal_razr([np.sqrt(3), 3], [0.8, -1.5], [0, 1, 3])
z = spiral(3, 1, X, Y)
# z = gauss(X, Y)

# z = generate_random_multifocal_razr()

# z = add_noise(z, 0.01, N)

z_approx = method_v(z, N, np.pi, 0, gamma=0.5)

spiral_flag = 1

offs = offset(z_approx, z, N, spiral_flag)


def prepare_for_visual(X1, Y1, Z1):
    if spiral_flag != 1:
        return Z1
    Z2 = Z1
    sz = np.shape(Z1)[0]
    for i in range(np.shape(X1)[0]):
        for j in range(np.shape(Y1)[0]):
            if X1[i][j] > 0 and j == sz // 2:
                Z2[i][j] = np.nan

    return Z2


fig1, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf2 = ax.plot_surface(X, Y, prepare_for_visual(X, Y, z), cmap='plasma')
ax.set_title('Spiral')
ax.view_init(45, 60)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
surf1 = ax.plot_surface(X, Y, prepare_for_visual(X, Y, z), cmap='plasma')
ax.set_title('Original function')
# ax.set_zlim([0, np.max(fm)])
ax.view_init(30, -120)
# fig.colorbar(surf1, location='bottom', shrink=0.6, aspect=7)

ax = fig.add_subplot(122, projection='3d')
surf2 = ax.plot_surface(X, Y, prepare_for_visual(X, Y, z_approx - offs), cmap='plasma')
ax.set_title('Method approximation')
# ax.set_zlim([0, np.max((z_approx - offs))])
ax.view_init(30, -120)
# fig.colorbar(surf2, location='bottom', shrink=0.6, aspect=7)
plt.show()
