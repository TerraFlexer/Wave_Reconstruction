import numpy as np
import matplotlib.pyplot as plt
import finit_module as fmd
from method import method_v

N = 32
Edge = np.pi


def add_noise(z, perc):
    ampl = (z.max() - z.min()) * perc
    return z + 2 * ampl * np.random.randn(N, N) - ampl / 2


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


def continue_even(z):
    cnt = np.shape(z)[0] * 2
    z4 = np.zeros((cnt, cnt))
    z4[:cnt // 2, :cnt // 2] = z
    z4[:cnt // 2, cnt // 2:] = np.flip(z, 1)
    z4[cnt // 2:, :cnt // 2] = np.flip(z, 0)
    z4[cnt // 2:, cnt // 2:] = np.flip(z)
    return z4


def gauss(x, y):
    z = np.sin(x)
    return z


# z = multifocal([1, 3, 12], [0.8, -1.5, -11])
z = multifocal([1, 3], [0.8, -1.5])
# z = multifocal_razr([np.sqrt(3), 3], [0.8, -1.5], [0, 1, 3])
z = spiral(3, 1, X, Y)
# z = gauss(X, Y)

# fm = add_noise(z, 0.02)
fm = z * fmd.beta_1(X, Y)
fm = continue_even(fm)
x = np.linspace(-Edge, Edge, 2 * N, endpoint=False)
y = np.linspace(-Edge, Edge, 2 * N, endpoint=False)
Y2, X2 = np.meshgrid(x, y)
fm = np.roll(fm, N // 2, (0, 1)) * fmd.beta_0(X2, Y2)

z_approx = method_v(fm, 2 * N, np.pi, 0)
z_approx1 = method_v(z * fmd.beta_0(X, Y), N, np.pi, 0)

spiral_flag = 1

offs = (np.average(z_approx[0, :]) + np.average(z_approx[N - 1, :]) +
        np.average(z_approx[:, 0]) + np.average(z_approx[:, N - 1])) / 4
offs *= (1 - spiral_flag)

x = np.linspace(-Edge, Edge, 2 * N, endpoint=False)
y = np.linspace(-Edge, Edge, 2 * N, endpoint=False)
Yv, Xv = np.meshgrid(x, y)
zv = np.roll(fm, -N // 2, (0, 1))[:N, :N]
z_a = np.roll(z_approx, -N // 2, (0, 1))[:N, :N]
z2v = (z_a - offs)

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
surf1 = ax.plot_surface(X, Y, z * fmd.beta_0(X, Y), cmap='plasma')
ax.set_title('Original function')
# ax.set_zlim([0, np.max(fm)])
ax.view_init(45, 60)
# fig.colorbar(surf1, location='bottom', shrink=0.6, aspect=7)

ax = fig.add_subplot(122, projection='3d')
surf2 = ax.plot_surface(X, Y, (z_approx1 - offs), cmap='plasma')
ax.set_title('Method approximation')
# ax.set_zlim([0, np.max((z_approx - offs))])
ax.view_init(45, 60)
# fig.colorbar(surf2, location='bottom', shrink=0.6, aspect=7)
plt.show()


def prepare_for_visual(X1, Y1, Z1):
    Z2 = Z1
    sz = np.shape(Z1)[0]
    for i in range(np.shape(X1)[0]):
        for j in range(np.shape(Y1)[0]):
            if X1[i][j] > 0 and j == sz // 2:
                Z2[i][j] = np.nan

    return Z2


fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
surf5 = ax.plot_surface(X2, Y2, fm, cmap='plasma')
ax.set_title('Original function')
# ax.set_zlim([0, np.max(fm)])
ax.view_init(45, 60)
# fig.colorbar(surf1, location='bottom', shrink=0.6, aspect=7)

ax = fig.add_subplot(122, projection='3d')
surf6 = ax.plot_surface(X2, Y2, z_approx, cmap='plasma')
ax.set_title('Method approximation')
# ax.set_zlim([0, np.max((z_approx - offs))])
ax.view_init(45, 60)
# fig.colorbar(surf2, location='bottom', shrink=0.6, aspect=7)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
surf7 = ax.plot_surface(X, Y, fmd.beta_0(X, Y), cmap='plasma')
ax.set_title('beta_0')
# ax.set_zlim([0, np.max(fm)])
ax.view_init(45, 60)
# fig.colorbar(surf1, location='bottom', shrink=0.6, aspect=7)

ax = fig.add_subplot(122, projection='3d')
surf8 = ax.plot_surface(X, Y, fmd.beta_1(X, Y), cmap='plasma')
ax.set_title('beta_1')
# ax.set_zlim([0, np.max((z_approx - offs))])
ax.view_init(45, 60)
# fig.colorbar(surf2, location='bottom', shrink=0.6, aspect=7)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
surf3 = ax.plot_surface(X, Y, prepare_for_visual(X, Y, zv), cmap='plasma')
ax.set_title('Original function')
# ax.set_zlim([0, np.max(fm)])
ax.view_init(45, 60)
# fig.colorbar(surf1, location='bottom', shrink=0.6, aspect=7)

ax = fig.add_subplot(122, projection='3d')
surf4 = ax.plot_surface(X, Y, prepare_for_visual(X, Y, z2v), cmap='plasma')
ax.set_title('Method approximation')
'''ax.set_title('Method approximation\nMSE:' + str(mse(z, z_approx - offs)) +
             '\nDSSIM:' + str((1 - ssim(z, z_approx - offs,
                                        data_range=max(np.max(fm), np.max(z_approx - offs)) -
                                                   min(np.min(fm), np.min(z_approx - offs)))) / 2))'''
# ax.set_zlim([0, np.max((z_approx - offs))])
ax.view_init(45, 60)
# fig.colorbar(surf2, location='bottom', shrink=0.6, aspect=7)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
surf5 = ax.plot_surface(X, Y, -prepare_for_visual(X, Y, zv + z * fmd.beta_0(X, Y)), cmap='plasma')
ax.set_title('Original function')
# ax.set_zlim([0, np.max(fm)])
ax.view_init(45, 60)
# fig.colorbar(surf1, location='bottom', shrink=0.6, aspect=7)

ax = fig.add_subplot(122, projection='3d')
surf6 = ax.plot_surface(X, Y, -prepare_for_visual(X, Y, z2v + z_approx1), cmap='plasma')
ax.set_title('Method approximation')
'''ax.set_title('Method approximation\nMSE:' + str(mse(z, z_approx - offs)) +
             '\nDSSIM:' + str((1 - ssim(z, z_approx - offs,
                                        data_range=max(np.max(fm), np.max(z_approx - offs)) -
                                                   min(np.min(fm), np.min(z_approx - offs)))) / 2))'''
# ax.set_zlim([0, np.max((z_approx - offs))])
ax.view_init(45, 60)
# fig.colorbar(surf2, location='bottom', shrink=0.6, aspect=7)
plt.show()
