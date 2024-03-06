import numpy as np
from method import method_v
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from scipy.special import kl_div
import matplotlib.pyplot as plt


N = 64
Edge = np.pi
spiral_flag = 1


def multifocal(Rs, Zs, X, Y):
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


def multifocal_razr(Rs, Zs, R2s, X, Y):
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


def add_noise(z, perc, N):
    ampl = abs(z.max() - z.min()) * perc
    return z + ampl * np.random.randn(N, N)


def offset(z1, z2, pnt, sprl_flg):
    if sprl_flg != 1:
        v_offset = (np.average(z1[0, :]) + np.average(z1[pnt - 1, :]) +
                    np.average(z1[:, 0]) + np.average(z1[:, pnt - 1])) / 4
    else:
        v_offset = np.min(z1) - np.min(z2)

    return v_offset


def generate_random_multifocal(X, Y):
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

    return multifocal(rs[::-1], zs[::-1], X, Y)


def generate_random_multifocal_razr(X, Y):
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

    return multifocal_razr(rs[::-1], zs[::-1], r2s[::-1], X, Y)


def perform_trial(gammas, cnt_noise=5, cnt_in_noise=5, visual_flag=0, name='No name'):
    accuracy = 0
    ssim_accuracy = 0
    kl_divergence = 0

    mse_arr = np.zeros(cnt_noise)
    ssim_arr = np.zeros(cnt_noise)
    kl_div_arr = np.zeros(cnt_noise)

    perc_arr = np.zeros(cnt_noise)

    # Инициализируем сетку
    x = np.linspace(-Edge, Edge, N, endpoint=False)
    y = np.linspace(-Edge, Edge, N, endpoint=False)
    Y, X = np.meshgrid(x, y)

    for ind, el in enumerate(np.linspace(0.0, 0.1, num=cnt_noise)):
        accuracy_tmp = 0
        ssim_accuracy_tmp = 0
        kl_divergence_tmp = 0
        for i in range(cnt_in_noise):
            z = generate_random_multifocal(X, Y)

            z1 = add_noise(z, el, N)

            z_approx = method_v(z1, N, Edge, 1, gammas)

            offs = offset(z_approx, z, N, 0)

            accuracy_tmp += mse(z, z_approx - offs)

            ssim_accuracy_tmp += (1 - ssim(z, z_approx - offs,
                                       data_range=max(np.max(z), np.max(z_approx - offs)) -
                                                  min(np.min(z), np.min(z_approx - offs)))) / 2

            kl_divergence_tmp += np.sum(kl_div(z + 5, z_approx - offs + 5))

        mse_arr[ind] = accuracy_tmp / cnt_in_noise
        ssim_arr[ind] = ssim_accuracy_tmp / cnt_in_noise
        kl_div_arr[ind] = kl_divergence_tmp / cnt_in_noise

        perc_arr[ind] = el

        accuracy += accuracy_tmp
        ssim_accuracy += ssim_accuracy_tmp
        kl_divergence += kl_divergence_tmp

    if visual_flag:
        plt.plot(perc_arr, mse_arr, label='MSE')
        plt.plot(perc_arr, ssim_arr, label='SSIM')
        plt.plot(perc_arr, kl_div_arr, label='Kl_div')
        plt.ylabel("Metrics" + name)
        plt.xlabel("Шум(доля от максимального значения)")
        plt.legend()
        plt.show()

    return (kl_divergence + accuracy + ssim_accuracy) / (cnt_noise * cnt_in_noise)


def visualaize_param_matrix(param):
    x = np.linspace(0, N - 1, N, endpoint=False)
    y = np.linspace(0, N - 1, N, endpoint=False)
    Y, X = np.meshgrid(x, y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf1 = ax.plot_surface(X, Y, param, cmap='plasma')
    ax.set_title('gammas')
    ax.view_init(45, 60)
    plt.show()

    fig1, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf2 = ax.plot_surface(X, Y, param, cmap='plasma')
    ax.set_title('gammas')
    ax.view_init(90, 0)
    plt.show()


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