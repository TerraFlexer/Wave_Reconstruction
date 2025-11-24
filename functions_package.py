import numpy as np
from method import method_v
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from scipy.special import kl_div
import matplotlib.pyplot as plt
import torch

N = 64
Edge = np.pi
spiral_flag = 0


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


def multifocal_tilted(Rs, Zs, X, Y, alphas=None):
    n = len(Rs)
    N = X.shape[0]
    Z = np.zeros((N, N))

    if alphas is None:
        alphas = [(0.0, 0.0)] * n

    for k in range(n):
        R = Rs[k]
        zz = Zs[k]
        ax, ay = alphas[k]
        for i in range(N):
            for j in range(N):
                if R * R - X[i][j] ** 2 - Y[i][j] ** 2 > 0:
                    val = np.sqrt(R * R - X[i][j] ** 2 - Y[i][j] ** 2) + zz + ax * X[i][j] + ay * Y[i][j]
                    if Z[i][j] < val:
                        Z[i][j] = val
    return Z

def multifocal_with_anomalies(Rs, Zs, X, Y, dx=0, dy=0, dz=0, tilt_x=0, tilt_y=0, coma_x=0, coma_y=0, 
               cylinder_radius=0, cylinder_angle=0, cylinder_height=0):
    """
    Rs, Zs — радиусы и смещения сегментов
    X, Y — координатные сетки
    dx, dy, dz — смещение линзы
    tilt_x, tilt_y — наклоны (в радианах)
    coma_x, coma_y — коэффициенты комы по x и y
    cylinder_radius — радиус цилиндра
    cylinder_angle — угол поворота цилиндра (в радианах)
    cylinder_height — высота цилиндрической грани
    """
    EPS = 1e-2
    N = X.shape[0]
    Z = np.zeros((N, N))
    
    # Применяем смещение координат
    Xp = X - dx
    Yp = Y - dy
    
    # Вычисляем профиль многослойной линзы
    for R, zz in zip(Rs, Zs):
        mask = R**2 - Xp**2 - Yp**2 > 0
        Z_candidate = np.sqrt(np.clip(R**2 - Xp**2 - Yp**2, 0, None)) + zz + dz
        Z[mask] = np.maximum(Z[mask], Z_candidate[mask])
    
    # Создаем маску ненулевых точек (где есть волновой фронт)
    wavefront_mask = Z > EPS
    
    # Добавляем наклон ТОЛЬКО к волновому фронту
    Z_tilt = X * np.tan(tilt_x) + Y * np.tan(tilt_y)
    Z += Z_tilt
    
    # Добавляем цилиндрическую грань
    if cylinder_radius > 0 and cylinder_height > 0:
        # Поворачиваем координаты на угол цилиндра
        X_rot = X * np.cos(cylinder_angle) + Y * np.sin(cylinder_angle)
        Y_rot = -X * np.sin(cylinder_angle) + Y * np.cos(cylinder_angle)
        
        # Создаем цилиндрическую грань (боковая поверхность цилиндра)
        # Грань поднимается там, где X_rot приближается к cylinder_radius
        cylinder_mask = np.abs(X_rot) <= cylinder_radius
        cylinder_profile = cylinder_height * (1 - np.abs(X_rot) / cylinder_radius)
        
        # Добавляем цилиндрическую грань ко всему волновому фронту
        Z[wavefront_mask] += cylinder_profile[wavefront_mask] * cylinder_mask[wavefront_mask]
    
    # Добавляем кому (асимметричное искажение)
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    W_coma = (3 * rho**3 - 2 * rho) * (coma_x * np.cos(theta) + coma_y * np.sin(theta))
    
    # Применяем кому ТОЛЬКО к волновому фронту
    Z[wavefront_mask] += W_coma[wavefront_mask]
    
    return Z

def generate_random_multifocal_tilted(X, Y):
    n = np.random.randint(2, 6)
    hr = 3
    rs = np.zeros(n)
    zs = np.zeros(n)
    alphas = []

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
            zs[i] = np.random.uniform(rs[i - 1] + zs[i - 1] - rs[i], rz)

        ax = np.random.uniform(-0.2, 0.2)  # наклон по X
        ay = np.random.uniform(-0.2, 0.2)  # наклон по Y
        alphas.append((ax, ay))

    return multifocal_tilted(rs, zs, X, Y, alphas)


def generate_random_multifocal_anomaly(X, Y, offs=0, tilt=0, coma=0, cylinder=0):
    dx = np.random.uniform(-1.5, 1.5) * offs
    dy = np.random.uniform(-1.5, 1.5) * offs
    
    #TODO случайная генерация Rs и Zs
    Rs = [1, 3]
    Zs = [0.8, -1.5]
    
    return multifocal_with_anomalies(Rs, Zs, X, Y, dx, dy)


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


def offset(z1, z2, pnt, sprl_flg, torch_flag=0):
    if sprl_flg != 1:
        if torch_flag:
            v_offset = (torch.mean(z1[0, :]) + torch.mean(z1[pnt - 1, :]) +
                        torch.mean(z1[:, 0]) + torch.mean(z1[:, pnt - 1])) / 4
        else:
            v_offset = (np.average(z1[0, :]) + np.average(z1[pnt - 1, :]) +
                        np.average(z1[:, 0]) + np.average(z1[:, pnt - 1])) / 4
    else:
        if torch_flag:
            v_offset = torch.min(z1) - torch.min(z2)
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


def vortex(X, Y, x0, y0, m):
    m = 3  # топологический заряд

    R = np.sqrt((X - x0)**2 + (Y - y0)**2)
    Theta = np.arctan2(Y - y0, X - x0)
    amplitude = R**abs(m) * np.exp(-R**2)
    field = amplitude * np.exp(1j * m * Theta)
    
    return amplitude, np.angle(field)


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

        z = generate_random_multifocal(X, Y)

        for i in range(cnt_in_noise):
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
        plt.ylabel("MSE" + name)
        plt.xlabel("Шум(доля от максимального значения)")
        plt.legend()
        plt.show()

        plt.plot(perc_arr, ssim_arr, label='SSIM')
        plt.ylabel("SSIM" + name)
        plt.xlabel("Шум(доля от максимального значения)")
        plt.legend()
        plt.show()

        plt.plot(perc_arr, kl_div_arr, label='Kl_div')
        plt.ylabel("Kl_div" + name)
        plt.xlabel("Шум(доля от максимального значения)")
        plt.legend()
        plt.show()

    return (kl_divergence + accuracy + ssim_accuracy) / (cnt_noise * cnt_in_noise)


def visualaize_param_matrix(param, name='param'):
    x = np.linspace(0, N - 1, N, endpoint=False)
    y = np.linspace(0, N - 1, N, endpoint=False)
    Y, X = np.meshgrid(x, y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf1 = ax.plot_surface(X, Y, param, cmap='plasma')
    ax.set_title(name)
    ax.view_init(45, 60)
    plt.show()

    fig1, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf2 = ax.plot_surface(X, Y, param, cmap='plasma')
    ax.set_title(name)
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


def save_param_value_in_file(param_name, param_value, file_name):
    file_name += "_" + param_name + ".npy"
    with open(file_name, 'wb') as f:
        np.save(f, param_value)
        print("Saved " + param_name + " in " + file_name)


def load_param_value_from_file(file_name):
    with open(file_name + ".npy", 'rb') as f:
        param_value = np.load(f)
        print("Loaded params from " + file_name + ".npy")
        return param_value


def principal_value(x):
    # Приводим значение x в интервал [-pi, pi]
    while x > np.pi:
        x -= 2 * np.pi
    while x <= -np.pi:
        x += 2 * np.pi
    return x


def count_branch_sum(dx, dy, N, h, d1 = 20, d2 = 20, d3 = 20, d4 = 20):

    sum = 0

    # Движение вправо
    for j in range(d1, N - d3):
        sum += principal_value(dx[d2][j]) * h

    # Движение вниз
    for i in range(d2, N - d4):
        sum += principal_value(dy[i][N - d3 - 1]) * h

    # Движение влево
    for j in range(N - d3 - 1, d1, -1):
        sum += -principal_value(dx[N - d4 - 1][j - 1]) * h

    # Движение вверх
    for i in range(N - d4 - 1, d2, -1):
        sum += -principal_value(dy[i - 1][d1]) * h
    return sum