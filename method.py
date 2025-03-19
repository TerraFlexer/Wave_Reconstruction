import numpy as np
import torch
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from splines import spline_coefficients


def init_net(pnt_cnt, edge):
    x = np.linspace(-edge, edge, pnt_cnt, endpoint=False)
    y = np.linspace(-edge, edge, pnt_cnt, endpoint=False)
    Y, X = np.meshgrid(x, y)
    return Y, X


def prepare_data(pnt_cnt, gamma=0.5, s1=0.5, torch_flag=0):  # Инициализация всех данных
    h = 2 * np.pi / pnt_cnt

    I = []
    for s in range(-pnt_cnt // 2, pnt_cnt // 2):
        I.append(s)

    lambds = np.zeros(pnt_cnt)
    mus = np.zeros(pnt_cnt)
    for el in I:
        lambds[el + pnt_cnt // 2] = (4 / (h * h)) * np.sin(el * h / 2) * np.sin(el * h / 2)
        mus[el + pnt_cnt // 2] = 1 - (h * h / 6) * lambds[el + pnt_cnt // 2]

    if torch_flag:
        gammas = torch.ones((pnt_cnt, pnt_cnt)) * gamma
        ss = torch.ones((pnt_cnt, pnt_cnt)) * s1
    else:
        gammas = np.ones((pnt_cnt, pnt_cnt)) * gamma
        ss = np.ones((pnt_cnt, pnt_cnt)) * s1

    Lambd1 = np.zeros((pnt_cnt, pnt_cnt))
    Lambd2 = np.zeros((pnt_cnt, pnt_cnt))
    G1 = np.zeros((pnt_cnt, pnt_cnt))
    G2 = np.zeros((pnt_cnt, pnt_cnt))
    for i in range(pnt_cnt):
        for j in range(pnt_cnt):
            if i == j:
                Lambd1[i][j] = 2
                Lambd2[i][j] = 2
                if j > 0:
                    Lambd1[i][j - 1] = -1
                    Lambd2[i][j - 1] = -1
                    G1[i][j - 1] = 1
                    G2[i][j - 1] = 1
                if j < pnt_cnt - 1:
                    Lambd1[i][j + 1] = -1
                    Lambd2[i][j + 1] = -1
                    G1[i][j + 1] = -1
                    G2[i][j + 1] = -1
    Lambd1[0][pnt_cnt - 1] = -1
    Lambd2[0][pnt_cnt - 1] = -1
    Lambd1[pnt_cnt - 1][0] = -1
    Lambd2[pnt_cnt - 1][0] = -1
    G1[0][pnt_cnt - 1] = 1
    G2[0][pnt_cnt - 1] = 1
    G1[pnt_cnt - 1][0] = -1
    G2[pnt_cnt - 1][0] = -1
    B1 = np.eye(pnt_cnt) - (Lambd1 / 6)
    B2 = np.eye(pnt_cnt) - (Lambd2 / 6)
    Lambd1 /= (h * h)
    Lambd2 /= (h * h)
    G1 /= (2 * h)
    G2 /= (2 * h)
    return lambds, mus, gammas, ss, B1, B2, G1, G2


def fill_gammas(gamma_values, rad_values, gamma_side, pnt_cnt):
    gammas = np.ones((pnt_cnt, pnt_cnt)) * gamma_side

    n = len(gamma_values)
    for k in range(n):
        for i in range(pnt_cnt):
            for j in range(pnt_cnt):
                r2 = rad_values[k]
                if k == 0:
                    r1 = 0
                else:
                    r1 = rad_values[k - 1]
                if r1 ** 2 <= (pnt_cnt // 2 - 1 - i) ** 2 + (pnt_cnt // 2 - 1 - j) ** 2 <= r2 ** 2:
                    gammas[i][j] = gamma_values[k]

    return gammas


def affect_rows(A, V):  # Применение оператора построчно
    N = np.shape(A)[0]
    Vr = np.zeros((N, N))
    for ind, el in enumerate(V):
        it = el.reshape(N, -1)
        it = np.dot(A, it)
        Vr[ind] = np.squeeze(it)
    return Vr


def shiftrow(arr, shift):  # Сдвиг строк
    arr = np.roll(arr, axis=0, shift=shift)
    # arr[shift, :] = arr[shift - 1, :]
    return arr


def shiftcolumn(arr, shift):  # Сдвиг столбцов
    arr = np.roll(arr, axis=1, shift=shift)
    # arr[:, shift] = arr[:, shift - 1]
    return arr


def fx(func, pnt_cnt, edge):  # Частная производная по x
    d = 2 * edge / pnt_cnt
    return (shiftrow(func, pnt_cnt - 1) - shiftrow(func, 1)) / (2 * d)


def fy(func, pnt_cnt, edge):  # Частная производная по y
    d = 2 * edge / pnt_cnt
    return (shiftcolumn(func, pnt_cnt - 1) - shiftcolumn(func, 1)) / (2 * d)


def method_count(f_kl, pnt_cnt, lambds, mus, gammas, ss, st_stb, torch_flag=0):  # Функция для вычислений самого метода
    # Осуществляем сдвиг для корректной работы БДПФ и выполняем БДПФ
    if torch_flag:
        f_kl = torch.from_numpy(f_kl)
        f_mn = torch.fft.fft2(torch.fft.fftshift(f_kl))
    else:
        f_mn = fft2(fftshift(f_kl))

    # Вычисляем знаменатель дроби из метода
    znam = np.dot(lambds.reshape(pnt_cnt, -1), mus.reshape(1, -1)) + np.dot(mus.reshape(pnt_cnt, -1),
                                                                            lambds.reshape(1, -1))

    # Вычисляем слагаемое дробного стабилизатора
    if torch_flag:
        drob_stab = gammas * torch.pow(torch.from_numpy(np.dot(lambds.reshape(pnt_cnt, -1), lambds.reshape(1, -1))), ss)
        znam = torch.from_numpy(znam)
    else:
        drob_stab = gammas * np.power(np.dot(lambds.reshape(pnt_cnt, -1), lambds.reshape(1, -1)), ss)

    # Прибавляем стабилизатор, если был передан параметр st_stb
    znam += drob_stab * st_stb

    # Делаем заглушку от 0 в знаменателе
    znam[pnt_cnt // 2][pnt_cnt // 2] = 1

    # Получаем u_mn
    if torch_flag:
        u_mn = f_mn / torch.fft.fftshift(znam)
    else:
        u_mn = f_mn / fftshift(znam)

    # Возвращаемся к условию нулевого среднего
    u_mn[0][0] = 0

    # Выполняем обратное БДПФ
    if torch_flag:
        u_kl = torch.fft.ifftshift(torch.fft.ifft2(u_mn))
    else:
        u_kl = ifftshift(ifft2(u_mn))

    return u_kl


def method_v(z, pnt_cnt, edge, st_stb, gamma=0.5, s=0.5, torch_flag=0):  # Общая обертка метода
    # Инициализируем сетку
    Y, X = init_net(pnt_cnt, edge)

    # Вычисляем матрицы производных
    dx = fx(z, pnt_cnt, edge)
    dy = fy(z, pnt_cnt, edge)

    # Вычисляем матрицы производных по направлению функции и раскладываем их по базису сплайнов
    matrix_g1 = spline_coefficients(dx, pnt_cnt, X, Y)
    matrix_g2 = spline_coefficients(dy, pnt_cnt, X, Y)

    # Вычисляем необходимые для работы метода матрицы
    lambds, mus, gammas, ss, B1, B2, G1, G2 = prepare_data(pnt_cnt, gamma, s, torch_flag)

    # Вычисляем правую часть уравнения
    f_kl = affect_rows(B2, np.dot(G1, matrix_g1)) + np.dot(B1, affect_rows(G2, matrix_g2))

    # Запускаем вычисление самого метода
    u_res = method_count(f_kl, pnt_cnt, lambds, mus, gammas, ss, st_stb, torch_flag)

    # Раскладываем Real часть полученной функции и раскладываем ее по базису сплайнов
    # z_approx = spline_approximation(u_res.real, X, Y, pnt_cnt)

    return u_res.real


def method_v_slopes(dx, dy, pnt_cnt, edge, st_stb, gamma=0.75, s=0.5, torch_flag=0):  # Общая обертка метода при вызове для наклонов
    # Инициализируем сетку
    Y, X = init_net(pnt_cnt, edge)

    # Вычисляем матрицы производных по направлению функции и раскладываем их по базису сплайнов
    matrix_g1 = spline_coefficients(dx, pnt_cnt, X, Y)
    matrix_g2 = spline_coefficients(dy, pnt_cnt, X, Y)

    # Вычисляем необходимые для работы метода матрицы
    lambds, mus, gammas, ss, B1, B2, G1, G2 = prepare_data(pnt_cnt, gamma, s, torch_flag)

    # Вычисляем правую часть уравнения
    f_kl = affect_rows(B2, np.dot(G1, matrix_g1)) + np.dot(B1, affect_rows(G2, matrix_g2))

    # Запускаем вычисление самого метода
    u_res = method_count(f_kl, pnt_cnt, lambds, mus, gammas, ss, st_stb, torch_flag)

    # Раскладываем Real часть полученной функции и раскладываем ее по базису сплайнов
    # z_approx = spline_approximation(u_res.real, X, Y, pnt_cnt)

    return u_res.real