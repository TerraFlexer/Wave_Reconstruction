import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from splines import spline_coefficients
from splines import spline_approximation


def prepare_data(pnt_cnt, gamma=0.75):  # Инициализация всех данных
    h = 2 * np.pi / pnt_cnt

    I = []
    for s in range(-pnt_cnt // 2, pnt_cnt // 2):
        I.append(s)

    lambds = np.zeros(pnt_cnt)
    mus = np.zeros(pnt_cnt)
    for el in I:
        lambds[el + pnt_cnt // 2] = (4 / (h * h)) * np.sin(el * h / 2) * np.sin(el * h / 2)
        mus[el + pnt_cnt // 2] = 1 - (h * h / 6) * lambds[el + pnt_cnt // 2]

    gammas = np.ones((pnt_cnt, pnt_cnt)) * gamma
    ss = np.ones((pnt_cnt, pnt_cnt)) * 0.5

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


def method_count(f_kl, pnt_cnt, lambds, mus, gammas, ss, st_stb):  # Функция для вычислений самого метода
    # Осуществляем сдвиг для корректной работы БДПФ и выполняем БДПФ
    f_mn = fft2(fftshift(f_kl))

    # Вычисляем знаменатель дроби из метода
    znam = np.dot(lambds.reshape(pnt_cnt, -1), mus.reshape(1, -1)) + np.dot(mus.reshape(pnt_cnt, -1),
                                                                            lambds.reshape(1, -1))
    # Вычисляем слагаемое дробного стабилизатора
    drob_stab = gammas * np.power(np.dot(lambds.reshape(pnt_cnt, -1), lambds.reshape(1, -1)), ss)

    # Прибавляем стабилизатор, если был передан параметр st_stb
    znam += drob_stab * st_stb

    # Делаем заглушку от 0 в знаменателе
    znam[pnt_cnt // 2][pnt_cnt // 2] = 1

    # Получаем u_mn
    u_mn = f_mn / fftshift(znam)

    # Возвращаемся к условию нулевого среднего
    u_mn[0][0] = 0

    # Выполняем обратное БДПФ
    u_kl = ifftshift(ifft2(u_mn))

    return u_kl


def continue_even(z):
    cnt = np.shape(z)[0] * 2
    z4 = np.zeros((cnt, cnt))
    z4[:cnt // 2, :cnt // 2] = z
    z4[:cnt // 2, cnt // 2:] = np.flip(z, 1)
    z4[cnt // 2:, :cnt // 2] = np.flip(z, 0)
    z4[cnt // 2:, cnt // 2:] = np.flip(z)
    return z4


def continue_even_dx(z):
    cnt = np.shape(z)[0] * 2
    z4 = np.zeros((cnt, cnt))
    z4[:cnt // 2, :cnt // 2] = z
    z4[:cnt // 2, cnt // 2:] = np.flip(z, 1)
    z4[cnt // 2:, :cnt // 2] = -z
    z4[cnt // 2:, cnt // 2:] = -np.flip(z, 1)
    return z4


def continue_even_dy(z):
    cnt = np.shape(z)[0] * 2
    z4 = np.zeros((cnt, cnt))
    z4[:cnt // 2, :cnt // 2] = z
    z4[:cnt // 2, cnt // 2:] = -np.flip(z, 0)
    z4[cnt // 2:, :cnt // 2] = np.flip(z, 0)
    z4[cnt // 2:, cnt // 2:] = -z
    return z4


def method_v(z, pnt_cnt, edge, st_stb, gamma=0.75):  # Общая обертка метода

    # Вычисляем матрицы производных
    dx = fx(z, pnt_cnt, edge)
    dy = fy(z, pnt_cnt, edge)

    # Инициализируем сетку
    x = np.linspace(-edge, edge, pnt_cnt, endpoint=False)
    y = np.linspace(-edge, edge, pnt_cnt, endpoint=False)
    Y, X = np.meshgrid(x, y)

    # Вычисляем матрицы производных по направлению функции и раскладываем их по базису сплайнов
    matrix_g1 = spline_coefficients(dx, pnt_cnt, X, Y)
    matrix_g2 = spline_coefficients(dy, pnt_cnt, X, Y)

    # Вычисляем необходимые для работы метода матрицы
    lambds, mus, gammas, ss, B1, B2, G1, G2 = prepare_data(pnt_cnt, gamma)

    # Вычисляем правую часть уравнения
    f_kl = affect_rows(B2, np.dot(G1, matrix_g1)) + np.dot(B1, affect_rows(G2, matrix_g2))

    # Запускаем вычисление самого метода
    u_res = method_count(f_kl, pnt_cnt, lambds, mus, gammas, ss, st_stb)

    # Раскладываем Real часть полученной функции и раскладываем ее по базису сплайнов
    z_approx = spline_approximation(u_res.real, X, Y, pnt_cnt)

    return z_approx
