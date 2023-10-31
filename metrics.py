import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from main import add_noise, spiral, continue_even, Edge, N
from method import method_v
from splines import spline_approximation, spline_coefficients
import finit_module as fmd

arr_perc = np.zeros(40)
arr_mse = np.zeros(40)
arr_ssim = np.zeros(40)
arr_mse_stb = np.zeros(40)
arr_ssim_stb = np.zeros(40)

for ind, el in enumerate(np.linspace(0.01, 0.2, num=40)):
    mse_avg = 0
    mse_avg_stb = 0
    ssim_avg = 0
    ssim_avg_stb = 0
    for i in range(5):
        # Инициализируем сетку
        x = np.linspace(-Edge, Edge, N, endpoint=False)
        y = np.linspace(-Edge, Edge, N, endpoint=False)
        Y, X = np.meshgrid(x, y)

        # Считаем исходную спиральную функцию
        z_src = spiral(3, 1, X, Y)

        # Добавляем шум к исходному фронту
        z = add_noise(z_src, el, N)

        # Инициализируем расширение для одного из кусков
        fm = z * fmd.beta_1(X, Y)
        fm = continue_even(fm)
        x = np.linspace(-Edge, Edge, 2 * N, endpoint=False)
        y = np.linspace(-Edge, Edge, 2 * N, endpoint=False)
        Y2, X2 = np.meshgrid(x, y)
        fm = np.roll(fm, N // 2, (0, 1)) * fmd.beta_0(X2, Y2)

        # Вычисляем прогоны метода для каждого из двух кусков
        z_approx = method_v(fm, 2 * N, np.pi, 0)
        z_approx1 = method_v(z * fmd.beta_0(X, Y), N, np.pi, 0)

        # Обрезаем расширенный кусок
        z_approx = np.roll(z_approx, -N // 2, (0, 1))[:N, :N]

        # Вычисляем прогоны метода для каждого из двух кусков со стабилизатором
        z_approx_stb = method_v(fm, 2 * N, np.pi, 1)
        z_approx1_stb = method_v(z * fmd.beta_0(X, Y), N, np.pi, 1)

        # Обрезаем расширенный кусок
        z_approx_stb = np.roll(z_approx_stb, -N // 2, (0, 1))[:N, :N]

        u_res = z_approx + z_approx1
        u_res_stb = z_approx_stb + z_approx1_stb

        z_approx = spline_approximation(u_res.real, X, Y, N)
        z_approx_stb = spline_approximation(u_res_stb.real, X, Y, N)

        # offs = (np.average(z_approx[0, :]) + np.average(z_approx[N - 1, :]) +
        # np.average(z_approx[:, 0]) + np.average(z_approx[:, N - 1])) / 4

        # offs_stb = (np.average(z_approx_stb[0, :]) + np.average(z_approx_stb[N - 1, :]) +
        # np.average(z_approx_stb[:, 0]) + np.average(z_approx_stb[:, N - 1])) / 4

        offs = np.min(z_approx) - np.min(z_src)
        offs_stb = np.min(z_approx_stb) - np.min(z_src)

        mse_avg += mse(z_src, z_approx - offs)
        mse_avg_stb += mse(z_src, z_approx_stb - offs_stb)

        ssim_avg += (1 - ssim(z_src, z_approx - offs,
                              data_range=max(np.max(z_src), np.max(z_approx - offs)) -
                                         min(np.min(z_src), np.min(z_approx - offs)))) / 2
        ssim_avg_stb += (1 - ssim(z_src, z_approx_stb - offs_stb,
                                  data_range=max(np.max(z_src), np.max(z_approx_stb - offs_stb)) -
                                             min(np.min(z_src), np.min(z_approx_stb - offs_stb)))) / 2

    arr_perc[ind] = el

    arr_mse[ind] = mse_avg / 5
    arr_mse_stb[ind] = mse_avg_stb / 5

    arr_ssim[ind] = ssim_avg / 5
    arr_ssim_stb[ind] = ssim_avg_stb / 5
    print("Epoch: ", ind, " mse_avg = ", mse_avg / 5, " mse_avg_stb = ", mse_avg_stb / 5, "\n")
    print(" ssim_avg = ", ssim_avg / 5, " ssim_avg_stb = ", ssim_avg_stb / 5, "\n\n")

plt.plot(arr_perc, arr_mse, label='Без стабилизатора')
plt.plot(arr_perc, arr_mse_stb, label='Со стабилизатором')
plt.ylabel("MSE")
plt.xlabel("Шум(доля от максимального значения)")
plt.legend()
plt.show()

plt.plot(arr_perc, arr_ssim, label='Без стабилизатора')
plt.plot(arr_perc, arr_ssim_stb, label='Со стабилизатором')
plt.ylabel("DSSIM")
plt.xlabel("Шум(доля от максимального значения)")
plt.legend()
plt.show()
