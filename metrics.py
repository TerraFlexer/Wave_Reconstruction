import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import functions_package as fpckg
from method import method_v, fill_gammas
from splines import spline_approximation
from scipy.special import kl_div

N = fpckg.N


def count_metrics(gammas):
    arr_perc = np.zeros(40)
    arr_mse = np.zeros(40)
    arr_ssim = np.zeros(40)
    arr_mse_stb_5 = np.zeros(40)
    arr_ssim_stb_5 = np.zeros(40)
    arr_mse_stb_75 = np.zeros(40)
    arr_ssim_stb_75 = np.zeros(40)
    arr_mse_stb_g = np.zeros(40)
    arr_ssim_stb_g = np.zeros(40)

    gamma_center = 0.3503128025624347
    gamma_middle1 = 0.4214517768191933
    gamma_middle2 = 0.6134753952099001
    gamma_side = 0.42702833893284037

    # gammas = fill_gammas([gamma_center, gamma_middle1, gamma_middle2], [8, 15, 25], gamma_side, N)

    for ind, el in enumerate(np.linspace(0.0, 0.2, num=40)):
        mse_avg = 0
        mse_avg_stb_5 = 0
        mse_avg_stb_75 = 0
        ssim_avg = 0
        ssim_avg_stb_5 = 0
        ssim_avg_stb_75 = 0
        mse_avg_stb_g = 0
        ssim_avg_stb_g = 0
        # Инициализируем сетку
        x = np.linspace(-fpckg.Edge, fpckg.Edge, N, endpoint=False)
        y = np.linspace(-fpckg.Edge, fpckg.Edge, N, endpoint=False)
        Y, X = np.meshgrid(x, y)

        # Считаем исходную мультифокальную/спиральную функцию
        if fpckg.spiral_flag:
            z_src = fpckg.spiral(3, 1, X, Y)
        else:
            z_src = fpckg.multifocal([1, 3], [0.8, -1.5])

        for i in range(5):
            # Добавляем шум к исходному фронту
            z = fpckg.add_noise(z_src, el, N)

            u_res = method_v(z, N, np.pi, 0)
            u_res_stb_5 = method_v(z, N, np.pi, 1, 0.5)
            u_res_stb_75 = method_v(z, N, np.pi, 1, 0.75)
            u_res_stb_g = method_v(z, N, np.pi, 1, gammas)

            z_approx = spline_approximation(u_res.real, X, Y, N)
            z_approx_stb_5 = spline_approximation(u_res_stb_5.real, X, Y, N)
            z_approx_stb_75 = spline_approximation(u_res_stb_75.real, X, Y, N)
            z_approx_stb_g = spline_approximation(u_res_stb_g.real, X, Y, N)

            offs = fpckg.offset(z_approx, z_src, N, fpckg.spiral_flag)
            offs_stb_5 = fpckg.offset(z_approx_stb_5, z_src, N, fpckg.spiral_flag)
            offs_stb_75 = fpckg.offset(z_approx_stb_75, z_src, N, fpckg.spiral_flag)
            offs_stb_g = fpckg.offset(z_approx_stb_g, z_src, N, fpckg.spiral_flag)

            mse_avg += mse(z_src, z_approx - offs)
            mse_avg_stb_5 += mse(z_src, z_approx_stb_5 - offs_stb_5)
            mse_avg_stb_75 += mse(z_src, z_approx_stb_75 - offs_stb_75)
            mse_avg_stb_g += mse(z_src, z_approx_stb_g - offs_stb_g)

            ssim_avg += (1 - ssim(z_src, z_approx - offs,
                                  data_range=max(np.max(z_src), np.max(z_approx - offs)) -
                                             min(np.min(z_src), np.min(z_approx - offs)))) / 2
            ssim_avg_stb_5 += (1 - ssim(z_src, z_approx_stb_5 - offs_stb_5,
                                        data_range=max(np.max(z_src), np.max(z_approx_stb_5 - offs_stb_5)) -
                                                   min(np.min(z_src), np.min(z_approx_stb_5 - offs_stb_5)))) / 2
            ssim_avg_stb_75 += (1 - ssim(z_src, z_approx_stb_75 - offs_stb_75,
                                         data_range=max(np.max(z_src), np.max(z_approx_stb_75 - offs_stb_75)) -
                                                    min(np.min(z_src), np.min(z_approx_stb_75 - offs_stb_75)))) / 2
            ssim_avg_stb_g += (1 - ssim(z_src, z_approx_stb_g - offs_stb_g,
                                        data_range=max(np.max(z_src), np.max(z_approx_stb_g - offs_stb_g)) -
                                                   min(np.min(z_src), np.min(z_approx_stb_g - offs_stb_g)))) / 2

        arr_perc[ind] = el

        arr_mse[ind] = mse_avg / 5
        arr_mse_stb_5[ind] = mse_avg_stb_5 / 5
        arr_mse_stb_75[ind] = mse_avg_stb_75 / 5
        arr_mse_stb_g[ind] = mse_avg_stb_g / 5

        arr_ssim[ind] = ssim_avg / 5
        arr_ssim_stb_5[ind] = ssim_avg_stb_5 / 5
        arr_ssim_stb_75[ind] = ssim_avg_stb_75 / 5
        arr_ssim_stb_g[ind] = ssim_avg_stb_g / 5

        print("Epoch: ", ind, " mse_avg = ", mse_avg / 5, " mse_avg_stb_5 = ", mse_avg_stb_5 / 5)
        print("ssim_avg = ", ssim_avg / 5, " ssim_avg_stb_5 = ", ssim_avg_stb_5 / 5, "\n")

    plt.plot(arr_perc, arr_mse, label='Без стабилизатора')
    plt.plot(arr_perc, arr_mse_stb_5, label='Со стабилизатором 0.5')
    plt.plot(arr_perc, arr_mse_stb_75, label='Со стабилизатором 0.75')
    plt.plot(arr_perc, arr_mse_stb_g, label='Со стабилизатором gamma')
    plt.ylabel("MSE")
    plt.xlabel("Шум(доля от максимального значения)")
    plt.legend()
    plt.show()

    plt.plot(arr_perc, arr_ssim, label='Без стабилизатора')
    plt.plot(arr_perc, arr_ssim_stb_5, label='Со стабилизатором 0.5')
    plt.plot(arr_perc, arr_ssim_stb_75, label='Со стабилизатором 0.75')
    plt.plot(arr_perc, arr_ssim_stb_g, label='Со стабилизатором gamma')
    plt.ylabel("DSSIM")
    plt.xlabel("Шум(доля от максимального значения)")
    plt.legend()
    plt.show()
