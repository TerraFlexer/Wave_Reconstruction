import numpy as np
import matplotlib.pyplot as plt
from main import z, X, Y, N, B1, B2, G1, G2, z4
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from main import add_noise, method_count, affect_rows, fx, fy
from splines import spline_approximation, spline_coefficients

arr_perc = np.zeros(50)
arr_mse = np.zeros(50)
arr_ssim = np.zeros(50)
arr_mse_stb = np.zeros(50)
arr_ssim_stb = np.zeros(50)

for ind, el in enumerate(np.linspace(0.01, 0.2, num=50)):
    mse_avg = 0
    mse_avg_stb = 0
    ssim_avg = 0
    ssim_avg_stb = 0
    for i in range(5):
        fmi = add_noise(z4, el)

        g1 = spline_coefficients(fx(fmi), N, X, Y)
        g2 = spline_coefficients(fy(fmi), N, X, Y)

        f_kl = affect_rows(B2, np.dot(G1, g1)) + np.dot(B1, affect_rows(G2, g2))

        u_res = method_count(f_kl, 0)
        u_res_stb = method_count(f_kl, 1)

        z_approx = spline_approximation(u_res.real, X, Y)
        z_approx_stb = spline_approximation(u_res_stb.real, X, Y)

        offs = (np.average(z_approx[0, :]) + np.average(z_approx[N - 1, :]) +
                np.average(z_approx[:, 0]) + np.average(z_approx[:, N - 1])) / 4

        offs_stb = (np.average(z_approx_stb[0, :]) + np.average(z_approx_stb[N - 1, :]) +
                    np.average(z_approx_stb[:, 0]) + np.average(z_approx_stb[:, N - 1])) / 4

        offs = np.max(z_approx) - np.max(z4)
        offs_stb = np.max(z_approx_stb) - np.max(z4)

        mse_avg += mse(z4, z_approx - offs)
        mse_avg_stb += mse(z4, z_approx_stb - offs_stb)

        ssim_avg += (1 - ssim(z4, z_approx - offs,
                              data_range=max(np.max(z4), np.max(z_approx - offs)) -
                                         min(np.min(z4), np.min(z_approx - offs)))) / 2
        ssim_avg_stb += (1 - ssim(z4, z_approx_stb - offs_stb,
                                  data_range=max(np.max(z4), np.max(z_approx_stb - offs_stb)) -
                                             min(np.min(z4), np.min(z_approx_stb - offs_stb)))) / 2

    arr_perc[ind] = el

    arr_mse[ind] = mse_avg / 5
    arr_mse_stb[ind] = mse_avg_stb / 5

    arr_ssim[ind] = ssim_avg / 5
    arr_ssim_stb[ind] = ssim_avg_stb / 5

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
