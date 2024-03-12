import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from method import dev_ss, dev_gamma, method_v
import functions_package as fpckg
from metrics import count_metrics


def dev_mse_gammas(gammas, ss, u_orig, pnt_cnt, edge):
    u_res = method_v(u_orig, pnt_cnt, edge, 1, gammas, ss)
    mse_dev = 2 * (u_orig - u_res)
    dg = dev_gamma(gammas, ss, u_orig, pnt_cnt, edge)
    return mse_dev * dg


def dev_mse_ss(gammas, ss, u_orig, pnt_cnt, edge):
    u_res = method_v(u_orig, pnt_cnt, edge, 1, gammas, ss)
    mse_dev = 2 * (u_orig - u_res)
    return mse_dev * dev_ss(gammas, ss, u_orig, pnt_cnt, edge)


def grad_descent(grad_f, x0, scnd_param, u_orig, pnt_cnt, edge, eps=1e-8, max_iter=10):
    # Initialzaing start values
    x = x0.copy()
    # g = grad_f(x)
    # g = dev_mse_gammas(x, scnd_param, u_orig, pnt_cnt, edge)
    g = dev_mse_ss(scnd_param, x, u_orig, pnt_cnt, edge)
    # print(g)
    d = -g
    delta_new = np.dot(g, g)

    for i in range(max_iter):
        # step_size = delta_new / np.dot(d, grad_f(x))
        step_size = 0.001

        x = x + step_size * d
        # g = grad_f(x)
        # g = dev_mse_gammas(x, scnd_param, u_orig, pnt_cnt, edge)
        g = dev_mse_ss(scnd_param, x, u_orig, pnt_cnt, edge)
        delta_old = delta_new
        delta_new = np.dot(g, g)

        if np.sqrt(np.sum(delta_new)) < eps:
            break

        beta = delta_new / delta_old
        d = -g + beta * d

    return x


N = fpckg.N
Edge = fpckg.Edge


x = np.linspace(-Edge, Edge, N, endpoint=False)
y = np.linspace(-Edge, Edge, N, endpoint=False)
Y, X = np.meshgrid(x, y)

u_orig = fpckg.multifocal([1, 3], [0.8, -1.5], X, Y)

gamma0 = np.ones((N, N)) * 0.5

# fpckg.visualaize_param_matrix(gamma0)

res = grad_descent(np.sin, np.ones((N, N)) * 0.5, 0.5, u_orig, N, Edge)

fpckg.visualaize_param_matrix(res)

# count_metrics(res)