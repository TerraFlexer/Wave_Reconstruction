import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from method import dev_ss, dev_gamma, method_v
import functions_package as fpckg


def dev_mse_gammas(gammas, ss, u_orig, u_res, pnt_cnt, edge):
    mse_dev = 2 * (u_orig - u_res)
    return mse_dev * dev_gamma(gammas, ss, u_orig, pnt_cnt, edge)


def dev_mse_ss(gammas, ss, u_orig, u_res, pnt_cnt, edge):
    mse_dev = 2 * (u_orig - u_res)
    return mse_dev * dev_ss(gammas, ss, u_orig, pnt_cnt, edge)


def sopr_grad(grad_f, x0, eps=1e-8, max_iter=1000):
    # Initialzaing start values
    x = x0.copy()
    g = grad_f(x)
    d = -g
    delta_new = np.dot(g, g)

    for i in range(max_iter):
        # step_size = delta_new / np.dot(d, grad_f(x))
        step_size = 0.001

        x = x + step_size * d
        g = grad_f(x)
        delta_old = delta_new
        delta_new = np.dot(g, g)

        if np.sqrt(delta_new) < eps:
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