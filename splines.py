import numpy as np


def basis(i, x, pnt_cnt):
    h = 2 * np.pi / pnt_cnt
    if i == -pnt_cnt // 2:
        return basis(-pnt_cnt // 2 + 1, x + h, pnt_cnt) + basis(pnt_cnt // 2 - 1, x - h, pnt_cnt)
    else:
        return np.maximum(0, 1 - np.abs(x - i * h) / h)


def spline_coefficients(f_matrix, pnt_cnt, x, y):
    coefficients = np.zeros((pnt_cnt, pnt_cnt))
    for k in range(-pnt_cnt // 2, pnt_cnt // 2):
        for l in range(-pnt_cnt // 2, pnt_cnt // 2):
            def integrand(x, y):
                return f_matrix * basis(k, x, pnt_cnt) * basis(l, y, pnt_cnt)

            coefficients[k + pnt_cnt // 2, l + pnt_cnt // 2] = np.sum(integrand(x, y)) / np.sum(basis(k, x, pnt_cnt) *
                                                                                                basis(l, y, pnt_cnt))
    return coefficients


def spline_approximation(c, x, y, N):
    f_approx = np.zeros((x.shape[0], y.shape[0]))
    for k in range(N):
        for l in range(N):
            f_approx += c[k, l] * basis(k - N // 2, x, N) * basis(l - N // 2, y, N)
    return f_approx
