import optuna
import numpy as np
from method import method_v
from main import spiral, Edge, N, add_noise, multifocal, offset
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from scipy.special import kl_div
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt


def objective(trial):
    gammas = np.ones((N, N)) * 0.5
    gamma_center = trial.suggest_float(f"gamma_center", 0.25, 0.75)
    gamma_middle1 = trial.suggest_float(f"gamma_middle1", 0.25, 0.75)
    gamma_middle2 = trial.suggest_float(f"gamma_middle2", 0.25, 0.75)
    for i in range(N):
        for j in range(N):
            if 25 < (N // 2 - 1 - i) ** 2 + (N // 2 - 1 - j) ** 2 <= 36:
                gammas[i][j] = gamma_middle1
    for i in range(N):
        for j in range(N):
            if 9 < (N // 2 - 1 - i) ** 2 + (N // 2 - 1 - j) ** 2 <= 25:
                gammas[i][j] = gamma_middle2
    for i in range(N):
        for j in range(N):
            if (N // 2 - 1 - i) ** 2 + (N // 2 - 1 - j) ** 2 <= 9:
                gammas[i][j] = gamma_center

    x = np.linspace(-Edge, Edge, N, endpoint=False)
    y = np.linspace(-Edge, Edge, N, endpoint=False)
    Y, X = np.meshgrid(x, y)

    accuracy = 0
    ssim_accuracy = 0
    kl_divergence = 0

    for el in enumerate(np.linspace(0.0, 0.05, num=5)):
        z = multifocal([1, 3], [0.8, -1.5])

        z1 = add_noise(z, 0.15, N)

        z_approx = method_v(z1, N, Edge, 1, gammas)

        offs = offset(z_approx, z, N, 0)

        accuracy += mse(z, z_approx - offs)

        ssim_accuracy += (1 - ssim(z, z_approx - offs,
                                   data_range=max(np.max(z), np.max(z_approx - offs)) -
                                              min(np.min(z), np.min(z_approx - offs)))) / 2

        kl_divergence += np.sum(kl_div(z + 5, z_approx - offs + 5))

    return kl_divergence


# 3. Create a study object and optimize the objective function.
study = optuna.create_study()
study.optimize(objective, n_trials=500)

print(study.best_params)

gammas = np.ones((N, N)) * 0.5
gamma_center = study.best_params[f"gamma_center"]
gamma_middle1 = study.best_params[f"gamma_middle1"]
gamma_middle2 = study.best_params[f"gamma_middle2"]

for i in range(N):
    for j in range(N):
        if 25 < (N // 2 - 1 - i) ** 2 + (N // 2 - 1 - j) ** 2 <= 36:
            gammas[i][j] = gamma_middle1
for i in range(N):
    for j in range(N):
        if 9 < (N // 2 - 1 - i) ** 2 + (N // 2 - 1 - j) ** 2 <= 25:
            gammas[i][j] = gamma_middle2
for i in range(N):
    for j in range(N):
        if (N // 2 - 1 - i) ** 2 + (N // 2 - 1 - j) ** 2 <= 9:
            gammas[i][j] = gamma_center

x = np.linspace(0, N - 1, N, endpoint=False)
y = np.linspace(0, N - 1, N, endpoint=False)
Y, X = np.meshgrid(x, y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf1 = ax.plot_surface(X, Y, gammas, cmap='plasma')
ax.set_title('gammas')
ax.view_init(45, 60)
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf1 = ax.plot_surface(X, Y, gammas, cmap='plasma')
ax.set_title('gammas')
ax.view_init(0, 90)
plt.show()

fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
surf2 = ax1.plot_surface(X, Y, ifft2(gammas).real, cmap='plasma')
ax.set_title('gammas')
ax.view_init(45, 60)
plt.show()
