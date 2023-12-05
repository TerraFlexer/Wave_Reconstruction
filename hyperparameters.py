import optuna
import numpy as np
from method import method_v, fill_gammas
from main import spiral, Edge, N, add_noise, multifocal, offset, generate_random_multifocal
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from scipy.special import kl_div
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt


def objective(trial):
    gamma_center = trial.suggest_float(f"gamma_center", 0.25, 0.75)
    gamma_middle1 = trial.suggest_float(f"gamma_middle1", 0.25, 0.75)
    gamma_middle2 = trial.suggest_float(f"gamma_middle2", 0.25, 0.75)
    gamma_side = trial.suggest_float(f"gamma_side", 0.25, 0.75)

    gammas = fill_gammas([gamma_center, gamma_middle1, gamma_middle2], [8, 15, 25], gamma_side, N)

    x = np.linspace(-Edge, Edge, N, endpoint=False)
    y = np.linspace(-Edge, Edge, N, endpoint=False)
    Y, X = np.meshgrid(x, y)

    accuracy = 0
    ssim_accuracy = 0
    kl_divergence = 0

    for el in enumerate(np.linspace(0.0, 0.05, num=5)):
        for i in range(5):
            z = generate_random_multifocal()

            z1 = add_noise(z, 0.15, N)

            z_approx = method_v(z1, N, Edge, 1, gammas)

            offs = offset(z_approx, z, N, 0)

            accuracy += mse(z, z_approx - offs)

            ssim_accuracy += (1 - ssim(z, z_approx - offs,
                                   data_range=max(np.max(z), np.max(z_approx - offs)) -
                                              min(np.min(z), np.min(z_approx - offs)))) / 2

            kl_divergence += np.sum(kl_div(z + 5, z_approx - offs + 5))

    return kl_divergence + accuracy + ssim_accuracy


# 3. Create a study object and optimize the objective function.
study = optuna.create_study()
study.optimize(objective, n_trials=100)

print(study.best_params)

gamma_center = study.best_params[f"gamma_center"]
gamma_middle1 = study.best_params[f"gamma_middle1"]
gamma_middle2 = study.best_params[f"gamma_middle2"]
gamma_side = study.best_params[f"gamma_side"]

gammas = fill_gammas([gamma_center, gamma_middle1, gamma_middle2], [8, 15, 25], gamma_side, N)

x = np.linspace(0, N - 1, N, endpoint=False)
y = np.linspace(0, N - 1, N, endpoint=False)
Y, X = np.meshgrid(x, y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf1 = ax.plot_surface(X, Y, gammas, cmap='plasma')
ax.set_title('gammas')
ax.view_init(45, 60)
plt.show()

fig1, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf2 = ax.plot_surface(X, Y, gammas, cmap='plasma')
ax.set_title('gammas')
ax.view_init(90, 0)
plt.show()
