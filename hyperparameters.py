import optuna
import numpy as np
from method import method_v
from main import spiral, Edge, N, add_noise, multifocal, offset
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from scipy.special import kl_div
import matplotlib.pyplot as plt


def objective(trial):
    gammas = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            gammas[i][j] = trial.suggest_float(f"gammas{i}{j}", 0.4, 0.6)

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
study.optimize(objective, n_trials=200)

print(study.best_params)

gammas = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        gammas[i][j] = study.best_params[f"gammas{i}{j}"]


x = np.linspace(0, N - 1, N, endpoint=False)
y = np.linspace(0, N - 1, N, endpoint=False)
Y, X = np.meshgrid(x, y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf1 = ax.plot_surface(X, Y, gammas, cmap='plasma')
ax.set_title('gammas')
ax.view_init(45, 60)
plt.show()