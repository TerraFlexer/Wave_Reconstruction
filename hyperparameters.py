import optuna
import numpy as np
from method import method_v
from main import spiral, Edge, N, add_noise
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from scipy.special import kl_div


def objective(trial):
    gamma = trial.suggest_float('gammas', 0, 5)

    x = np.linspace(-Edge, Edge, N, endpoint=False)
    y = np.linspace(-Edge, Edge, N, endpoint=False)
    Y, X = np.meshgrid(x, y)

    accuracy = 0
    ssim_accuracy = 0
    kl_divergence = 0

    for el in enumerate(np.linspace(0.0, 0.05, num=5)):
        z = spiral(3, 1, X, Y)

        z1 = add_noise(z, 0.15, N)

        z_approx = method_v(z1, N, Edge, 1, gamma)

        offs = np.min(z_approx) - np.min(z)

        accuracy += mse(z, z_approx - offs)

        ssim_accuracy += (1 - ssim(z, z_approx - offs,
                                   data_range=max(np.max(z), np.max(z_approx - offs)) -
                                              min(np.min(z), np.min(z_approx - offs)))) / 2

        kl_divergence += np.sum(kl_div(z + 5, z_approx - offs + 5))

    return kl_divergence


# 3. Create a study object and optimize the objective function.
study = optuna.create_study()
study.optimize(objective, n_trials=100)

print(study.best_params)
