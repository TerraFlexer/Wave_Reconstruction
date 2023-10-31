import optuna
import numpy as np
from method import method_v
from main import spiral, Edge, N, add_noise
from skimage.metrics import mean_squared_error as mse


def objective(trial):

    # 2. Suggest values of the hyperparameters using a trial object.
    gamma = trial.suggest_float('gammas', 0, 10)

    x = np.linspace(-Edge, Edge, N, endpoint=False)
    y = np.linspace(-Edge, Edge, N, endpoint=False)
    Y, X = np.meshgrid(x, y)

    z = spiral(3, 1, X, Y)

    z1 = add_noise(z, 0.15, N)

    z_approx = method_v(z1, N, Edge, 1, gamma)

    offs = np.min(z_approx) - np.min(z)

    accuracy = mse(z, z_approx - offs)

    return accuracy


# 3. Create a study object and optimize the objective function.
study = optuna.create_study()
study.optimize(objective, n_trials=300)

print(study.best_params)
