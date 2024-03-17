import optuna
import numpy as np
from method import fill_gammas
import matplotlib.pyplot as plt
import functions_package as fpckg


N = fpckg.N


def objective(trial):
    gamma_center = trial.suggest_float(f"gamma_center", 0.25, 0.75)
    gamma_middle1 = trial.suggest_float(f"gamma_middle1", 0.25, 0.75)
    gamma_middle2 = trial.suggest_float(f"gamma_middle2", 0.25, 0.75)
    gamma_side = trial.suggest_float(f"gamma_side", 0.25, 0.75)

    gammas = fill_gammas([gamma_center, gamma_middle1, gamma_middle2], [8, 15, 25], gamma_side, N)

    result = fpckg.perform_trial(gammas)

    return result


# 3. Create a study object and optimize the objective function.
study = optuna.create_study()
study.optimize(objective, n_trials=100)

print(study.best_params)

gamma_center = study.best_params[f"gamma_center"]
gamma_middle1 = study.best_params[f"gamma_middle1"]
gamma_middle2 = study.best_params[f"gamma_middle2"]
gamma_side = study.best_params[f"gamma_side"]

gammas = fill_gammas([gamma_center, gamma_middle1, gamma_middle2], [8, 15, 25], gamma_side, N)

fpckg.visualaize_param_matrix(gammas)
