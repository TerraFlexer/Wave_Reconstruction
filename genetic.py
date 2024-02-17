import numpy as np
import functions_package as fpckg


N = fpckg.N

generation_size = 10


def initiation():
    ampl = 0.25
    center = 0.5

    first_gen = np.zeros((generation_size, N, N))

    for i in range(generation_size):
        first_gen[i] = center + ampl * np.random.randn(N, N)

    return first_gen


def selection(generation):
    scores = np.ones(generation_size)
    for i in range(generation_size):
        scores[i] = fpckg.perform_trial(generation[i], 3, 3)

    return scores


print(selection(initiation()))