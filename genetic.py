import numpy as np
import matplotlib.pyplot as plt
import functions_package as fpckg

N = fpckg.N

generation_size = 20


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
        scores[i] = fpckg.perform_trial(generation[i], 3, 1)

    return scores


def random_specimen():
    return np.random.randint(0, generation_size - 1, 1)


def crossover(parent1, parent2):
    return (parent1 + parent2) / 2


def mutation(specimen):
    return specimen + np.random.randn(N, N) * 0.01


def scores_plot(epochs, avg_scores, best_scores):
    epochs_arr = np.arange(epochs)
    plt.plot(epochs_arr, avg_scores, label='Средний score (по поколению)')
    plt.plot(epochs_arr, best_scores, label='Лучший score (в поколении)')
    plt.ylabel("Score")
    plt.xlabel("Эпоха")
    plt.legend()
    plt.show()


def life_cycle(eps=3, epochs=30, mutation_prob=0.05):
    generation = initiation()
    new_generation = []

    avg_scores = np.zeros(epochs)
    best_scores = np.zeros(epochs)

    for i in range(epochs):
        print("Epoch " + str(i))
        scores = selection(generation)

        avg_scores[i] = np.average(scores)
        best_scores[i] = np.min(scores)

        best5 = np.argpartition(scores, -5)[-5:]
        print("5 best specimen are " + str(best5) + " with scores: " + str(scores[best5]))

        new_generation = generation

        for j in range(generation_size):
            if j not in best5:
                new_generation[j] = crossover(generation[random_specimen()], generation[random_specimen()])
            elif np.random.random() < mutation_prob:
                new_generation[j] = mutation(generation[j])
                print("Mutation occured on " + str(j) + "th specimen")

        generation = new_generation

        print("\n")

    print("Life Cycle ended")
    scores_plot(epochs, avg_scores, best_scores)


life_cycle()
