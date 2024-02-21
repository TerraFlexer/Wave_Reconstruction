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
        scores[i] = fpckg.perform_trial(generation[i], 3, 3)

    return scores


def random_specimen():
    return np.random.randint(0, generation_size, 1)[0]


def crossover(parent1, parent2):
    k = 20

    child = parent1
    for l in range(k):
        i = np.random.randint(0, N, 1)[0]
        j = np.random.randint(0, N, 1)[0]
        child[i][j] = parent2[i][j]

    return child


def mutation(specimen):
    return specimen + np.random.randn(N, N) * 0.01


def scores_plot(epochs, avg_scores, best_scores, glbl_scores):
    epochs_arr = np.arange(epochs)
    plt.plot(epochs_arr, avg_scores, label='Средний score (по поколению)')
    plt.plot(epochs_arr, best_scores, label='Лучший score (в поколении)')
    plt.plot(epochs_arr, glbl_scores, label='score при gamma=0.5')
    plt.ylabel("Score")
    plt.xlabel("Эпоха")
    plt.legend()
    plt.show()


def life_cycle(eps=3, epochs=50, mutation_prob=0.08):
    generation = initiation()
    new_generation = []

    avg_scores = np.zeros(epochs)
    best_scores = np.zeros(epochs)
    glbl_gamma_scores = np.zeros(epochs)

    for i in range(epochs):
        print("Epoch " + str(i))
        scores = selection(generation)

        glbl_gamma_scores[i] = fpckg.perform_trial(np.ones((N, N)) * 0.5, 3, 2)

        avg_scores[i] = np.average(scores)
        best_scores[i] = np.min(scores)

        best5 = np.argpartition(scores, 5)[:5]
        print("5 best specimen are " + str(best5) + " with scores: " + str(scores[best5]))
        print("Score with global gamma = 0.5: " + str(glbl_gamma_scores[i]))

        new_generation = generation

        for j in range(generation_size):
            if j not in best5:
                new_generation[j] = crossover(generation[random_specimen()], generation[random_specimen()])
            if np.random.random() < mutation_prob:
                new_generation[j] = mutation(generation[j])
                print("Mutation occured on " + str(j) + "th specimen")

        generation = new_generation

        print("\n")

    print("Life Cycle ended")
    scores_plot(epochs, avg_scores, best_scores, glbl_gamma_scores)


life_cycle()
