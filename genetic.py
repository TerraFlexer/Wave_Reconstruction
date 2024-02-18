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
        scores[i] = fpckg.perform_trial(generation[i], 3, 5)

    return scores


def random_specimen():
    return np.random.randint(0, generation_size - 1, 1)


def crossover(parent1, parent2):
    return (parent1 + parent2) / 2


def mutation(specimen):
    return specimen + np.random.randn(N, N) * 0.01


def life_cycle(eps=3, epochs=20, mutation_prob=0.05):
    generation = initiation()
    new_generation = []

    for i in range(epochs):
        print("Epoch " + str(i))
        scores = selection(generation)
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


life_cycle()
