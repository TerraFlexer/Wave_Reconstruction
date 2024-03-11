import numpy as np
import matplotlib.pyplot as plt
import functions_package as fpckg
from method import fill_gammas
from metrics import count_metrics

N = fpckg.N

generation_size = 20


def initiation():
    ampl = 0.25
    center = 0.5

    first_gen = np.zeros((generation_size, N, N))

    for i in range(generation_size):
        # first_gen[i] = center + ampl * np.random.randn(N, N)
        gamma_center = center + ampl * np.random.randn()
        gamma_middle1 = center + ampl * np.random.randn()
        gamma_middle2 = center + ampl * np.random.randn()
        gamma_side = center + ampl * np.random.randn()

        first_gen[i] = fill_gammas([gamma_center, gamma_middle1, gamma_middle2], [8, 15, 25], gamma_side, N)

    x = np.linspace(0, N - 1, N, endpoint=False)
    y = np.linspace(0, N - 1, N, endpoint=False)
    Y, X = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    surf1 = ax.plot_surface(X, Y, first_gen[0], cmap='plasma')
    ax.view_init(45, 60)

    ax = fig.add_subplot(222, projection='3d')
    surf1 = ax.plot_surface(X, Y, first_gen[5], cmap='plasma')
    ax.view_init(45, 60)

    ax = fig.add_subplot(223, projection='3d')
    surf1 = ax.plot_surface(X, Y, first_gen[10], cmap='plasma')
    ax.view_init(45, 60)

    ax = fig.add_subplot(224, projection='3d')
    surf1 = ax.plot_surface(X, Y, first_gen[15], cmap='plasma')
    ax.view_init(45, 60)

    plt.show()

    return first_gen


def selection(generation):
    scores = np.ones(generation_size)
    for i in range(generation_size):
        scores[i] = fpckg.perform_trial(generation[i], 3, 2)

    return scores


def random_specimen():
    return np.random.randint(0, generation_size, 1)[0]


def crossover(parent1, parent2):
    k = N * N // 3

    pairs = set()

    child1 = parent1
    child2 = parent2
    for l in range(k):
        i = np.random.randint(0, N, 1)[0]
        j = np.random.randint(0, N, 1)[0]

        while (i, j) in pairs:
            i = np.random.randint(0, N, 1)[0]
            j = np.random.randint(0, N, 1)[0]
        pairs.add((i, j))
        child1[i][j] = parent2[i][j]
        child2[i][j] = parent1[i][j]

    return child1, child2


def mutation(specimen):
    return specimen + np.random.randn(N, N) * 0.05


def scores_plot(epochs, avg_scores, best_scores, glbl_scores):
    # Function for plotting scores
    epochs_arr = np.arange(epochs)
    plt.plot(epochs_arr, avg_scores, label='Средний score (по поколению)')
    plt.plot(epochs_arr, best_scores, label='Лучший score (в поколении)')
    plt.plot(epochs_arr, glbl_scores, label='score при gamma=0.5')
    plt.ylabel("Score")
    plt.xlabel("Эпоха")
    plt.legend()
    plt.show()


def life_cycle(eps=3, epochs=30, mutation_prob=0.08):
    # Initialazing first generation
    generation = initiation()

    # Initialazing arrays for graph scores
    avg_scores = np.zeros(epochs)
    best_scores = np.zeros(epochs)
    glbl_gamma_scores = np.zeros(epochs)

    # Variables for saving best score and best gamma
    best_score = 20
    best_gamma = np.ones((N, N)) * 0.5

    for i in range(epochs):
        print("Epoch " + str(i))

        # Counting scores for generation
        scores = selection(generation)

        # Counting score for global gamma choice
        glbl_gamma_scores[i] = fpckg.perform_trial(np.ones((N, N)) * 0.5, 3, 2)

        # Counting avg and best scores for graphs
        avg_scores[i] = np.average(scores)
        best_scores[i] = np.min(scores)

        # Updating the best gamma if necessary
        if np.min(scores) < best_score:
            best_score = np.min(scores)
            best_gamma = generation[np.argmin(scores)]

        # Finding 6 best specimen and printing their results
        best6 = np.argpartition(scores, 6)[:6]
        print("6 best specimen are " + str(best6) + " with scores: " + str(scores[best6]))
        print("Score with global gamma = 0.5: " + str(glbl_gamma_scores[i]))

        # Initialazing new generation
        new_generation = np.zeros(np.shape(generation))

        # Saving 6 best specimen for next_generation
        ind = 0
        while ind < 6:
            new_generation[ind] = generation[best6[ind]]
            ind += 1

        # Filling remaining slots in new generation with kids
        ind = 6
        while ind < generation_size:
            new_generation[ind], new_generation[ind + 1] = crossover(generation[random_specimen()],
                                                                     generation[random_specimen()])
            ind += 2

        # Comitting mutations with specimen with mutation_probability
        for j in range(generation_size):
            if np.random.random() < mutation_prob:
                new_generation[j] = mutation(generation[j])
                print("Mutation occured on " + str(j) + "th specimen")

        generation = new_generation

        print("\n")

    print("Life Cycle ended")
    # Visualize graphs through lifetime
    scores_plot(epochs, avg_scores, best_scores, glbl_gamma_scores)

    # Visualaize best_gamma
    fpckg.visualaize_param_matrix(best_gamma)

    # Count and visualaize metrics for best_gamma
    count_metrics(best_gamma)


life_cycle()
