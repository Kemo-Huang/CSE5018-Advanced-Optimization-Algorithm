import numpy as np
import scipy.spatial
from time import time
from matplotlib import pyplot as plt


def greedy(dist, start_idx):
    solution = [start_idx]
    cur_idx = start_idx
    _dist = dist.copy()
    route_length = 0
    for _ in range(len(_dist) - 1):
        tmp_dist = _dist[cur_idx, :]
        tmp_dist[solution] = np.inf
        cur_idx = np.argmin(tmp_dist)
        solution.append(cur_idx)
        route_length += tmp_dist[cur_idx]
    route_length += _dist[start_idx, cur_idx]
    return solution, route_length


def compute_length(route, dist):
    n = len(route)
    route = [*route, route[0]]
    length = 0
    for k in range(n):
        length += dist[route[k], route[k + 1]]
    return length


def weighted_selection(route1, dist, power=1.0):
    n = len(route1)
    w = np.zeros(n)
    for i in range(1, n - 1):
        w[i] = max(dist[route1[i - 1], route1[i]], dist[route1[i], route1[i + 1]])

    w[0] = max(dist[route1[n - 1], route1[0]], dist[route1[0], route1[1]])
    w[n - 1] = max(dist[route1[n - 2], route1[n - 1]], dist[route1[n - 1], route1[0]])
    w = w ** power
    i1 = np.random.choice(np.arange(n), 1, p=w / np.sum(w))[0]
    # w = 1. / dist[:, i1]
    # w[i1] = 0
    # w = w
    # i2 = np.random.choice(np.arange(n), 1, p=w / np.sum(w))[0]
    i2 = np.random.choice(np.concatenate([np.arange(i1), np.arange(i1 + 1, n)]), 1)[0]
    return i1, i2


def uniform_selection(route1):
    return np.random.choice(np.arange(len(route1)), 2)


def inversion(route1, i1, i2):
    min_i = min(i1, i2)
    max_i = max(i1, i2)
    route2 = route1.copy()
    route2[min_i:max_i] = reversed(route1[min_i:max_i])
    return route2


def insertion(route1, i1, i2):
    route2 = route1.copy()
    if i1 < i2:
        route2[i1:i2] = route1[i1 + 1:i2 + 1]
        route2[i2] = route1[i1]
    else:
        route2[i2 + 1:i1 + 1] = route1[i2:i1]
        route2[i2] = route1[i1]
    return route2


def check_duplicate(new_solution, unique_solutions):
    next_array = np.zeros(len(new_solution))
    for idx in range(len(new_solution) - 1):
        next_array[new_solution[idx]] = new_solution[idx + 1]
    next_array[len(new_solution) - 1] = new_solution[0]

    flag = False
    for array in unique_solutions:
        if (array == next_array).all():
            flag = True
            break
    return flag, next_array


def main():
    rand_inversion = 0.5

    with open('TSP_100Cities.txt') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        cur_x, cur_y = line.split()
        data.append((float(cur_x), float(cur_y)))
    data = np.array(data)
    dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data))
    route_length = np.inf
    solution = []
    start_time = time()
    for i in range(len(data)):
        cur_route, cur_route_length = greedy(dist, i)

        if cur_route_length < route_length:
            route_length = cur_route_length
            solution = cur_route
    greedy_time = time() - start_time
    print("greedy time: ", greedy_time)

    best_solution = []
    best_length = np.inf
    best_curve = []
    all_lengths = []

    start_time = time()
    for seed in [2333]:
        np.random.seed(seed)
        # seed_unique_solutions = unique_solutions.copy()
        seed_solution = solution.copy()
        seed_best_solution = solution.copy()
        seed_route_length = route_length
        seed_best_length = route_length
        seed_curve = [seed_route_length]
        conv_counter = 0
        evaluation = 100
        while evaluation < 2e6:
            # while True:
            # i1, i2 = weighted_selection(seed_solution, dist)
            i1, i2 = uniform_selection(seed_solution)
            if np.random.rand(1) < rand_inversion:
                new_solution = inversion(seed_solution, i1, i2)
            else:
                new_solution = insertion(seed_solution, i1, i2)
                # is_dup, next_array = check_duplicate(new_solution, seed_unique_solutions)
                # if not is_dup:
                #     seed_unique_solutions.append(next_array)
                #     break
            new_length = compute_length(new_solution, dist)
            evaluation += 1
            if new_length < seed_route_length:
                seed_route_length = new_length
                seed_solution = new_solution
                # conv_counter = 0
            else:
                # new solution is worse than seed solution
                conv_counter += 1
                if conv_counter >= 30000:
                    if seed_route_length < seed_best_length:
                        seed_best_length = seed_route_length
                        seed_best_solution = seed_solution
                    while new_length == seed_route_length or new_length > seed_route_length + 5:
                        i1, i2 = uniform_selection(seed_solution)
                        if np.random.rand(1) < rand_inversion:
                            new_solution = inversion(seed_solution, i1, i2)
                        else:
                            new_solution = insertion(seed_solution, i1, i2)
                        new_length = compute_length(new_solution, dist)
                        evaluation += 1
                    seed_route_length = new_length
                    seed_solution = new_solution
                    conv_counter = 0
            seed_curve.append(seed_route_length)

        all_lengths.append(seed_best_length)
        if seed_best_length < best_length:
            best_length = seed_best_length
            best_solution = seed_best_solution
            best_curve = seed_curve
    print('total time: ', greedy_time + time() - start_time)
    print('best: ', best_length)
    print('avg: ', sum(all_lengths) / len(all_lengths))
    best_solution = [*best_solution, best_solution[0]]
    plt.figure()
    plt.plot(best_curve)
    plt.ylabel('total length')
    plt.xlabel('evaluations')

    plt.figure()
    plt.plot(data[best_solution][:, 0], data[best_solution][:, 1], 'k-o', linewidth=1.5, markerfacecolor='white',
             markersize=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    main()
