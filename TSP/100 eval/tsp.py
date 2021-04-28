import numpy as np
import scipy.spatial
from matplotlib import pyplot as plt
import tqdm


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


def weighted_selection(route1, dist):
    n = len(route1)
    w = np.zeros(n)
    for i in range(1, n - 1):
        w[i] = max(dist[route1[i - 1], route1[i]], dist[route1[i], route1[i + 1]])

    w[0] = max(dist[route1[n - 1], route1[0]], dist[route1[0], route1[1]])
    w[n - 1] = max(dist[route1[n - 2], route1[n - 1]], dist[route1[n - 1], route1[0]])
    w = w ** 15
    # w = np.ones(n)
    i1 = np.random.choice(np.arange(n), 1, p=w / np.sum(w))[0]
    # w = 1. / dist[:, i1]
    # w = w
    # w[i1] = 0
    # i2 = np.random.choice(np.arange(n), 1, p=w / np.sum(w))[0]
    i2 = np.random.choice(np.concatenate([np.arange(i1), np.arange(i1 + 1, n)]), 1)[0]

    return i1, i2


def inversion(route1, dist):
    i1, i2 = weighted_selection(route1, dist)
    min_i = min(i1, i2)
    max_i = max(i1, i2)
    route2 = route1.copy()
    route2[min_i:max_i] = reversed(route1[min_i:max_i])
    return route2


def two_switch(route1, dist):
    i1, i2 = weighted_selection(route1, dist)
    route2 = route1.copy()
    route2[i1] = route1[i2]
    route2[i2] = route1[i1]
    return route2


def insertion(route1, dist):
    i1, i2 = weighted_selection(route1, dist)
    route2 = route1.copy()
    if i1 < i2:
        route2[i1:i2] = route1[i1 + 1:i2 + 1]
        route2[i2] = route1[i1]
    else:
        route2[i2 + 1:i1 + 1] = route1[i2:i1]
        route2[i2] = route1[i1]
    return route2


def check_duplicate(new_solution, unique_solutions):
    next_array_1 = np.zeros(len(new_solution))
    next_array_2 = np.zeros(len(new_solution))
    for idx in range(len(new_solution) - 1):
        next_array_1[new_solution[idx]] = new_solution[idx + 1]
    next_array_1[len(new_solution) - 1] = new_solution[0]
    for idx in range(len(new_solution) - 1, 0, -1):
        next_array_2[new_solution[idx]] = new_solution[idx - 1]
    flag = False
    for array in unique_solutions:
        if (array == next_array_1).all() or (array == next_array_2).all():
            flag = True
            break
    return flag, next_array_1


def main():
    rand_two_switch = 0
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
    unique_solutions = []

    for i in range(len(data)):
        cur_route, cur_route_length = greedy(dist, i)
        is_dup, next_array = check_duplicate(cur_route, unique_solutions)
        if not is_dup:
            unique_solutions.append(next_array)
        else:
            print('dup greedy')

        if cur_route_length < route_length:
            route_length = cur_route_length
            solution = cur_route

    # greedy solution
    print(route_length)
    greedy_solution = [*solution, solution[0]]
    plt.figure()
    plt.plot(data[greedy_solution][:, 0], data[greedy_solution][:, 1], 'k-o', linewidth=1.5, markerfacecolor='white',
             markersize=10)

    best_solution = []
    best_length = np.inf
    best_curve = []
    all_lengths = []
    for seed in tqdm.tqdm(range(1000)):
        np.random.seed(seed)
        seed_unique_solutions = unique_solutions.copy()
        seed_solution = solution.copy()
        seed_route_length = route_length
        seed_curve = [seed_route_length]
        for i in range(101):
            while True:
                rand_num = np.random.rand(1)
                if rand_num < rand_two_switch:
                    new_solution = two_switch(seed_solution, dist)
                elif rand_num < rand_two_switch + rand_inversion:
                    new_solution = inversion(seed_solution, dist)
                else:
                    new_solution = insertion(seed_solution, dist)
                is_dup, next_array = check_duplicate(new_solution, seed_unique_solutions)
                if not is_dup:
                    seed_unique_solutions.append(next_array)
                    break
            new_length = compute_length(new_solution, dist)
            if new_length < seed_route_length:
                seed_route_length = new_length
                seed_solution = new_solution
            seed_curve.append(seed_route_length)

        all_lengths.append(seed_route_length)
        if seed_route_length < best_length:
            best_length = seed_route_length
            best_solution = seed_solution
            best_curve = seed_curve

    print(best_length)
    print(sum(all_lengths) / len(all_lengths))
    best_solution = [*best_solution, best_solution[0]]
    assert len(best_solution) == 101
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
