import itertools
from multiprocessing import Pool
from time import time

import numpy as np
from matplotlib import pyplot as plt
from numba import jit
import numba
import tqdm


@jit
def inversion(arguments):
    i1, i2, solution, processing_time = arguments
    new_solution = solution.copy()
    i_min = min(i1, i2)
    i_max = max(i1, i2) + 1
    new_solution[i_min:i_max] = np.flip(solution[i_min:i_max])
    return new_solution, compute_makespan(new_solution, processing_time)


@jit
def insertion(arguments):
    i1, i2, solution, processing_time = arguments
    new_solution = solution.copy()
    if i1 < i2:
        new_solution[i1:i2] = solution[i1 + 1:i2 + 1]
        new_solution[i2] = solution[i1]
    else:
        new_solution[i2 + 1:i1 + 1] = solution[i2:i1]
        new_solution[i2] = solution[i1]
    return new_solution, compute_makespan(new_solution, processing_time)


@jit
def arbitrary_two_change(arguments):
    i1, i2, solution, processing_time = arguments
    new_solution = solution.copy()
    new_solution[i1] = solution[i2]
    new_solution[i2] = solution[i1]
    return new_solution, compute_makespan(new_solution, processing_time)


def rapid_access_procedure(processing_time):
    virtual_first_machine = np.zeros(100)
    for i in range(10):
        virtual_first_machine += (10 - i) * processing_time[i]
    virtual_second_machine = np.zeros(100)
    for i in range(10):
        virtual_second_machine += (i + 1) * processing_time[i]
    return johnson_method(virtual_first_machine, virtual_second_machine)


def johnson_method(m1, m2):
    remaining_positions = list(range(100))
    remaining_jobs = list(range(100))
    all_time = np.vstack([m1, m2]).T
    solution = np.zeros(100, np.int)
    for _ in range(99):
        remaining_time = all_time[remaining_jobs]
        job_idx, machine_idx = np.unravel_index(
            np.argmin(remaining_time), remaining_time.shape)
        if machine_idx == 0:
            solution[remaining_positions[0]] = remaining_jobs[job_idx]
            del remaining_positions[0]
        else:
            solution[remaining_positions[-1]] = remaining_jobs[job_idx]
            del remaining_positions[-1]
        del remaining_jobs[job_idx]
    solution[remaining_positions[0]] = remaining_jobs[0]
    return solution


@jit
def compute_makespan(solution, processing_time):
    makespans = np.zeros(100, dtype=numba.int64)
    for job in solution:
        for machine in range(10):
            if machine == 0:
                makespans[0] += processing_time[0, job]
            else:
                makespans[machine] = max(makespans[machine - 1], makespans[machine]) + processing_time[
                    machine, job]
    return np.max(makespans)


def one_step_search(pool, neighbor_function, solution, processing_time, indices_list):
    neighborhood_arguments = [i + (solution, processing_time) for i in indices_list]
    new_solutions_and_makespans = pool.map(neighbor_function, neighborhood_arguments)
    new_makespans = [i[1] for i in new_solutions_and_makespans]
    best_idx = np.argmin(new_makespans)
    one_step_min_makespan = new_makespans[best_idx]
    one_step_best_solution = new_solutions_and_makespans[best_idx][0]
    return one_step_min_makespan, one_step_best_solution


def variable_neighborhood_search(solution, makespan, processing_time):
    combinations = list(itertools.combinations(range(100), 2))
    permutations = list(itertools.permutations(range(100), 2))
    curve = [makespan]

    pool = Pool()
    evaluation_times = 1
    k = 1
    while evaluation_times < 2000000:
        if k == 1:
            neighborhood = inversion
            neighbors = combinations
        elif k == 2:
            neighborhood = insertion
            neighbors = permutations
        else:
            neighborhood = arbitrary_two_change
            neighbors = combinations
        # shaking
        i1, i2 = np.random.choice(100, 2, False)
        local_solution, local_makespan = neighborhood((i1, i2, solution, processing_time))
        evaluation_times += 1
        curve.append(local_makespan)
        # local search
        while True:
            one_step_min_makespan, one_step_best_solution = one_step_search(pool,
                                                                            neighborhood,
                                                                            local_solution,
                                                                            processing_time,
                                                                            neighbors)
            evaluation_times += len(neighbors)
            curve += [one_step_min_makespan] * len(neighbors)
            if one_step_min_makespan < local_makespan:
                local_makespan = one_step_min_makespan
                local_solution = one_step_best_solution
            else:
                break
        # move or not
        if local_makespan < makespan:
            makespan = local_makespan
            solution = local_solution
            k = 1
        else:
            k += 1
            if k > 3:
                k = 1

    return solution, makespan, curve


def main():
    with open('data.txt', 'r') as f:
        processing_time = np.vstack(
            [np.array(line.split(), np.int) for line in f.readlines()])
    start = time()
    # initialization
    solution = rapid_access_procedure(processing_time)
    makespan = compute_makespan(solution, processing_time)
    init_time = time() - start
    makespans = []
    solutions = []
    curves = []

    runs = 10
    avg_time = 0

    for seed in tqdm.tqdm(range(runs)):
        np.random.seed(seed)
        start = time()
        best_solution, min_makespan, curve = variable_neighborhood_search(solution, makespan, processing_time)
        avg_time += time() - start
        solutions.append(best_solution)
        makespans.append(min_makespan)
        curves.append(curve)

    avg_time /= runs
    avg_time += init_time

    best_ind = np.argmin(makespans)
    min_makespan = makespans[best_ind]
    best_curve = curves[best_ind]

    print('avg time', avg_time)
    print('avg makespan', sum(makespans) / runs)
    print('min makespan', min_makespan)

    best_solution = solutions[best_ind]
    np.save('result.npy', best_solution)

    plt.figure()
    plt.plot(best_curve)
    plt.xlabel('evaluations')
    plt.ylabel('makespan')
    plt.show()


if __name__ == '__main__':
    main()
