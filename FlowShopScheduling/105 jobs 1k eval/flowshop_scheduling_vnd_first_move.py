import itertools
import json
from time import time

import numba
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from numba import jit


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
    m, n = processing_time.shape
    virtual_first_machine = np.zeros(n)
    for i in range(m):
        virtual_first_machine += (m - i) * processing_time[i]
    virtual_second_machine = np.zeros(n)
    for i in range(m):
        virtual_second_machine += (i + 1) * processing_time[i]
    return johnson_method(virtual_first_machine, virtual_second_machine)


def johnson_method(m1, m2):
    remaining_positions = list(range(105))
    remaining_jobs = list(range(105))
    all_time = np.vstack([m1, m2]).T
    solution = np.zeros(105, np.int)
    for _ in range(104):
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


def neh_heuristic(processing_time, solution):
    remaining_processing_time = processing_time[:, 100:]
    total_processing_time = np.sum(remaining_processing_time, axis=0)
    indices = np.argsort(total_processing_time)
    for idx in indices:
        true_idx = idx + 100
        tmp_solution = np.concatenate((np.array([true_idx]), solution))
        tmp_makespan = compute_makespan(tmp_solution, processing_time)
        best_solution = tmp_solution
        min_makespan = tmp_makespan
        for i2 in range(1, len(tmp_solution), 1):
            new_solution, new_makespan = insertion((0, i2, tmp_solution, processing_time))
            if new_makespan < min_makespan:
                best_solution = new_solution
                min_makespan = new_makespan
        solution = best_solution
    return np.array(solution), min_makespan


@jit
def compute_makespan(solution, processing_time):
    makespans = np.zeros(len(solution), dtype=numba.int64)
    for job in solution:
        for machine in range(processing_time.shape[0]):
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


def first_move(solution, makespan, processing_time, evaluations):
    curve = [makespan]
    k = 1
    for _ in range(evaluations - 1):
        i1, i2 = np.random.choice(len(solution), 2, replace=False)
        if k == 1:
            neighborhood = inversion
        elif k == 2:
            neighborhood = insertion
        else:
            neighborhood = arbitrary_two_change
        new_solution, new_makespan = neighborhood((i1, i2, solution, processing_time))
        curve.append(new_makespan)
        if new_makespan < makespan:
            makespan = new_makespan
            solution = new_solution
        k += 1

    return solution, makespan, curve


def first_move(solution, makespan, processing_time, indices, neighborhood):
    evaluations = 0
    for i1, i2 in indices:
        evaluations += 1
        new_solution, new_makespan = neighborhood((i1, i2, solution, processing_time))
        if new_makespan < makespan:
            makespan = new_makespan
            solution = new_solution
            break

    return solution, makespan, evaluations


def main():
    with open('data.txt', 'r') as f:
        processing_time = np.vstack(
            [np.array(line.split(), np.int) for line in f.readlines()])
    with open('6030.json', 'r') as f:
        solutions = [np.array(solution, np.int) for solution in json.load(f)]

    seed_avg_time = []
    neh_solutions = []
    neh_makespans = []
    combinations = list(itertools.combinations(range(100), 2))
    # permutations = list(itertools.permutations(range(100), 2))

    for solution in tqdm.tqdm(solutions):
        start = time()
        # initialization
        solution, makespan = neh_heuristic(processing_time, solution)
        seed_avg_time.append(time() - start)
        neh_solutions.append(solution)
        neh_makespans.append(makespan)

    print(sum(seed_avg_time)/len(seed_avg_time))
    return

    makespans = []
    solutions = []
    # curves = []

    for i in tqdm.tqdm(range(len(neh_solutions))):
        solution = neh_solutions[i]
        makespan = neh_makespans[i]
        start = time()
        best_solution, min_makespan, evaluations = first_move(solution, makespan, processing_time, 1000)
        # while evaluations < 1000:
        #     best_solution, min_makespan, one_step_evaluations = first_move(solution, makespan, processing_time, combinations, inversion)
        #     evaluations += one_step_evaluations
        #     if min_makespan == makespan:
        #         break
        #     solution = best_solution
        #     makespan = min_makespan
        seed_avg_time[i] += time() - start
        # print(i, evaluations)
        solutions.append(best_solution)
        makespans.append(min_makespan)
        # curves.append(curve)
    # print(makespans)

    seed_avg_time = sum(seed_avg_time) / len(seed_avg_time)
    print(seed_avg_time)
    # avg_time /= runs
    # avg_time += init_time

    # best_ind = np.argmin(makespans)
    # min_makespan = makespans[best_ind]
    # best_curve = curves[best_ind]

    # print('avg time', avg_time)
    # print('avg makespan', sum(makespans) / runs)
    # print('min makespan', min_makespan)

    # best_solution = solutions[best_ind]
    # np.save('result.npy', best_solution)

    # plt.figure()
    # plt.plot(best_curve)
    # plt.xlabel('evaluations')
    # plt.ylabel('makespan')
    # plt.show()


if __name__ == '__main__':
    main()
