import numpy as np
import tqdm
import copy


class Solution:
    def __init__(self, n_items):
        self.idx = np.zeros(n_items, dtype=np.bool)
        self.weight_1 = 0
        self.value_1 = 0
        self.weight_2 = 0
        self.value_2 = 0


class TwoKnapsacksProblem:
    def __init__(self, data_path='knapsack_2_500.txt'):
        with open(data_path, 'r') as f:
            lines = f.readlines()
        self.n_items = int(lines[0])
        self.weights_1 = np.zeros(self.n_items, dtype=np.int)
        self.values_1 = np.zeros(self.n_items, dtype=np.int)
        self.weights_2 = np.zeros(self.n_items, dtype=np.int)
        self.values_2 = np.zeros(self.n_items, dtype=np.int)
        self.evaluations = 0

        self.capacity_1 = int(float(lines[2]))
        lines = lines[3:]
        for i in range(self.n_items):
            self.weights_1[i] = int(lines[2 * i])
            self.values_1[i] = int(lines[2 * i + 1])
        self.capacity_2 = int(float(lines[2 * self.n_items]))
        lines = lines[2 * self.n_items + 1:]
        for i in range(self.n_items):
            self.weights_2[i] = int(lines[2 * i])
            self.values_2[i] = int(lines[2 * i + 1])

        self.vow_1 = self.values_1 / self.weights_1
        self.vow_2 = self.values_2 / self.weights_2

    def greedy(self, w1=1.0, w2=1.0):
        vow = w1 * self.vow_1 + w2 * self.vow_2
        idx = np.argsort(-vow)
        solution = Solution(self.n_items)
        for i in idx:
            self.evaluations += 1
            if self.weights_1[i] + solution.weight_1 <= self.capacity_1 \
                    and self.weights_2[i] + solution.weight_2 <= self.capacity_2:
                solution.idx[i] = True
                solution.weight_1 += self.weights_1[i]
                solution.value_1 += self.values_1[i]
                solution.weight_2 += self.weights_2[i]
                solution.value_2 += self.values_2[i]
            if solution.weight_1 == self.capacity_1 or solution.weight_2 == self.capacity_2:
                break
        return solution

    def flip(self, solution, n_offsprings=100):
        chosen_idx = np.arange(self.n_items)[solution.idx]
        not_chosen_idx = np.arange(self.n_items)[~solution.idx]
        paired_indices = np.array([(x, y) for x in chosen_idx for y in not_chosen_idx])
        new_solutions = []
        indices = np.random.choice(len(paired_indices), n_offsprings, replace=False)
        paired_indices = paired_indices[indices]
        self.evaluations += len(paired_indices)
        for idx_pair in paired_indices:
            new_solution_idx = solution.idx.copy()
            new_solution_idx[idx_pair[0]] = False
            new_solution_idx[idx_pair[1]] = True
            # feasible and non-dominated
            new_weight_1 = solution.weight_1 - self.weights_1[idx_pair[0]] + self.weights_1[idx_pair[1]]
            if new_weight_1 < self.capacity_1:
                new_weight_2 = solution.weight_2 - self.weights_2[idx_pair[0]] + self.weights_2[idx_pair[1]]
                if new_weight_2 < self.capacity_2:
                    new_value_1 = solution.value_1 - self.values_1[idx_pair[0]] + self.values_1[idx_pair[1]]
                    new_value_2 = solution.value_2 - self.values_2[idx_pair[0]] + self.values_2[idx_pair[1]]
                    if (solution.value_1 > new_value_1 and solution.value_2 >= new_value_2) or (
                            solution.value_2 > new_value_2 and solution.value_1 >= new_value_1):
                        continue
                    new_solution = Solution(self.n_items)
                    new_solution.idx = new_solution_idx
                    new_solution.weight_1 = new_weight_1
                    new_solution.weight_2 = new_weight_2
                    new_solution.value_1 = new_value_1
                    new_solution.value_2 = new_value_2
                    new_solutions.append(new_solution)
        return new_solutions


def non_dominated_solutions(solutions):
    dominated_idx = []
    for i in range(len(solutions)):
        cur_solution = solutions[i]
        for j in range(i + 1, len(solutions), 1):
            comp = solutions[j]
            if (cur_solution.value_1 < comp.value_1 and cur_solution.value_2 <= comp.value_2) or (
                    cur_solution.value_2 < comp.value_2 and cur_solution.value_1 <= comp.value_1):
                dominated_idx.append(i)
                break
    for i in range(len(solutions) - 1, -1, -1):
        cur_solution = solutions[i]
        for j in range(i - 1, -1, -1):
            comp = solutions[j]
            if (cur_solution.value_1 < comp.value_1 and cur_solution.value_2 <= comp.value_2) or (
                    cur_solution.value_2 < comp.value_2 and cur_solution.value_1 <= comp.value_1) or (
                    cur_solution.value_2 == comp.value_2 and cur_solution.value_1 == comp.value_1):
                dominated_idx.append(i)
                break
    non_dominated_idx = [i for i in range(len(solutions)) if i not in dominated_idx]
    # print(non_dominated_idx)
    return solutions[non_dominated_idx]


def main():
    problem = TwoKnapsacksProblem()
    initial_solutions = []
    for i in range(5):
        solution = problem.greedy(1.0 + 0.1 * i, 1.0)
        # print(solution.value_1, solution.value_2, solution.evaluations)
        initial_solutions.append(solution)
    # print('-----------------')
    for i in range(1, 5, 1):
        solution = problem.greedy(1.0, 1.0 + 0.1 * i)
        # print(solution.value_1, solution.value_2, solution.evaluations)
        initial_solutions.append(solution)
    initial_solutions = np.array(initial_solutions)
    initial_solutions = non_dominated_solutions(initial_solutions)

    evaluations = problem.evaluations
    all_solutions = []
    all_evaluations = []
    for _ in tqdm.tqdm(range(5)):
        solutions = copy.deepcopy(initial_solutions)
        problem.evaluations = evaluations
        while len(solutions) < 100:
            new_solutions = []
            for s in solutions:
                new_solutions += problem.flip(s)
            solutions = np.concatenate([np.array(new_solutions), solutions])
            solutions = non_dominated_solutions(solutions)

        all_solutions.append([(solution.value_1, solution.value_2) for solution in solutions[:100]])
        # print(problem.evaluations)
        all_evaluations.append(problem.evaluations)

    all_area = []
    for i in range(len(all_solutions)):
        solutions = all_solutions[i]
        solutions = sorted(solutions, key=lambda tup: tup[0])
        all_solutions[i] = solutions

        area = 0
        tmp = 0
        for value_1, value_2 in solutions:
            area += (value_1 - tmp) * value_2
            tmp = value_1
        all_area.append(area)

    all_area = np.array(all_area, dtype=np.float64)
    avg_area = np.sum(all_area) / len(all_area)
    print('avg area:', avg_area)

    max_idx = np.argmax(all_area)
    print('max area:', all_area[max_idx])

    all_evaluations = np.array(all_evaluations, dtype=np.float64)
    avg_evaluations = np.sum(all_evaluations) / len(all_evaluations)
    print('avg evaluations:', avg_evaluations)

    best_solutions = all_solutions[max_idx]

    with open('solution.csv', 'w+') as f:
        for value_1, value_2 in best_solutions:
            f.write(str(value_1) + ',' + str(value_2) + '\n')


if __name__ == '__main__':
    main()
