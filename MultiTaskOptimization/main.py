import numpy as np


class Solution:
    def __init__(self):
        self.idx = []
        self.weight_1 = 0
        self.value_1 = 0
        self.weight_2 = 0
        self.value_2 = 0
        self.evaluations = 0


class TwoKnapsacksProblem:
    def __init__(self, data_path='knapsack_2_500.txt'):
        with open(data_path, 'r') as f:
            lines = f.readlines()
        self.n_items = int(lines[0])
        self.weights_1 = np.zeros(self.n_items, dtype=np.int)
        self.values_1 = np.zeros(self.n_items, dtype=np.int)
        self.weights_2 = np.zeros(self.n_items, dtype=np.int)
        self.values_2 = np.zeros(self.n_items, dtype=np.int)

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
        solution = Solution()
        for i in idx:
            solution.evaluations += 1
            if self.weights_1[i] + solution.weight_1 <= self.capacity_1 \
                    and self.weights_2[i] + solution.weight_2 <= self.capacity_2:
                solution.idx.append(i)
                solution.weight_1 += self.weights_1[i]
                solution.value_1 += self.values_1[i]
                solution.weight_2 += self.weights_2[i]
                solution.value_2 += self.values_2[i]
            if solution.weight_1 == self.capacity_1 or solution.weight_2 == self.capacity_2:
                break
        return solution


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
    print(non_dominated_idx)
    return solutions[non_dominated_idx]


def main():
    problem = TwoKnapsacksProblem()
    solutions = []
    for i in range(10):
        solution = problem.greedy(1.0 + 0.1 * i, 1.0)
        # print(solution.value_1, solution.value_2, solution.evaluations)
        solutions.append(solution)
    # print('-----------------')
    for i in range(1, 10, 1):
        solution = problem.greedy(1.0, 1.0 + 0.1 * i)
        # print(solution.value_1, solution.value_2, solution.evaluations)
        solutions.append(solution)
    solutions.append(problem.greedy(sum(problem.vow_1) / problem.n_items, sum(problem.vow_2) / problem.n_items))
    solutions = np.array(solutions)
    solutions = non_dominated_solutions(solutions)
    print([(i.value_1, i.value_2) for i in solutions])


if __name__ == '__main__':
    main()
