from itertools import chain
from time import time

import numpy as np
import scipy.spatial
from matplotlib import pyplot as plt
from itertools import permutations


class Route:
    def __init__(self, route, fitness):
        self.route = route
        self.length = fitness


class GeneticAlgorithm:
    def __init__(self):
        self.population = []
        self.n_selection = 50
        self.n_inversion = 10
        self.n_insertion = 10
        self.n_local = 100
        self.n_evaluations = 0

        with open('TSP_100Cities.txt') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            cur_x, cur_y = line.split()
            data.append((float(cur_x), float(cur_y)))
        data = np.array(data)

        self.dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data))
        for i in range(len(data)):
            cur_route, cur_route_length = self.greedy(i)
            self.population.append(Route(cur_route, cur_route_length))
        sorted(self.population, key=lambda r: r.length)

    def greedy(self, start_idx):
        solution = [start_idx]
        cur_idx = start_idx
        route_length = 0
        for _ in range(len(self.dist) - 1):
            tmp_dist = self.dist[cur_idx, :].copy()
            tmp_dist[solution] = np.inf
            cur_idx = np.argmin(tmp_dist)
            solution.append(cur_idx)
            route_length += tmp_dist[cur_idx]
        route_length += self.dist[start_idx, cur_idx]
        return solution, route_length

    def compute_length(self, route):
        n = len(route)
        route1 = route + route[0]
        length = 0
        for k in range(n):
            length += self.dist[route1[k], route1[k + 1]]
        self.n_evaluations += 1
        return length

    def crossover(self):
        selected = self.population[:self.n_selection]
        offsprings = []
        for _ in range(self.n_selection // 2):
            r1, r2 = np.random.choice(self.n_selection, 2, replace=False)
            o1, o2 = self.crossover_operator(selected[r1].route, selected[r2].route)
            offsprings.append(Route(o1, self.compute_length(o1)))
            offsprings.append(Route(o2, self.compute_length(o2)))
        self.population = selected + offsprings

    def mutation(self):
        mutation_indices = np.random.choice(len(self.population), self.n_insertion + self.n_inversion, replace=False)
        for i in mutation_indices[:self.n_insertion]:
            r = self.population[i]
            i1, i2 = np.random.choice(len(r.route), 2, replace=False)
            r.route = self.insertion(r.route, i1, i2)
            r.length = self.compute_length(r.route)
        for i in mutation_indices[:self.n_insertion]:
            r = self.population[i]
            i1, i2 = np.random.choice(len(r.route), 2, replace=False)
            r.route = self.inversion(r.route, i1, i2)
            r.length = self.compute_length(r.route)

    def local_search(self):
        sorted(self.population, key=lambda x: x.length)
        selected = self.population[:self.n_local]
        indices = permutations(np.arange(100), 2)
        best_routes = []
        for r in selected:
            min_length = r.length
            best_route = r.route
            for i in indices:
                insert_route = self.insertion(r.route, i[0], i[1])
                insert_length = self.compute_length(insert_route)
                invert_route = self.inversion(r.route, i[0], i[1])
                invert_length = self.compute_length(invert_route)
                if insert_length < min_length:
                    best_route = insert_route
                    min_length = insert_length
                if invert_length < min_length:
                    best_route = invert_route
                    min_length = invert_length
            best_routes.append(Route(best_route, min_length))
        self.population = sorted(best_routes, key=lambda x: x.length) + self.population[self.n_local:]
        return self.population[0]

    @staticmethod
    def crossover_operator(p1, p2):
        i1, i2 = np.random.choice(len(p1), 2, replace=False)
        g1 = p1[i1:i2 + 1]
        g2 = p2[i1:i2 + 1]
        o1 = p1.copy()
        o2 = p2.copy()
        o1[i1:i2 + 1] = g2
        o2[i1:i2 + 1] = g1
        unique_g1 = [i for i in g1 if i not in g2]
        unique_g2 = [i for i in g2 if i not in g1]
        ptr1 = 0
        for i in chain(range(i1), range(i2 + 1, len(p1))):
            if o1[i] in g2:
                o1[i] = unique_g1[ptr1]
                ptr1 += 1
        ptr2 = 0
        for i in chain(range(i1), range(i2 + 1, len(p1))):
            if o2[i] in g1:
                o2[i] = unique_g2[ptr2]
                ptr2 += 1
        return o1, o2

    @staticmethod
    def inversion(route1, i1, i2):
        min_i = min(i1, i2)
        max_i = max(i1, i2)
        route2 = route1.copy()
        route2[min_i:max_i] = reversed(route1[min_i:max_i])
        return route2

    @staticmethod
    def insertion(route1, i1, i2):
        route2 = route1.copy()
        if i1 < i2:
            route2[i1:i2] = route1[i1 + 1:i2 + 1]
            route2[i2] = route1[i1]
        else:
            route2[i2 + 1:i1 + 1] = route1[i2:i1]
            route2[i2] = route1[i1]
        return route2

    def run(self):
        num_gen = 0
        curve = []
        best_route = None
        best_length = np.inf
        start = time()
        while self.n_evaluations < 2e6:
            self.crossover()
            self.mutation()
            cur_best_route = self.local_search()
            curve.append(cur_best_route.length)
            if cur_best_route.length < best_length:
                best_length = cur_best_route.length
                best_route = cur_best_route
            num_gen += 1
            print('generation', num_gen, 'cur best length', best_route.length)
        print('time cost', time() - start)
        print('num of generations', num_gen)
        print('num of evaluations', self.n_evaluations)
        print('best length', best_route.length)

        plt.figure()
        plt.plot(curve)
        plt.ylabel('shortest route length')
        plt.xlabel('generations')

        route = best_route.route
        route.append(route[0])
        route = np.array(route)

        plt.figure()
        plt.plot(route[:, 0], route[:, 1], 'k-o', linewidth=1.5, markerfacecolor='white',
                 markersize=10)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


def main():
    ga = GeneticAlgorithm()
    ga.run()


if __name__ == '__main__':
    main()
