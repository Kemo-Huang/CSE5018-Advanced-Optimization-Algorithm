#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>

using namespace std;

template<typename T>
vector<int> sort_indexes(const vector<T> &v) {
    // initialize original index locations
    vector<int> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    stable_sort(idx.begin(), idx.end(),
                [&v](int i1, int i2) { return v[i1] > v[i2]; });

    return idx;
}

struct Solution {
    vector<int> idx;
    int weight = 0;
    int value = 0;
};

class Knapsack {
public:
    int n_items;
    int capacity;
    vector<int> weights;
    vector<int> profits;

    Knapsack(int n_items, float capacity) {
        this->n_items = n_items;
        this->capacity = int(capacity);
        weights = vector<int>(n_items, 0);
        profits = vector<int>(n_items, 0);
    }

    void greedy(Solution &solution) {
        vector<float> vow(n_items, 0);
        for (int i = 0; i < n_items; ++i) {
            vow[i] = float(profits[i]) / float(weights[i]);
        }
        vector<int> idx = sort_indexes(vow);
        int tmp;
        for (auto &i: idx) {
            tmp = weights[i] + solution.weight;
            if (tmp <= capacity) {
                solution.weight = tmp;
                solution.value += profits[i];
                solution.idx.push_back(i);
            }
        }
    }

    int dp_solve() {
        vector<vector<int>> matrix(n_items + 1, vector<int>(capacity + 1, 0));
        int evaluations = 0;
        for (int i = 1; i <= n_items; ++i) {
            for (int w = 1; w <= capacity; ++w) {
                if (weights[i - 1] > w) {
                    matrix[i][w] = matrix[i - 1][w];
                } else {
                    evaluations += 1;
                    matrix[i][w] = max(matrix[i - 1][w], profits[i - 1] + matrix[i - 1][w - weights[i - 1]]);
                }
            }
        }
        cout << "best value " << matrix[n_items][capacity] << endl;
        return evaluations;
    }

    int dp_solve_value() {
        const int inf = 10000000;
        const int max_value = 21300;
        int value, evaluations = 0;
        vector<int> matrix(max_value + 1, inf);
        matrix[0] = 0;
        for (int i = 0; i < n_items; ++i) {
            for (int j = max_value; j >= profits[i]; --j) {
                evaluations += 1;
                matrix[j] = min(matrix[j - profits[i]] + weights[i], matrix[j]);
            }
        }
        for (int i = 0; i < max_value; ++i) {
            if (matrix[i] < capacity) {
                value = i;
            }
        }
        cout << "best value " << value << endl;
        return evaluations;
    }

};


void read_knapsack_data(const string &inputFileName, vector<Knapsack> &outputKnapsacks) {
    ifstream fin(inputFileName);
    if (!fin) {
        cout << "Cannot open file " << inputFileName << endl;
        exit(-1);
    }
    string s;
    getline(fin, s);
    int n_items = stoi(s);
    getline(fin, s);
    int n_objs = stoi(s);

    float capacity;
    for (int i = 0; i < n_objs; ++i) {
        getline(fin, s);
        capacity = stof(s);
        Knapsack knapsack(n_items, capacity);
        for (int j = 0; j < n_items; ++j) {
            getline(fin, s);
            knapsack.weights[j] = stoi(s);
            getline(fin, s);
            knapsack.profits[j] = stoi(s);
        }
        outputKnapsacks.push_back(knapsack);
    }
}

int main() {
    vector<Knapsack> knapsacks;
    read_knapsack_data("../knapsack_2_500.txt", knapsacks);
    for (Knapsack &knapsack : knapsacks) {
        int evaluations = knapsack.dp_solve();
        cout << "evaluations " << evaluations << endl;
        Solution solution;
        knapsack.greedy(solution);
        cout << solution.value << endl;
    }
    return 0;
}