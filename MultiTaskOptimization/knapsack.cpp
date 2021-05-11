#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

class Knapsack {
public:
    float capacity;
    vector<int> weights;
    vector<int> profits;

    Knapsack(int n_items, float capacity) {
        this->capacity = capacity;
        weights = vector<int>(n_items, 0);
        profits = vector<int>(n_items, 0);
    }

    void solve(){

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

    return 0;
}