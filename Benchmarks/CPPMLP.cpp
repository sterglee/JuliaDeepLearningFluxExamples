
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <numeric>

using namespace std;

// Minimal Matrix-Vector Multiplication for this specific architecture
void train_mlp() {
    const int N = 1000, in = 2, hid = 4, out = 1;
    vector<vector<double>> X(in, vector<double>(N));
    vector<double> y(N);

    // Initialize random data
    for(int i=0; i<N; ++i) {
        X[0][i] = (double)rand()/RAND_MAX;
        X[1][i] = (double)rand()/RAND_MAX;
        y[i] = (X[0][i]*X[0][i] + X[1][i]*X[1][i] > 0.5) ? 1.0 : 0.0;
    }

    // Weights (simplified 1D storage for efficiency)
    vector<double> W1(hid * in, 0.1), b1(hid, 0.0), W2(out * hid, 0.1), b2(out, 0.0);

    auto start = chrono::high_resolution_clock::now();
    int epochs=1000000;
    for(int epoch = 0; epoch < epochs; ++epoch) {
        for(int i = 0; i < N; ++i) {
            // Forward
            vector<double> h(hid);
            for(int j=0; j<hid; ++j) {
                double z = b1[j];
                for(int k=0; k<in; ++k) z += W1[j*in + k] * X[k][i];
                h[j] = z > 0 ? z : 0; // ReLU
            }

            double z2 = b2[0];
            for(int j=0; j<hid; ++j) z2 += W2[j] * h[j];
            double pred = 1.0 / (1.0 + exp(-z2)); // Sigmoid

            // Backprop (Stochastic Gradient Descent)
            double dz2 = pred - y[i];
            for(int j=0; j<hid; ++j) {
                double dW2 = dz2 * h[j];
                double dh = (h[j] > 0) ? (dz2 * W2[j]) : 0;
                W2[j] -= 0.01 * dW2;
                for(int k=0; k<in; ++k) {
                    W1[j*in + k] -= 0.01 * dh * X[k][i];
                }
            }
            b2[0] -= 0.01 * dz2;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    cout << "C++ Training Time: " << diff.count() << "s" << endl;
}

int main() { train_mlp(); return 0; }

