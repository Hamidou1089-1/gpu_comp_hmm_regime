#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include "linalg_cpu.hpp"
#include "algo_hmm_cpu.hpp"

using namespace hmm::cpu;

// Colors
#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

static int g_tests_failed = 0;

#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cout << RED << "[ECHEC] " << message << RESET << " (Ligne " << __LINE__ << ")" << std::endl; \
            g_tests_failed++; \
        } else { \
            std::cout << GREEN << "[REUSSI] " << message << RESET << std::endl; \
        } \
    } while (0)

#define TEST_ASSERT_FLOAT_EQ(val1, val2, epsilon, message) \
    do { \
        if (std::abs((val1) - (val2)) > (epsilon)) { \
            std::cout << RED << "[ECHEC] " << message << " : Attendu " << (val1) << ", Obtenu " << (val2) << RESET << std::endl; \
            g_tests_failed++; \
        } else { \
            std::cout << GREEN << "[REUSSI] " << message << RESET << std::endl; \
        } \
    } while (0)

inline int print_test_summary() {
    if (g_tests_failed == 0) {
        std::cout << "\n" << GREEN << ">>> TOUS LES TESTS SONT PASSES ! <<<" << RESET << "\n";
        return 0;
    } else {
        std::cout << "\n" << RED << ">>> " << g_tests_failed << " TEST(S) ECHOUE(S) <<<" << RESET << "\n";
        return 1;
    }
}

inline void print_matrix_limited(const float* M, int rows, int cols, int limit = 15) {
    std::cout << "Matrice (" << rows << "x" << cols << ") :" << std::endl;
    int r_lim = std::min(rows, limit);
    int c_lim = std::min(cols, limit);
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < r_lim; i++) {
        std::cout << "[ ";
        for (int j = 0; j < c_lim; j++) {
            std::cout << M[i * cols + j] << "\t";
        }
        if (cols > limit) std::cout << "...";
        std::cout << " ]" << std::endl;
    }
    if (rows > limit) std::cout << "... (lignes masquees) ..." << std::endl;
    std::cout << std::defaultfloat << std::endl;
}

inline void generate_random_pd_matrix(float* Sigma, int n, float condition_number = 10.0f) {
    std::vector<float> A(n * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for(int i=0; i<n*n; i++) A[i] = dist(gen);
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            float sum = 0.0f;
            for(int k=0; k<n; k++) sum += A[i*n + k] * A[j*n + k];
            Sigma[i*n + j] = sum;
        }
    }
    for(int i=0; i<n; i++) Sigma[i*n + i] += condition_number;
}

struct SyntheticHMMData {
    algo::HMMModel model;
    int* true_states;
    float* observations;
};

inline void generate_transition_matrix(float* A, int N, float self_persistence = 0.6f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; i++) {
        A[i * N + i] = self_persistence;
        float remaining = 1.0f - self_persistence;
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            if (j != i) {
                float val = dist(gen);
                A[i * N + j] = val;
                sum += val;
            }
        }
        for (int j = 0; j < N; j++) {
            if (j != i) A[i * N + j] = (A[i * N + j] / sum) * remaining;
        }
    }
}

inline void generate_initial_distribution(float* pi, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        pi[i] = dist(gen);
        sum += pi[i];
    }
    for (int i = 0; i < N; i++) pi[i] /= sum;
}

inline void generate_separated_means(float* mu, int N, int K, float separation = 5.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.1f);
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            mu[i * K + k] = (i * separation) + noise(gen);
        }
    }
}

inline void generate_covariance_matrices(float* Sigma, int N, int K, 
                                          float variance = 2.0f, 
                                          float correlation = 0.1f) {
    for (int i = 0; i < N; i++) {
        float* Sigma_i = Sigma + i * K * K;
        for (int r = 0; r < K; r++) {
            for (int c = 0; c < K; c++) {
                if (r == c) {
                    Sigma_i[r * K + c] = variance;
                } else {
                    Sigma_i[r * K + c] = correlation * variance;
                }
            }
        }
        for (int r = 0; r < K; r++) Sigma_i[r * K + r] += 0.1f;
    }
}

inline SyntheticHMMData generate_synthetic_hmm_sequence(
    int T, int K, int N,
    float self_persistence = 0.8f,
    float mean_separation = 2.0f,
    float variance = 2.0f
) {
    SyntheticHMMData data;
    data.model.T = T;
    data.model.K = K;
    data.model.N = N;
    data.model.pi = new float[N];
    data.model.A = new float[N * N];
    data.model.mu = new float[N * K];
    data.model.Sigma = new float[N * K * K];
    data.model.L = new float[N * K * K];
    data.model.log_det = new float[N];
    data.true_states = new int[T];
    data.observations = new float[T * K];
    
    generate_transition_matrix(data.model.A, N, self_persistence);
    generate_initial_distribution(data.model.pi, N);
    generate_separated_means(data.model.mu, N, K, mean_separation);
    generate_covariance_matrices(data.model.Sigma, N, K, variance);
    algo::precompute_cholesky(data.model);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> unif(0.0f, 1.0f);
    
    float cumsum = 0.0f;
    float rand_val = unif(gen);
    for (int i = 0; i < N; i++) {
        cumsum += data.model.pi[i];
        if (rand_val <= cumsum) {
            data.true_states[0] = i;
            break;
        }
    }
    
    for (int t = 1; t < T; t++) {
        int prev_state = data.true_states[t-1];
        cumsum = 0.0f;
        rand_val = unif(gen);
        for (int j = 0; j < N; j++) {
            cumsum += data.model.A[prev_state * N + j];
            if (rand_val <= cumsum) {
                data.true_states[t] = j;
                break;
            }
        }
    }
    
    for (int t = 0; t < T; t++) {
        int state = data.true_states[t];
        const float* mu_state = data.model.mu + state * K;
        const float* L_state = data.model.L + state * K * K;
        
        std::normal_distribution<float> standard_normal(0.0f, 1.0f);
        std::vector<float> z(K);
        for (int k = 0; k < K; k++) z[k] = standard_normal(gen);
        
        for (int k = 0; k < K; k++) {
            float sum = 0.0f;
            for (int j = 0; j <= k; j++) sum += L_state[k * K + j] * z[j];
            data.observations[t * K + k] = mu_state[k] + sum;
        }
    }
    
    return data;
}

inline void free_synthetic_data(SyntheticHMMData& data) {
    delete[] data.model.pi;
    delete[] data.model.A;
    delete[] data.model.mu;
    delete[] data.model.Sigma;
    delete[] data.model.L;
    delete[] data.model.log_det;
    delete[] data.true_states;
    delete[] data.observations;
}