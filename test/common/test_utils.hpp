#pragma once

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include "low_level_linear_algebra.hpp"
// Codes couleurs ANSI pour le terminal
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

    // Sigma = A * A^T
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            float sum = 0.0f;
            for(int k=0; k<n; k++) {
                sum += A[i*n + k] * A[j*n + k];
            }
            Sigma[i*n + j] = sum;
        }
    }


    for(int i=0; i<n; i++) {
        Sigma[i*n + i] += condition_number;
    }
}




struct SyntheticHMMData {
    // Paramètres du modèle
    float* A;              // Matrice de transition N×N
    float* pi;             // Distribution initiale N
    float* mu;             // Moyennes N×K
    float* Sigma;          // Covariances N×K×K
    
    // Données générées
    int* true_states;      // États cachés réels T
    float* observations;   // Observations T×K
    
    int T, K, N;
};

// 1. GÉNÉRATION DE MATRICE DE TRANSITION VALIDE
inline void generate_transition_matrix(float* A, int N, float self_persistence = 0.7f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < N; i++) {
        // Probabilité de rester dans le même état
        A[i * N + i] = self_persistence;
        
        // Distribuer le reste uniformément
        float remaining = 1.0f - self_persistence;
        float sum = 0.0f;
        
        for (int j = 0; j < N; j++) {
            if (j != i) {
                float val = dist(gen);
                A[i * N + j] = val;
                sum += val;
            }
        }
        
        // Normaliser pour que la somme = 1
        for (int j = 0; j < N; j++) {
            if (j != i) {
                A[i * N + j] = (A[i * N + j] / sum) * remaining;
            }
        }
    }
    
    // Vérification : chaque ligne doit sommer à 1
    for (int i = 0; i < N; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row_sum += A[i * N + j];
        }
        if (std::abs(row_sum - 1.0f) > 1e-5) {
            std::cerr << "Erreur: Ligne " << i << " ne somme pas à 1 (" << row_sum << ")" << std::endl;
        }
    }
}

// 2. GÉNÉRATION DE DISTRIBUTION INITIALE
inline void generate_initial_distribution(float* pi, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        pi[i] = dist(gen);
        sum += pi[i];
    }
    
    // Normaliser
    for (int i = 0; i < N; i++) {
        pi[i] /= sum;
    }
}

// 3. GÉNÉRATION DE MOYENNES BIEN SÉPARÉES
inline void generate_separated_means(float* mu, int N, int K, float separation = 3.0f) {
    // Placer les moyennes sur une grille pour maximiser la séparation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.1f);
    
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            // Grille régulière + petit bruit
            mu[i * K + k] = (i * separation) + noise(gen);
        }
    }
}

// 4. GÉNÉRATION DE MATRICES DE COVARIANCE
inline void generate_covariance_matrices(float* Sigma, int N, int K, 
                                          float variance = 1.0f, 
                                          float correlation = 0.3f) {
    for (int i = 0; i < N; i++) {
        float* Sigma_i = Sigma + i * K * K;
        
        // Matrice avec corrélation contrôlée
        for (int r = 0; r < K; r++) {
            for (int c = 0; c < K; c++) {
                if (r == c) {
                    Sigma_i[r * K + c] = variance;
                } else {
                    Sigma_i[r * K + c] = correlation * variance;
                }
            }
        }
        
        // S'assurer qu'elle est définie positive
        for (int r = 0; r < K; r++) {
            Sigma_i[r * K + r] += 0.1f;
        }
    }
}

// 5. GÉNÉRATION DE SÉQUENCE COMPLÈTE (états + observations)
inline SyntheticHMMData generate_synthetic_hmm_sequence(
    int T, int K, int N,
    float self_persistence = 0.8f,
    float mean_separation = 4.0f,
    float variance = 1.0f
) {
    SyntheticHMMData data;
    data.T = T;
    data.K = K;
    data.N = N;
    
    // Allocation
    data.A = new float[N * N];
    data.pi = new float[N];
    data.mu = new float[N * K];
    data.Sigma = new float[N * K * K];
    data.true_states = new int[T];
    data.observations = new float[T * K];
    
    // Génération des paramètres
    generate_transition_matrix(data.A, N, self_persistence);
    generate_initial_distribution(data.pi, N);
    generate_separated_means(data.mu, N, K, mean_separation);
    generate_covariance_matrices(data.Sigma, N, K, variance);
    
    // Génération de la séquence d'états
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> unif(0.0f, 1.0f);
    
    // État initial selon π
    float cumsum = 0.0f;
    float rand_val = unif(gen);
    for (int i = 0; i < N; i++) {
        cumsum += data.pi[i];
        if (rand_val <= cumsum) {
            data.true_states[0] = i;
            break;
        }
    }
    
    // Transition selon A
    for (int t = 1; t < T; t++) {
        int prev_state = data.true_states[t-1];
        cumsum = 0.0f;
        rand_val = unif(gen);
        for (int j = 0; j < N; j++) {
            cumsum += data.A[prev_state * N + j];
            if (rand_val <= cumsum) {
                data.true_states[t] = j;
                break;
            }
        }
    }
    
    // Génération des observations selon N(mu[state], Sigma[state])
    for (int t = 0; t < T; t++) {
        int state = data.true_states[t];
        const float* mu_state = data.mu + state * K;
        const float* Sigma_state = data.Sigma + state * K * K;
        
        // Cholesky pour générer gaussienne multivariée
        std::vector<float> L(K * K);
        choleskyDecomposition(Sigma_state, L.data(), K);
        
        // Génération : y = mu + L * z, où z ~ N(0, I)
        std::normal_distribution<float> standard_normal(0.0f, 1.0f);
        std::vector<float> z(K);
        for (int k = 0; k < K; k++) {
            z[k] = standard_normal(gen);
        }
        
        // Multiplication L * z
        for (int k = 0; k < K; k++) {
            float sum = 0.0f;
            for (int j = 0; j <= k; j++) { // L triangulaire inf
                sum += L[k * K + j] * z[j];
            }
            data.observations[t * K + k] = mu_state[k] + sum;
        }
    }
    
    return data;
}

inline void free_synthetic_data(SyntheticHMMData& data) {
    delete[] data.A;
    delete[] data.pi;
    delete[] data.mu;
    delete[] data.Sigma;
    delete[] data.true_states;
    delete[] data.observations;
}