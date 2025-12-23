// test_real_data_cpu.cpp
#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>
#include <limits>
#include "test_utils.hpp"
#include "data_loader.hpp"
#include "algo_hmm.hpp"
#include "low_level_linear_algebra.hpp"


void test_load_real_data() {
    std::cout << "\n--- Test Load Real Financial Data ---\n";
    
    ObservationData data;
    bool success = load_observations(
        data,
        "../../data/processed/observations.bin",
        "../../data/processed/observations_info.json"
    );
    
    TEST_ASSERT(success, "Load observations");
    TEST_ASSERT(data.T > 0, "T > 0");
    TEST_ASSERT(data.K > 0, "K > 0");
    
    std::cout << "Dataset: T=" << data.T << ", K=" << data.K << std::endl;
    
    // Print first few observations
    std::cout << "First 5 observations (first 3 assets):\n";
    for (int t = 0; t < 5 && t < data.T; t++) {
        std::cout << "  t=" << t << ": ";
        for (int k = 0; k < 3 && k < data.K; k++) {
            std::cout << get_observation(data, t, k) << " ";
        }
        std::cout << "\n";
    }
    
    
}


// ============================================================================
// SAUVEGARDE DES PARAMÈTRES
// ============================================================================

void save_parameters(
    const char* filename,
    const float* A, const float* pi, const float* mu, const float* Sigma,
    int N, int K
) {
    std::ofstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Erreur : impossible d'ouvrir " << filename << std::endl;
        return;
    }
    
    // Écrire les dimensions
    file.write(reinterpret_cast<const char*>(&N), sizeof(int));
    file.write(reinterpret_cast<const char*>(&K), sizeof(int));
    
    // Écrire A (N×N)
    file.write(reinterpret_cast<const char*>(A), N * N * sizeof(float));
    
    // Écrire π (N)
    file.write(reinterpret_cast<const char*>(pi), N * sizeof(float));
    
    // Écrire μ (N×K)
    file.write(reinterpret_cast<const char*>(mu), N * K * sizeof(float));
    
    // Écrire Σ (N×K×K)
    file.write(reinterpret_cast<const char*>(Sigma), N * K * K * sizeof(float));
    
    file.close();
    std::cout << "Paramètres sauvegardés dans " << filename << std::endl;
}

void save_training_history(
    const char* filename,
    const float* log_likelihoods,
    int num_iterations
) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Erreur : impossible d'ouvrir " << filename << std::endl;
        return;
    }
    
    file << "iteration,log_likelihood\n";
    for (int i = 0; i < num_iterations; i++) {
        file << i << "," << log_likelihoods[i] << "\n";
    }
    
    file.close();
    std::cout << "Historique sauvegardé dans " << filename << std::endl;
}

// ============================================================================
// INITIALISATION DES PARAMÈTRES
// ============================================================================

void initialize_parameters_kmeans_style(
    float* mu, float* Sigma, float* A, float* pi,
    const float* observations,
    int T, int K, int N
) {
    std::cout << "Initialisation des paramètres...\n";
    
    // 1. Initialiser μ : diviser les observations en N clusters grossiers
    int chunk_size = T / N;
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            float sum = 0.0f;
            int count = 0;
            
            // Moyenne des observations dans le chunk i
            int start = i * chunk_size;
            int end = (i + 1) * chunk_size;
            if (end > T) end = T;
            
            for (int t = start; t < end; t++) {
                sum += observations[t * K + k];
                count++;
            }
            
            mu[i * K + k] = (count > 0) ? (sum / count) : 0.0f;
        }
    }
    
    // 2. Initialiser Σ : covariance empirique globale
    float* global_cov = new float[K * K];
    memset(global_cov, 0, K * K * sizeof(float));
    
    // Moyenne globale
    float* global_mean = new float[K];
    memset(global_mean, 0, K * sizeof(float));
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            global_mean[k] += observations[t * K + k];
        }
    }
    for (int k = 0; k < K; k++) {
        global_mean[k] /= T;
    }
    
    // Covariance globale
    for (int t = 0; t < T; t++) {
        for (int r = 0; r < K; r++) {
            for (int c = 0; c < K; c++) {
                float diff_r = observations[t * K + r] - global_mean[r];
                float diff_c = observations[t * K + c] - global_mean[c];
                global_cov[r * K + c] += diff_r * diff_c;
            }
        }
    }
    for (int r = 0; r < K; r++) {
        for (int c = 0; c < K; c++) {
            global_cov[r * K + c] /= T;
        }
    }
    
    // Régularisation
    for (int k = 0; k < K; k++) {
        global_cov[k * K + k] += 0.01f;
    }
    
    // Copier pour chaque état
    for (int i = 0; i < N; i++) {
        memcpy(Sigma + i * K * K, global_cov, K * K * sizeof(float));
    }
    
    delete[] global_cov;
    delete[] global_mean;
    
    // 3. Initialiser A : matrice de transition avec forte persistance
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                A[i * N + j] = 0.7f;  // Probabilité de rester
            } else {
                A[i * N + j] = 0.3f / (N - 1);  // Distribué uniformément
            }
        }
    }
    
    // 4. Initialiser π : uniforme
    for (int i = 0; i < N; i++) {
        pi[i] = 1.0f / N;
    }
    
    std::cout << "Initialisation terminée.\n";
}

// ============================================================================
// TEST ENTRAÎNEMENT
// ============================================================================


void decode_states_and_save(
    const float* A, const float* pi, const float* mu, const float* Sigma,
    const float* observations,
    int T, int K, int N
) {
    // Calculer potentiels
    GaussianParams params;
    precomputeGaussianParams(params, mu, Sigma, N, K);
    
    float* log_potentials;
    compute_log_gaussian_potentials(log_potentials, A, params, 
                                    pi, observations, T, K, N);
    
    // Viterbi
    int* best_path = new int[T];
    float log_prob;
    viterbi_log(best_path, &log_prob, log_potentials, T, N);
    
    // Sauvegarder la séquence d'états
    std::ofstream file("../../data/decoded_states.csv");
    file << "timestep,state\n";
    for (int t = 0; t < T; t++) {
        file << t << "," << best_path[t] << "\n";
    }
    file.close();
    
    std::cout << "États décodés sauvegardés dans decoded_states.csv\n";
    
    // Statistiques
    int* state_counts = new int[N]();
    for (int t = 0; t < T; t++) {
        state_counts[best_path[t]]++;
    }
    
    std::cout << "Distribution des états:\n";
    for (int i = 0; i < N; i++) {
        float pct = 100.0f * state_counts[i] / T;
        std::cout << "  État " << i << ": " << state_counts[i] 
                  << " timesteps (" << pct << "%)\n";
    }
    
    delete[] best_path;
    delete[] state_counts;
    freePotentials(log_potentials);
    freeGaussianParams(params);
}



void test_em_training_real_data() {
    std::cout << "\n=======================================================\n";
    std::cout << "   TRAINING HMM ON REAL FINANCIAL DATA\n";
    std::cout << "=======================================================\n\n";
    
    // Load data
    ObservationData data;
    bool success = load_observations(
        data,
        "../../data/processed/observations.bin",
        "../../data/processed/observations_info.json"
    );
    
    if (!success) {
        std::cerr << "Erreur : impossible de charger les données\n";
        return;
    }
    
    int N = 3;  // 3 régimes : bull, bear, neutral
    int T = data.T;
    int K = data.K;
    
    std::cout << "Dataset: T=" << T << ", K=" << K << ", N=" << N << "\n\n";
    
    // Allouer les paramètres
    float* A = new float[N * N];
    float* pi = new float[N];
    float* mu = new float[N * K];
    float* Sigma = new float[N * K * K];
    
    // Initialiser
    initialize_parameters_kmeans_style(mu, Sigma, A, pi, data.observations, T, K, N);
    
    // Allouer l'historique
    int max_iterations = 150;
    float* log_likelihood_history = new float[max_iterations];
    int num_iterations = 0;
    
    // Entraînement EM
    auto start = std::chrono::high_resolution_clock::now();
    
    em_algorithm_with_history(A, pi, mu, Sigma, data.observations, T, K, N,
                 log_likelihood_history, &num_iterations,
                 max_iterations, 1e-6f);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "\nTemps d'entraînement : " << elapsed.count() << " secondes\n";
    std::cout << "Nombre d'itérations : " << num_iterations << "\n";
    
    // Sauvegarder les paramètres
    save_parameters("../../data/trained_params.bin", A, pi, mu, Sigma, N, K);
    save_training_history("../../data/training_history.csv", 
                          log_likelihood_history, num_iterations);
    
    // Afficher les paramètres finaux
    std::cout << "\n--- Paramètres finaux ---\n";
    std::cout << "Distribution initiale π :\n";
    for (int i = 0; i < N; i++) {
        std::cout << "  π[" << i << "] = " << pi[i] << "\n";
    }
    
    std::cout << "\nMatrice de transition A :\n";
    for (int i = 0; i < N; i++) {
        std::cout << "  ";
        for (int j = 0; j < N; j++) {
            std::cout << A[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\nMoyennes μ (3 premiers assets) :\n";
    for (int i = 0; i < N; i++) {
        std::cout << "  État " << i << ": ";
        for (int k = 0; k < 3 && k < K; k++) {
            std::cout << mu[i * K + k] << " ";
        }
        std::cout << "\n";
    }

    // Appeler dans test_em_training_real_data() après l'entraînement
    decode_states_and_save(A, pi, mu, Sigma, data.observations, T, K, N);
    
    // Cleanup
    delete[] log_likelihood_history;
    delete[] A;
    delete[] pi;
    delete[] mu;
    delete[] Sigma;
    
}






// ============================================================================
// MAIN
// ============================================================================

int main() {
    test_load_real_data();
    test_em_training_real_data();
    return print_test_summary();
}