#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>
#include <limits>
#include "test_utils.hpp"
#include "data_loader.hpp"
#include "algo_hmm_cpu.hpp"
#include "linalg_cpu.hpp"

using namespace hmm::cpu;

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
    std::cout << "First 5 observations (first 3 assets):\n";
    for (int t = 0; t < 5 && t < data.T; t++) {
        std::cout << "  t=" << t << ": ";
        for (int k = 0; k < 3 && k < data.K; k++) {
            std::cout << get_observation(data, t, k) << " ";
        }
        std::cout << "\n";
    }
}

void save_parameters(const char* filename, const algo::HMMModel& model) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return;
    
    file.write(reinterpret_cast<const char*>(&model.N), sizeof(int));
    file.write(reinterpret_cast<const char*>(&model.K), sizeof(int));
    file.write(reinterpret_cast<const char*>(model.A), model.N * model.N * sizeof(float));
    file.write(reinterpret_cast<const char*>(model.pi), model.N * sizeof(float));
    file.write(reinterpret_cast<const char*>(model.mu), model.N * model.K * sizeof(float));
    file.write(reinterpret_cast<const char*>(model.Sigma), model.N * model.K * model.K * sizeof(float));
    
    file.close();
    std::cout << "Paramètres sauvegardés dans " << filename << std::endl;
}

void save_training_history(const char* filename, const float* log_likelihoods, int num_iterations) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    file << "iteration,log_likelihood\n";
    for (int i = 0; i < num_iterations; i++) {
        file << i << "," << log_likelihoods[i] << "\n";
    }
    file.close();
    std::cout << "Historique sauvegardé dans " << filename << std::endl;
}

void initialize_hmm_model(algo::HMMModel& model, const float* observations, int T, int K, int N) {
    std::cout << "Initialisation du modèle HMM...\n";
    
    model.T = T;
    model.K = K;
    model.N = N;
    model.pi = new float[N];
    model.A = new float[N * N];
    model.mu = new float[N * K];
    model.Sigma = new float[N * K * K];
    model.L = new float[N * K * K];
    model.log_det = new float[N];
    
    // μ: moyennes par chunks temporels
    int chunk_size = T / N;
    for (int i = 0; i < N; i++) {
        int start = i * chunk_size;
        int end = (i == N-1) ? T : (i+1) * chunk_size;
        for (int k = 0; k < K; k++) {
            float sum = 0.0f;
            for (int t = start; t < end; t++) sum += observations[t * K + k];
            model.mu[i * K + k] = sum / (end - start);
        }
    }
    
    // Σ: covariance globale + régularisation
    std::vector<float> global_mean(K, 0.0f);
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) global_mean[k] += observations[t * K + k];
    }
    for (int k = 0; k < K; k++) global_mean[k] /= T;
    
    for (int i = 0; i < N; i++) {
        float* Sigma_i = model.Sigma + i * K * K;
        std::memset(Sigma_i, 0, K * K * sizeof(float));
        
        for (int t = 0; t < T; t++) {
            for (int k1 = 0; k1 < K; k1++) {
                float d1 = observations[t * K + k1] - global_mean[k1];
                for (int k2 = 0; k2 < K; k2++) {
                    float d2 = observations[t * K + k2] - global_mean[k2];
                    Sigma_i[k1 * K + k2] += d1 * d2;
                }
            }
        }
        
        for (int k = 0; k < K * K; k++) Sigma_i[k] /= T;
        for (int k = 0; k < K; k++) Sigma_i[k * K + k] += 0.01f;
    }
    
    // A: forte persistance
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            model.A[i * N + j] = (i == j) ? 0.7f : 0.3f / (N - 1);
        }
    }
    
    // π: uniforme
    for (int i = 0; i < N; i++) model.pi[i] = 1.0f / N;
    
    algo::precompute_cholesky(model);
    std::cout << "Initialisation terminée.\n";
}

void decode_states_and_save(const algo::HMMModel& model, const float* observations) {
    int pot_size = model.N + (model.T - 1) * model.N * model.N;
    std::vector<float> log_potentials(pot_size);
    std::vector<float> workspace(2 * model.K);
    
    algo::compute_log_gaussian_potentials(model, observations, log_potentials.data(), workspace.data());
    
    std::vector<int> path(model.T);
    float log_prob = algo::viterbi_algorithm(log_potentials.data(), path.data(), model.T, model.N);
    
    std::ofstream file("../../data/decoded_states.csv");
    file << "timestep,state\n";
    for (int t = 0; t < model.T; t++) file << t << "," << path[t] << "\n";
    file.close();
    
    std::cout << "États décodés sauvegardés (log_prob=" << log_prob << ")\n";
    
    std::vector<int> counts(model.N, 0);
    for (int t = 0; t < model.T; t++) counts[path[t]]++;
    
    std::cout << "Distribution des états:\n";
    for (int i = 0; i < model.N; i++) {
        std::cout << "  État " << i << ": " << counts[i] 
                  << " (" << 100.0f * counts[i] / model.T << "%)\n";
    }
}

void test_em_training_real_data() {
    std::cout << "\n=======================================================\n";
    std::cout << "   TRAINING HMM ON REAL FINANCIAL DATA\n";
    std::cout << "=======================================================\n\n";
    
    ObservationData data;
    if (!load_observations(data, "../../data/processed/observations.bin",
                          "../../data/processed/observations_info.json")) {
        std::cerr << "Erreur: impossible de charger les données\n";
        return;
    }
    
    const int N = 3;
    std::cout << "Dataset: T=" << data.T << ", K=" << data.K << ", N=" << N << "\n\n";
    
    algo::HMMModel model;
    initialize_hmm_model(model, data.observations, data.T, data.K, N);
    
    const int max_iterations = 150;
    std::vector<float> ll_history(max_iterations);
    std::vector<float> workspace(2 * data.K);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    float prev_ll = -std::numeric_limits<float>::infinity();
    int num_iterations = 0;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        std::vector<float> gamma(data.T * N);
        std::vector<float> xi((data.T - 1) * N * N);
        
        float ll = algo::baum_welch_e_step(model, data.observations, gamma.data(), xi.data(), workspace.data());
        ll_history[iter] = ll;
        num_iterations = iter + 1;
        
        std::cout << "Iter " << iter << ": log-likelihood = " << ll << "\n";
        
        if (iter > 0 && ll - prev_ll < 1e-6f) break;
        
        algo::baum_welch_m_step(model, data.observations, gamma.data(), xi.data());
        prev_ll = ll;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "\nTemps: " << elapsed.count() << "s, Itérations: " << num_iterations << "\n";
    
    save_parameters("../../data/trained_params.bin", model);
    save_training_history("../../data/training_history.csv", ll_history.data(), num_iterations);
    
    std::cout << "\nDistribution initiale π:\n";
    for (int i = 0; i < N; i++) std::cout << "  π[" << i << "] = " << model.pi[i] << "\n";
    
    std::cout << "\nMatrice de transition A:\n";
    for (int i = 0; i < N; i++) {
        std::cout << "  ";
        for (int j = 0; j < N; j++) std::cout << model.A[i * N + j] << " ";
        std::cout << "\n";
    }
    
    decode_states_and_save(model, data.observations);
    
    delete[] model.pi;
    delete[] model.A;
    delete[] model.mu;
    delete[] model.Sigma;
    delete[] model.L;
    delete[] model.log_det;
}

int main() {
    test_load_real_data();
    test_em_training_real_data();
    return print_test_summary();
}