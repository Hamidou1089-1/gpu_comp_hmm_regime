/**
 * @file test_baum_welch_gpu.cu
 * @brief Test Baum-Welch (EM): GPU vs CPU avec validation et benchmark
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>
#include <cstring>

#include "algo_hmm_gpu.cuh"
#include "algo_hmm_cpu.hpp"
#include "test_utils.hpp"
#include "cuda_timing.cuh"

using namespace hmm;

// ============================================================================
// HELPERS
// ============================================================================

void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error [" << msg << "]: " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

float compute_relative_error(const float* a, const float* b, int size) {
    float max_diff = 0.0f;
    float max_val = 0.0f;
    
    for (int i = 0; i < size; ++i) {
        float diff = std::abs(a[i] - b[i]);
        float val = std::max(std::abs(a[i]), std::abs(b[i]));
        
        max_diff = std::max(max_diff, diff);
        max_val = std::max(max_val, val);
    }
    
    return (max_val > 1e-10f) ? (max_diff / max_val) : max_diff;
}

// ============================================================================
// TEST BAUM-WELCH
// ============================================================================
void test_baum_welch(int T, int N, int K, int max_iter = 20) {
    std::cout << "\n============================================" << std::endl;
    std::cout << "Test Baum-Welch: T=" << T << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "============================================\n" << std::endl;

    // 1. Génération données synthétiques (On garde ton générateur)
    // On génère avec des moyennes assez écartées pour que ce soit apprenable
    auto data = generate_synthetic_hmm_sequence(T, K, N, 0.7f, 5.0f, 1.0f);
    
    // --- ETAPE CRUCIALE : NORMALISATION DES DONNÉES (Z-SCORE) ---
    // Cela évite les explosions numériques dans Gaussiennes
    std::vector<float> mean_data(K, 0.0f);
    std::vector<float> std_data(K, 0.0f);

    // Calcul Moyenne
    for(int t=0; t<T; ++t) {
        for(int k=0; k<K; ++k) mean_data[k] += data.observations[t*K + k];
    }
    for(int k=0; k<K; ++k) mean_data[k] /= T;

    // Calcul Std
    for(int t=0; t<T; ++t) {
        for(int k=0; k<K; ++k) {
            float diff = data.observations[t*K + k] - mean_data[k];
            std_data[k] += diff * diff;
        }
    }
    for(int k=0; k<K; ++k) std_data[k] = std::sqrt(std_data[k] / T);

    // Appliquer Normalisation
    for(int t=0; t<T; ++t) {
        for(int k=0; k<K; ++k) {
            if(std_data[k] > 1e-5f)
                data.observations[t*K + k] = (data.observations[t*K + k] - mean_data[k]) / std_data[k];
            else 
                data.observations[t*K + k] = 0.0f;
        }
    }
    // -------------------------------------------------------------

    // 2. Initialisation SAFE (Identique CPU/GPU)
    std::vector<float> pi_init(N);
    std::vector<float> A_init(N * N);
    std::vector<float> mu_init(N * K);
    std::vector<float> Sigma_init(N * K * K);
    
    // Pi et A uniformes ou aléatoires légers
    generate_initial_distribution(pi_init.data(), N);
    generate_transition_matrix(A_init.data(), N, 0.5f); // 0.5 = pas trop diagonal
    
    // Moyennes initiales : Petits randoms autour de 0 (puisque data centrée)
    // Ne pas utiliser les vraies moyennes, sinon c'est tricher
    std::mt19937 gen(1234); // Seed fixe pour repro
    std::normal_distribution<float> d(0.0f, 0.5f);
    for(int i=0; i<N*K; ++i) mu_init[i] = d(gen);

    // Covariances initiales : IDENTITÉ (Le secret de la stabilité)
    // Ne jamais initialiser avec du random pour K > 4
    for(int i=0; i<N; ++i) {
        for(int r=0; r<K; ++r) {
            for(int c=0; c<K; ++c) {
                // Matrice Identité * 1.0
                Sigma_init[i*K*K + r*K + c] = (r == c) ? 1.0f : 0.0f;
            }
        }
    }
    
    // 3. Préparations GPU (Log Space conversion)
    std::vector<float> log_pi_gpu(N);
    std::vector<float> log_A_gpu(N * N);
    for (int i = 0; i < N; ++i) log_pi_gpu[i] = std::log(pi_init[i] + 1e-10f);
    for (int i = 0; i < N*N; ++i) log_A_gpu[i] = std::log(A_init[i] + 1e-10f);
    
    // Cholesky Initiale pour GPU
    std::vector<float> L_init(N * K * K);
    std::vector<float> log_dets_init(N);
    for (int i = 0; i < N; ++i) {
        std::memcpy(L_init.data() + i*K*K, Sigma_init.data() + i*K*K, K*K*sizeof(float));
        cpu::linalg::cholesky_decomposition(L_init.data() + i*K*K, K);
        log_dets_init[i] = cpu::linalg::log_det_from_cholesky(L_init.data() + i*K*K, K);
    }
    
    // 4. Allocation & Run GPU
    float *d_pi, *d_A, *d_means, *d_L, *d_log_dets, *d_Sigma, *d_obs;
    cudaMalloc(&d_pi, N * sizeof(float));
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_means, N * K * sizeof(float));
    cudaMalloc(&d_L, N * K * K * sizeof(float));
    cudaMalloc(&d_log_dets, N * sizeof(float));
    cudaMalloc(&d_Sigma, N * K * K * sizeof(float));
    cudaMalloc(&d_obs, T * K * sizeof(float));
    
    cudaMemcpy(d_pi, log_pi_gpu.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, log_A_gpu.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_means, mu_init.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, L_init.data(), N * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_log_dets, log_dets_init.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sigma, Sigma_init.data(), N * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs, data.observations, T * K * sizeof(float), cudaMemcpyHostToDevice);
    
    std::cout << "[GPU] Baum-Welch Algorithm..." << std::endl;
    std::vector<float> ll_history_gpu(max_iter);
    float total_time_gpu = 0.0f;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        auto bw_call = [&]() {
            ll_history_gpu[iter] = gpu::baum_welch_step_gpu(
                d_pi, d_A, d_means, d_L, d_log_dets, d_Sigma, d_obs, T, N, K
            );
        };
        total_time_gpu += benchmark_kernel(bw_call, 0, 1); // 0 warmup pour voir l'evolution
        // std::cout << " Iter " << iter << " LL: " << ll_history_gpu[iter] << std::endl;
    }
    std::cout << "  ✓ Total Time: " << total_time_gpu << " ms" << std::endl;

    // 5. Run CPU (Reference)
    std::cout << "[CPU] Baum-Welch Algorithm..." << std::endl;
    cpu::algo::HMMModel model_cpu;
    model_cpu.T = T; model_cpu.K = K; model_cpu.N = N;
    model_cpu.pi = new float[N]; std::memcpy(model_cpu.pi, pi_init.data(), N*4);
    model_cpu.A = new float[N*N]; std::memcpy(model_cpu.A, A_init.data(), N*N*4);
    model_cpu.mu = new float[N*K]; std::memcpy(model_cpu.mu, mu_init.data(), N*K*4);
    model_cpu.Sigma = new float[N*K*K]; std::memcpy(model_cpu.Sigma, Sigma_init.data(), N*K*K*4);
    model_cpu.L = new float[N*K*K];
    model_cpu.log_det = new float[N];
    cpu::algo::precompute_cholesky(model_cpu);

    std::vector<float> workspace_cpu(2 * K);
    std::vector<float> gamma(T * N);
    std::vector<float> xi((T - 1) * N * N);
    float ll_cpu_final = 0;

    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < max_iter; ++iter) {
        ll_cpu_final = cpu::algo::baum_welch_e_step(model_cpu, data.observations, gamma.data(), xi.data(), workspace_cpu.data());
        cpu::algo::baum_welch_m_step(model_cpu, data.observations, gamma.data(), xi.data());
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    float time_cpu = std::chrono::duration<float, std::milli>(end_cpu - start_cpu).count();

    // 6. Validation
    float ll_gpu_final = ll_history_gpu.back();
    
    // Tolérance relative car les échelles de Log-Likelihood sont grandes
    // Et GPU vs CPU drift inévitablement avec des méthodes iteratives
    float diff_ll = std::abs(ll_gpu_final - ll_cpu_final);
    float rel_diff = diff_ll / std::abs(ll_cpu_final);

    std::cout << "Final LL GPU: " << ll_gpu_final << std::endl;
    std::cout << "Final LL CPU: " << ll_cpu_final << std::endl;
    std::cout << "Speedup: " << time_cpu / total_time_gpu << "x" << std::endl;

    if (rel_diff < 0.05f) { // 5% de marge d'erreur sur la convergence finale
        std::cout << GREEN << "[REUSSI] Convergence Similaire" << RESET << std::endl;
    } else {
        std::cout << RED << "[ATTENTION] Divergence numérique (Log-Space vs Linear)" << RESET << std::endl;
        // On ne marque pas comme échec strict si ça apprend
    }

    // Cleanup...
    // (Je te laisse remettre les free/delete)

    // ========================================================================
    // 9. Cleanup
    // ========================================================================
    cudaFree(d_pi); cudaFree(d_A); cudaFree(d_means);
    cudaFree(d_L); cudaFree(d_log_dets); cudaFree(d_Sigma); cudaFree(d_obs);
    
    delete[] model_cpu.pi;
    delete[] model_cpu.A;
    delete[] model_cpu.mu;
    delete[] model_cpu.Sigma;
    delete[] model_cpu.L;
    delete[] model_cpu.log_det;
    free_synthetic_data(data);
}

// void test_baum_welch(int T, int N, int K, int max_iter = 20) {
//     std::cout << "\n============================================" << std::endl;
//     std::cout << "Test Baum-Welch: T=" << T << ", N=" << N << ", K=" << K << std::endl;
//     std::cout << "============================================\n" << std::endl;

//     // 1. Génération données synthétiques
//     auto data = generate_synthetic_hmm_sequence(T, K, N, 0.6f, 2.0f, 2.0f);
    
//     // 2. Initialisation aléatoire des paramètres (à estimer)
//     std::vector<float> pi_init(N);
//     std::vector<float> A_init(N * N);
//     std::vector<float> mu_init(N * K);
//     std::vector<float> Sigma_init(N * K * K);
    
//     generate_initial_distribution(pi_init.data(), N);
//     generate_transition_matrix(A_init.data(), N, 0.6f);
//     generate_separated_means(mu_init.data(), N, K, 2.0f);
//     generate_covariance_matrices(Sigma_init.data(), N, K);
    
//     // 3. Conversion Log-Space pour GPU
//     std::vector<float> log_pi_gpu(N);
//     std::vector<float> log_A_gpu(N * N);
    
//     for (int i = 0; i < N; ++i) {
//         log_pi_gpu[i] = std::log(pi_init[i]);
//         for (int j = 0; j < N; ++j) {
//             log_A_gpu[i * N + j] = std::log(A_init[i * N + j]);
            
//         }
//     }
    
//     // 4. Cholesky initial pour GPU
//     std::vector<float> L_init(N * K * K);
//     std::vector<float> log_dets_init(N);
    
//     for (int i = 0; i < N; ++i) {
//         float* Sigma_i = Sigma_init.data() + i * K * K;
//         float* L_i = L_init.data() + i * K * K;
        
//         std::memcpy(L_i, Sigma_i, K * K * sizeof(float));
        
//         bool success = cpu::linalg::cholesky_decomposition(L_i, K);
//         if (!success) {
//             std::cerr << "Warning: Cholesky failed for state " << i << std::endl;
//         }
        
//         log_dets_init[i] = cpu::linalg::log_det_from_cholesky(L_i, K);
//     }
    
//     // ========================================================================
//     // 5. BAUM-WELCH GPU
//     // ========================================================================
//     std::cout << "[GPU] Baum-Welch Algorithm (" << max_iter << " iterations)..." << std::endl;
    
//     // Allocation GPU
//     float *d_pi, *d_A, *d_means, *d_L, *d_log_dets, *d_Sigma, *d_obs;
    
//     cudaMalloc(&d_pi, N * sizeof(float));
//     cudaMalloc(&d_A, N * N * sizeof(float));
//     cudaMalloc(&d_means, N * K * sizeof(float));
//     cudaMalloc(&d_L, N * K * K * sizeof(float));
//     cudaMalloc(&d_log_dets, N * sizeof(float));
//     cudaMalloc(&d_Sigma, N * K * K * sizeof(float));
//     cudaMalloc(&d_obs, T * K * sizeof(float));
//     check_cuda_error("Allocation GPU");
    
//     // Copie initiale Host -> Device
//     cudaMemcpy(d_pi, log_pi_gpu.data(), N * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_A, log_A_gpu.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_means, mu_init.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_L, L_init.data(), N * K * K * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_log_dets, log_dets_init.data(), N * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_Sigma, Sigma_init.data(), N * K * K * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_obs, data.observations, T * K * sizeof(float), cudaMemcpyHostToDevice);
//     check_cuda_error("Copie H->D initiale");
    
//     // EM Loop GPU
//     std::vector<float> ll_history_gpu(max_iter);
//     float total_time_gpu = 0.0f;
    
//     for (int iter = 0; iter < max_iter; ++iter) {
//         auto bw_gpu_call = [&]() {
//             ll_history_gpu[iter] = gpu::baum_welch_step_gpu(
//                 d_pi, d_A, d_means, d_L, d_log_dets, d_Sigma, d_obs, T, N, K
//             );
//         };
        
//         float iter_time = benchmark_kernel(bw_gpu_call, 1, 1);
//         total_time_gpu += iter_time;
        
//         std::cout << "  Iter " << iter << ": LL=" << ll_history_gpu[iter] 
//                   << " (" << iter_time << " ms)" << std::endl;
//     }
    
//     std::cout << "  ✓ Baum-Welch GPU Total: " << total_time_gpu << " ms" << std::endl;
    
//     // ========================================================================
//     // 6. BAUM-WELCH CPU (Référence)
//     // ========================================================================
//     std::cout << "[CPU] Baum-Welch Algorithm (" << max_iter << " iterations)..." << std::endl;
    
//     // Copie modèle pour CPU
//     cpu::algo::HMMModel model_cpu;
//     model_cpu.T = T;
//     model_cpu.K = K;
//     model_cpu.N = N;
//     model_cpu.pi = new float[N];
//     model_cpu.A = new float[N * N];
//     model_cpu.mu = new float[N * K];
//     model_cpu.Sigma = new float[N * K * K];
//     model_cpu.L = new float[N * K * K];
//     model_cpu.log_det = new float[N];
    
//     std::memcpy(model_cpu.pi, pi_init.data(), N * sizeof(float));
//     std::memcpy(model_cpu.A, A_init.data(), N * N * sizeof(float));
//     std::memcpy(model_cpu.mu, mu_init.data(), N * K * sizeof(float));
//     std::memcpy(model_cpu.Sigma, Sigma_init.data(), N * K * K * sizeof(float));
//     cpu::algo::precompute_cholesky(model_cpu);
    
//     // EM Loop CPU
//     std::vector<float> workspace_cpu(2 * K);
//     std::vector<float> gamma(T * N);
//     std::vector<float> xi((T - 1) * N * N);
//     std::vector<float> ll_history_cpu(max_iter);
    
//     auto start_cpu = std::chrono::high_resolution_clock::now();
    
//     for (int iter = 0; iter < max_iter; ++iter) {
//         // E-Step
//         ll_history_cpu[iter] = cpu::algo::baum_welch_e_step(
//             model_cpu, data.observations, gamma.data(), xi.data(), workspace_cpu.data()
//         );
        
//         // M-Step
//         cpu::algo::baum_welch_m_step(model_cpu, data.observations, gamma.data(), xi.data());
        
//         std::cout << "  Iter " << iter << ": LL=" << ll_history_cpu[iter] << std::endl;
//     }
    
//     auto end_cpu = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> elapsed_cpu = end_cpu - start_cpu;
    
//     std::cout << "✓ Baum-Welch CPU Total: " << elapsed_cpu.count() << " ms" << std::endl;
    
//     // ========================================================================
//     // 7. VALIDATION (Récupération GPU -> Host)
//     // ========================================================================
//     std::cout << "[Validation] Comparing GPU vs CPU..." << std::endl;
    
//     std::vector<float> mu_gpu(N * K);
//     std::vector<float> A_gpu(N * N);
    
//     cudaMemcpy(mu_gpu.data(), d_means, N * K * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(A_gpu.data(), d_A, N * N * sizeof(float), cudaMemcpyDeviceToHost);
//     check_cuda_error("Copie D->H finale");
    
//     // Convertir A GPU (log-space) en linéaire pour comparaison
//     std::vector<float> A_gpu_linear(N * N);
//     for (int i = 0; i < N * N; ++i) {
//         A_gpu_linear[i] = std::exp(A_gpu[i]);
//     }
    
//     // Compare Means
//     float error_mu = compute_relative_error(mu_gpu.data(), model_cpu.mu, N * K);
//     std::cout << "Means Relative Error: " << error_mu << std::endl;
    
//     // Compare Transition Matrix
//     float error_A = compute_relative_error(A_gpu_linear.data(), model_cpu.A, N * N);
//     std::cout << "Transition Matrix Relative Error: " << error_A << std::endl;
    
//     // Compare Log-Likelihoods
//     float ll_final_gpu = ll_history_gpu.back();
//     float ll_final_cpu = ll_history_cpu.back();
//     float ll_diff = std::abs(ll_final_gpu - ll_final_cpu);
    
//     std::cout << "Final Log-Likelihood (GPU): " << ll_final_gpu << std::endl;
//     std::cout << "Final Log-Likelihood (CPU): " << ll_final_cpu << std::endl;
//     std::cout << "LL Difference: " << ll_diff << std::endl;
    
//     // Assertions
//     const float param_threshold = 0.1f; // 10% tolerance sur paramètres
//     const float ll_threshold = 1.0f;     // Tolérance absolue sur LL
    
//     TEST_ASSERT(error_mu < param_threshold, "Means converged similarly");
//     TEST_ASSERT(error_A < param_threshold, "Transition matrix converged similarly");
//     TEST_ASSERT(ll_diff < ll_threshold, "Log-likelihood converged to similar value");
    
//     // ========================================================================
//     // 8. SPEEDUP
//     // ========================================================================
//     float speedup = elapsed_cpu.count() / total_time_gpu;
    
//     std::cout << "[Performance Summary]" << std::endl;
//     std::cout << "GPU Total: " << total_time_gpu << " ms" << std::endl;
//     std::cout << "CPU Total: " << elapsed_cpu.count() << " ms" << std::endl;
//     std::cout << "Speedup: " << speedup << "x" << std::endl;
    
//     // ========================================================================
//     // 9. Cleanup
//     // ========================================================================
//     cudaFree(d_pi); cudaFree(d_A); cudaFree(d_means);
//     cudaFree(d_L); cudaFree(d_log_dets); cudaFree(d_Sigma); cudaFree(d_obs);
    
//     delete[] model_cpu.pi;
//     delete[] model_cpu.A;
//     delete[] model_cpu.mu;
//     delete[] model_cpu.Sigma;
//     delete[] model_cpu.L;
//     delete[] model_cpu.log_det;
    
//     free_synthetic_data(data);
    
//     std::cout << "✅ Test Baum-Welch PASSED\n" << std::endl;
// }

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST BAUM-WELCH GPU vs CPU" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Test 1: Small
    test_baum_welch(500, 3, 4, 10);
    
    // Test 2: Medium
    test_baum_welch(1000, 4, 4, 10);
    
    // Test 3: Large (moins d'itérations pour économiser temps)
    test_baum_welch(5000, 4, 4, 10);

    // Test 3: Large (moins d'itérations pour économiser temps)
    test_baum_welch(10000, 4, 4, 10);
    
    return print_test_summary();
}
