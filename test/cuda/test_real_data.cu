#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>

#include "algo_hmm_gpu.cuh"
#include "linalg_cpu.hpp" 
#include "cuda_timing.cuh"

using namespace hmm;

// Helper lecture binaire
void load_bin(const std::string& path, float* buffer, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "ERREUR: Impossible d'ouvrir " << path << std::endl;
        exit(1);
    }
    f.read(reinterpret_cast<char*>(buffer), count * sizeof(float));
    if (!f) {
        std::cerr << "ERREUR: Lecture incomplète ou échouée de " << path << std::endl;
        exit(1);
    }
    f.close();
}
int main() {
    // 1. Lire Dimensions
    int T, N, K;
    std::ifstream fdim("../../scripts/python/golden_data/dims.txt");
    if (!fdim) { 
        std::cerr << "Lancez le script Python d'abord !" << std::endl; 
        return 1; 
    }
    fdim >> T >> N >> K;
    fdim.close();

    std::cout << "Chargement Real Data: T=" << T << " N=" << N << " K=" << K << std::endl;

    // 2. Allocations Host
    std::vector<float> h_obs(T * K);
    std::vector<float> h_pi(N);
    std::vector<float> h_A(N * N);
    std::vector<float> h_mu(N * K);
    std::vector<float> h_sigma(N * K * K);
    
    // 3. Chargement
    load_bin("../../scripts/python/golden_data/obs.bin", h_obs.data(), T*K);
    load_bin("../../scripts/python/golden_data/pi_init.bin", h_pi.data(), N);
    load_bin("../../scripts/python/golden_data/A_init.bin", h_A.data(), N*N);
    load_bin("../../scripts/python/golden_data/mu_init.bin", h_mu.data(), N*K);
    load_bin("../../scripts/python/golden_data/sigma_init.bin", h_sigma.data(), N*K*K);

    // 4. Préparation (Log-Space + Cholesky) pour le GPU
    // Log Pi, Log A
    for(auto& v : h_pi) v = std::log(v + 1e-10f); // Protection log(0)
    for(auto& v : h_A) v = std::log(v + 1e-10f);

    // Cholesky Initiale (L) et LogDet
    std::vector<float> h_L(N * K * K);
    std::vector<float> h_log_dets(N);

    for(int i=0; i<N; ++i) {
        // Copie Sigma dans L
        float* src = h_sigma.data() + i*K*K;
        float* dst = h_L.data() + i*K*K;
        std::copy(src, src + K*K, dst);
        
        // Cholesky CPU
        if(!hmm::cpu::linalg::cholesky_decomposition(dst, K)) {
            std::cerr << "Erreur Cholesky Init State " << i << std::endl;
        }
        // LogDet
        float ld = 0.0f;
        for(int k=0; k<K; ++k) ld += 2.0f * std::log(dst[k*K+k]);
        h_log_dets[i] = ld;
    }

    // 5. Envoi sur GPU
    float *d_obs, *d_pi, *d_A, *d_mu, *d_L, *d_dets, *d_Sigma_out;
    cudaMalloc(&d_obs, T*K*sizeof(float));
    cudaMalloc(&d_pi, N*sizeof(float));
    cudaMalloc(&d_A, N*N*sizeof(float));
    cudaMalloc(&d_mu, N*K*sizeof(float));
    cudaMalloc(&d_L, N*K*K*sizeof(float));
    cudaMalloc(&d_dets, N*sizeof(float));
    cudaMalloc(&d_Sigma_out, N*K*K*sizeof(float)); // Buffer temp pour BW

    cudaMemcpy(d_obs, h_obs.data(), T*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pi, h_pi.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A.data(), N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu, h_mu.data(), N*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, h_L.data(), N*K*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dets, h_log_dets.data(), N*sizeof(float), cudaMemcpyHostToDevice);

    int max_iter = 20;
    float tolerance = 1e-4f;
    float prev_ll = -INFINITY;
    
    std::cout << "Starting training for " << max_iter << " iterations..." << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    auto start_train = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < max_iter; ++iter) {
        // --- STEP ---
        float log_likelihood = hmm::gpu::baum_welch_step_gpu(
            d_pi, d_A, d_mu, d_L, d_dets, d_Sigma_out, d_obs, T, N, K
        );

        // --- MONITORING ---
        float diff = log_likelihood - prev_ll;
        
        // Affichage compact
        if (iter % 10 == 0 || iter == max_iter - 1) {
            std::cout << "Iter " << std::setw(3) << iter 
                      << " | LL: " << std::setw(12) << log_likelihood 
                      << " | Delta: " << diff << std::endl;
        }

        // --- CONVERGENCE CHECK ---
        if (std::abs(diff) < tolerance && iter > 0) {
            std::cout << "\n✅ Converged at iter " << iter << std::endl;
            break;
        }
        
        prev_ll = log_likelihood;
    }

    auto end_train = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_ms = end_train - start_train;

    // ========================================================================
    // 6. RÉSULTATS & NETTOYAGE
    // ========================================================================
    std::cout << "\n------------------------------------------------" << std::endl;
    std::cout << "Total Time: " << total_ms.count() << " ms" << std::endl;
    std::cout << "Avg Time/Iter: " << total_ms.count() / max_iter << " ms" << std::endl;
    std::cout << "Final Log Likelihood: " << prev_ll << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // Récupérer les moyennes finales pour inspection
    cudaMemcpy(h_mu.data(), d_mu, N*K*sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Learned Means (State 0, first 4 dims): [ ";
    for(int k=0; k<std::min(K, 4); ++k) std::cout << h_mu[k] << " ";
    std::cout << "]" << std::endl;

    cudaFree(d_obs); cudaFree(d_pi); cudaFree(d_A);
    cudaFree(d_mu); cudaFree(d_L); cudaFree(d_dets); cudaFree(d_Sigma_out);

    return 0;
}