#include "algo_hmm_gpu.cuh"
#include "scan_kernels.cuh"
#include "potential_kernels.cuh"
#include "hmm_semirings.cuh"
#include "viterbi_kernels.cuh" 
#include "baum_welch_kernels.cuh" 
#include "linalg_cpu.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

namespace hmm {
namespace gpu {

// Helpers extraction
__global__ void extract_row0_kernel(const float* scan, float* out, int T, int N) {
    int t = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T && i < N) out[t*N + i] = scan[t*N*N + i]; // Ligne 0
}

__global__ void extract_reverse_beta_kernel(const float* scan, float* out, int T, int N) {
    int t = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T && i < N) {
        // Output[t] = Scan[T - 1 - t] (ligne 0)
        int k = T - 1 - t;
        out[t*N + i] = scan[k*N*N + i];
    }
}

__global__ void gamma_kernel(const float* a, const float* b, float* g, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) g[idx] = a[idx] + b[idx];
}

// ----------------------------------------------------------------------------
// FORWARD
// ----------------------------------------------------------------------------
void forward_gpu(const float* d_pi, const float* d_A, const float* d_means, const float* d_L, const float* d_log_dets, const float* d_obs, float* d_alpha_out, int T, int N, int K) {
    float *d_scan, *d_tmp;
    size_t sz = T * N * N * sizeof(float);
    cudaMalloc(&d_scan, sz);
    cudaMalloc(&d_tmp, sz);

    kernels::launch_prepare_forward_inputs(d_obs, d_means, d_L, d_log_dets, d_pi, d_A, d_scan, T, N, K);
    
    kernels::run_parallel_scan<semiring::LogSum>(d_scan, d_tmp, T, N);
    
    dim3 b(256); dim3 g((N+255)/256, T);
    extract_row0_kernel<<<g, b>>>(d_scan, d_alpha_out, T, N);

    cudaFree(d_scan); cudaFree(d_tmp);
}

// ----------------------------------------------------------------------------
// BACKWARD
// ----------------------------------------------------------------------------
void backward_gpu(const float* d_pi, const float* d_A, const float* d_means, const float* d_L, const float* d_log_dets, const float* d_obs, float* d_beta_out, int T, int N, int K) {
    float *d_scan, *d_tmp;
    size_t sz = T * N * N * sizeof(float);
    cudaMalloc(&d_scan, sz);
    cudaMalloc(&d_tmp, sz);

    kernels::launch_prepare_backward_inputs(d_obs, d_means, d_L, d_log_dets, d_pi, d_A, d_scan, T, N, K);
    
    kernels::run_parallel_scan<semiring::LogSum>(d_scan, d_tmp, T, N);
    
    dim3 b(256); dim3 g((N+255)/256, T);
    extract_reverse_beta_kernel<<<g, b>>>(d_scan, d_beta_out, T, N);

    cudaFree(d_scan); cudaFree(d_tmp);
}

// ----------------------------------------------------------------------------
// VITERBI SCORE
// ----------------------------------------------------------------------------
void viterbi_score_gpu(const float* d_pi, const float* d_A, const float* d_means, const float* d_L, const float* d_log_dets, const float* d_obs, float* d_delta_out, int T, int N, int K) {
    float *d_scan, *d_tmp;
    size_t sz = T * N * N * sizeof(float);
    cudaMalloc(&d_scan, sz);
    cudaMalloc(&d_tmp, sz);

    kernels::launch_prepare_forward_inputs(d_obs, d_means, d_L, d_log_dets, d_pi, d_A, d_scan, T, N, K);
    
    kernels::run_parallel_scan<semiring::MaxSum>(d_scan, d_tmp, T, N);
    
    dim3 b(256); dim3 g((N+255)/256, T);
    extract_row0_kernel<<<g, b>>>(d_scan, d_delta_out, T, N);

    cudaFree(d_scan); cudaFree(d_tmp);
}

// ----------------------------------------------------------------------------
// SMOOTHING
// ----------------------------------------------------------------------------
void smoothing_gpu(const float* d_alpha, const float* d_beta, float* d_gamma_out, int T, int N) {
    int total = T * N;
    gamma_kernel<<<(total+255)/256, 256>>>(d_alpha, d_beta, d_gamma_out, total);
}


// ============================================================================
// VITERBI PATH COMPLET
// ============================================================================
void viterbi_path_gpu(
    const float* d_pi, const float* d_A, 
    const float* d_means, const float* d_L, const float* d_log_dets,
    const float* d_obs, 
    int* d_path_out, 
    int T, int N, int K
) {
    float *d_scan, *d_tmp, *d_delta;
    cudaMalloc(&d_scan, T * N * N * sizeof(float));
    cudaMalloc(&d_tmp, T * N * N * sizeof(float));
    cudaMalloc(&d_delta, T * N * sizeof(float));

    // 1. Scan Max-Sum
    kernels::launch_prepare_forward_inputs(d_obs, d_means, d_L, d_log_dets, d_pi, d_A, d_scan, T, N, K);
    kernels::run_parallel_scan<semiring::MaxSum>(d_scan, d_tmp, T, N);
    
    dim3 block_extract(256);
    dim3 grid_extract((N + 255) / 256, T);
    extract_row0_kernel<<<grid_extract, block_extract>>>(d_scan, d_delta, T, N);
    
    // 3. Backtrack
    kernels::launch_viterbi_backtrack(d_delta, d_A, d_path_out, T, N);

    cudaFree(d_scan); cudaFree(d_tmp); cudaFree(d_delta);
}

// ============================================================================
// BAUM-WELCH STEP (Avec Cholesky CPU)
// ============================================================================
float baum_welch_step_gpu(
    float* d_pi, float* d_A, 
    float* d_means, float* d_L, float* d_log_dets, 
    float* d_Sigma, // Buffer GPU pour Covariance [N*K*K]
    const float* d_obs,
    int T, int N, int K
) {
    // 1. Allocations temporaires
    float *d_alpha, *d_beta, *d_log_gamma, *d_emissions, *d_sum_gamma;
    
    float *d_new_mu, *d_new_Sigma;

    cudaMalloc(&d_alpha, T*N*sizeof(float));
    cudaMalloc(&d_beta, T*N*sizeof(float));
    cudaMalloc(&d_log_gamma, T*N*sizeof(float));
    cudaMalloc(&d_emissions, T*N*sizeof(float));
    cudaMalloc(&d_sum_gamma, N*sizeof(float)); // Pour dénominateur

    

    cudaMalloc(&d_new_mu, N*K*sizeof(float));
    cudaMalloc(&d_new_Sigma, N*K*K*sizeof(float));

    // 2. E-STEP
    forward_gpu(d_pi, d_A, d_means, d_L, d_log_dets, d_obs, d_alpha, T, N, K);
    backward_gpu(d_pi, d_A, d_means, d_L, d_log_dets, d_obs, d_beta, T, N, K);
    
    // Emissions pures
    kernels::launch_compute_emissions(d_obs, d_means, d_L, d_log_dets, d_emissions, T, N, K);

    // Calcul LogLikelihood (sur CPU pour l'instant pour la réduction)
    float* d_log_likelihood_ptr;
    cudaMalloc(&d_log_likelihood_ptr, sizeof(float));

    // 2. Lancer le kernel de réduction
    kernels::launch_compute_log_likelihood(d_alpha, d_log_likelihood_ptr, T, N);

    // 3. Récupérer le résultat scalaire sur CPU (pour le retour de fonction et log)
    float log_likelihood;
    cudaMemcpy(&log_likelihood, d_log_likelihood_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_log_likelihood_ptr);

    
    // Gamma
    kernels::launch_compute_gamma(d_alpha, d_beta, d_log_gamma, T, N);

    // 3. M-STEP (Updates GPU)
    // Update A
    kernels::launch_update_transition(d_alpha, d_beta, d_emissions, d_A, d_A, log_likelihood, T, N);
    kernels::launch_normalize_A(d_A, N); // A est maintenant mis à jour en log-prob

    // Update Mu & Sigma
    kernels::launch_update_gaussian(d_obs, d_log_gamma, d_new_mu, d_new_Sigma, d_sum_gamma, log_likelihood, T, N, K);
    
    cudaMemcpy(d_means, d_new_mu, N*K*sizeof(float), cudaMemcpyDeviceToDevice);
    
    // B. Recopier les nouvelles covariances dans d_Sigma (pour export CPU Cholesky)
    cudaMemcpy(d_Sigma, d_new_Sigma, N*K*K*sizeof(float), cudaMemcpyDeviceToDevice);

    // --- HYBRID CHOLESKY (GPU -> CPU -> GPU) ---
    
    // a. Copier Sigma (GPU) -> Host
    int sigma_size = N * K * K;
    std::vector<float> h_Sigma(sigma_size);
    cudaMemcpy(h_Sigma.data(), d_Sigma, sigma_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // b. Buffer pour LogDet (Host)
    std::vector<float> h_log_dets(N);

    // c. Traitement CPU (Cholesky pour chaque état)
    for (int i = 0; i < N; ++i) {
        float* sigma_i = h_Sigma.data() + i * K * K;
        
        // Ajout epsilon pour stabilité numérique (Ridge)
        for(int k=0; k<K; ++k) sigma_i[k*K + k] += 1e-5f;

        // Cholesky In-Place (modifie sigma_i pour devenir L)
        bool success = hmm::cpu::linalg::cholesky_decomposition(sigma_i, K);
        
        if (!success) {
            std::cerr << "WARNING: Cholesky failed for state " << i << ". Keeping old L." << std::endl;
            // On pourrait recharger l'ancien L ici si on avait gardé une copie
        }
        
        // Calcul Log Det (2 * sum log diag L)
        float ld = 0.0f;
        for(int k=0; k<K; ++k) ld += 2.0f * logf(sigma_i[k*K + k]);
        h_log_dets[i] = ld;
    }

    // d. Copier L et LogDets (Host) -> GPU
    // Note: h_Sigma contient maintenant L car cholesky est in-place
    cudaMemcpy(d_L, h_Sigma.data(), sigma_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_log_dets, h_log_dets.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Nettoyage
    cudaFree(d_alpha); cudaFree(d_beta); cudaFree(d_log_gamma); 
    cudaFree(d_emissions); cudaFree(d_sum_gamma);
    cudaFree(d_new_mu); cudaFree(d_new_Sigma);
    return log_likelihood;
}

} // namespace gpu
} // namespace hmm