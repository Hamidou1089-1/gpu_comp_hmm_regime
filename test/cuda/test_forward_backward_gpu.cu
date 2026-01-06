/**
 * @file test_forward_backward_gpu.cu
 * @brief Test Forward-Backward: GPU vs CPU avec validation et benchmark
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

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
// TEST FORWARD-BACKWARD
// ============================================================================

void test_forward_backward(int T, int N, int K) {
    std::cout << "============================================" << std::endl;
    std::cout << "Test Forward-Backward: T=" << T << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "============================================\n" << std::endl;

    // 1. Génération données synthétiques
    auto data = generate_synthetic_hmm_sequence(T, K, N, 0.8f, 5.0f, 1.0f);
    
    // 2. Conversion des paramètres en Log-Space (pour GPU)
    std::vector<float> log_pi(N);
    std::vector<float> log_A(N * N);
    
    for (int i = 0; i < N; ++i) {
        log_pi[i] = std::log(data.model.pi[i]);
        for (int j = 0; j < N; ++j) {
            log_A[i * N + j] = std::log(data.model.A[i * N + j]);
        }
    }
    
    // 3. Allocation GPU
    float *d_pi, *d_A, *d_means, *d_L, *d_log_dets, *d_obs;
    float *d_alpha, *d_beta;
    
    cudaMalloc(&d_pi, N * sizeof(float));
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_means, N * K * sizeof(float));
    cudaMalloc(&d_L, N * K * K * sizeof(float));
    cudaMalloc(&d_log_dets, N * sizeof(float));
    cudaMalloc(&d_obs, T * K * sizeof(float));
    cudaMalloc(&d_alpha, T * N * sizeof(float));
    cudaMalloc(&d_beta, T * N * sizeof(float));
    check_cuda_error("Allocation GPU");
    
    // 4. Copie Host -> Device
    cudaMemcpy(d_pi, log_pi.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, log_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_means, data.model.mu, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, data.model.L, N * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_log_dets, data.model.log_det, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs, data.observations, T * K * sizeof(float), cudaMemcpyHostToDevice);
    check_cuda_error("Copie H->D");
    
    // ========================================================================
    // 5. FORWARD GPU
    // ========================================================================
    std::cout << "[GPU] Forward Algorithm..." << std::endl;
    
    auto forward_gpu_call = [&]() {
        gpu::forward_gpu(d_pi, d_A, d_means, d_L, d_log_dets, d_obs, d_alpha, T, N, K);
    };
    
    float time_forward_gpu = benchmark_kernel(forward_gpu_call, 3, 10);
    std::cout << "  ✓ Forward GPU: " << time_forward_gpu << " ms (avg)" << std::endl;
    
    // ========================================================================
    // 6. BACKWARD GPU
    // ========================================================================
    std::cout << "[GPU] Backward Algorithm..." << std::endl;
    
    auto backward_gpu_call = [&]() {
        gpu::backward_gpu(d_pi, d_A, d_means, d_L, d_log_dets, d_obs, d_beta, T, N, K);
    };
    
    float time_backward_gpu = benchmark_kernel(backward_gpu_call, 3, 10);
    std::cout << "✓ Backward GPU: " << time_backward_gpu << " ms (avg)" << std::endl;
    
    // ========================================================================
    // 7. FORWARD-BACKWARD CPU (Référence)
    // ========================================================================
    std::cout << "[CPU] Forward-Backward Algorithm..." << std::endl;
    
    std::vector<float> log_potentials(N + (T - 1) * N * N);
    std::vector<float> workspace(2 * K);
    std::vector<float> alpha_cpu(T * N);
    std::vector<float> beta_cpu(T * N);
    
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    // Compute potentials
    cpu::algo::compute_log_gaussian_potentials(
        data.model, data.observations, log_potentials.data(), workspace.data()
    );
    
    // Forward
    cpu::algo::forward_algorithm(log_potentials.data(), alpha_cpu.data(), T, N);
    
    // Backward
    cpu::algo::backward_algorithm(log_potentials.data(), beta_cpu.data(), T, N);
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_cpu = end_cpu - start_cpu;
    
    std::cout << "  ✓ Forward-Backward CPU: " << elapsed_cpu.count() << " ms" << std::endl;
    
    // ========================================================================
    // 8. VALIDATION (Récupération GPU -> Host)
    // ========================================================================
    std::cout << "[Validation] Comparing GPU vs CPU..." << std::endl;
    
    std::vector<float> alpha_gpu(T * N);
    std::vector<float> beta_gpu(T * N);
    
    cudaMemcpy(alpha_gpu.data(), d_alpha, T * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(beta_gpu.data(), d_beta, T * N * sizeof(float), cudaMemcpyDeviceToHost);
    check_cuda_error("Copie D->H");
    
    // Compare Alpha
    float error_alpha = compute_relative_error(alpha_gpu.data(), alpha_cpu.data(), T * N);
    std::cout << "Alpha Relative Error: " << error_alpha << std::endl;
    
    // Compare Beta
    float error_beta = compute_relative_error(beta_gpu.data(), beta_cpu.data(), T * N);
    std::cout << "Beta Relative Error: " << error_beta << std::endl;
    
    // Assertions
    const float threshold = 1e-3f; // Tolérance log-space
    TEST_ASSERT(error_alpha < threshold, "Alpha GPU vs CPU match");
    TEST_ASSERT(error_beta < threshold, "Beta GPU vs CPU match");
    
    // ========================================================================
    // 9. SPEEDUP
    // ========================================================================
    float total_gpu = time_forward_gpu + time_backward_gpu;
    float speedup = elapsed_cpu.count() / total_gpu;
    
    std::cout << "\n[Performance Summary]" << std::endl;
    std::cout << "  GPU Total: " << total_gpu << " ms" << std::endl;
    std::cout << "  CPU Total: " << elapsed_cpu.count() << " ms" << std::endl;
    std::cout << "  Speedup: " << speedup << "x" << std::endl;
    
    // ========================================================================
    // 10. Cleanup
    // ========================================================================
    cudaFree(d_pi); cudaFree(d_A); cudaFree(d_means);
    cudaFree(d_L); cudaFree(d_log_dets); cudaFree(d_obs);
    cudaFree(d_alpha); cudaFree(d_beta);
    
    free_synthetic_data(data);
    
    std::cout << "\n✅ Test Forward-Backward PASSED\n" << std::endl;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST FORWARD-BACKWARD GPU vs CPU" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Test 1: Small
    test_forward_backward(500, 3, 4);
    
    // Test 2: Medium
    test_forward_backward(1000, 4, 8);
    
    // Test 3: Large
    test_forward_backward(5000, 5, 10);

    // Test 3: Large
    test_forward_backward(10000, 5, 10);
    
    return print_test_summary();
}
