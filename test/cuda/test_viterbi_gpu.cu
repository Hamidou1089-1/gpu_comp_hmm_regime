/**
 * @file test_viterbi_gpu.cu
 * @brief Test Viterbi (MAP): GPU vs CPU avec validation et benchmark
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

float compute_path_accuracy(const int* path_a, const int* path_b, int T) {
    int correct = 0;
    for (int t = 0; t < T; ++t) {
        if (path_a[t] == path_b[t]) correct++;
    }
    return (float)correct / T;
}

// ============================================================================
// TEST VITERBI
// ============================================================================

void test_viterbi(int T, int N, int K) {
    std::cout << "\n============================================" << std::endl;
    std::cout << "Test Viterbi: T=" << T << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "============================================\n" << std::endl;

    // 1. Génération données synthétiques
    auto data = generate_synthetic_hmm_sequence(T, K, N, 0.8f, 5.0f, 1.0f);
    
    // 2. Conversion Log-Space
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
    int *d_path;
    
    cudaMalloc(&d_pi, N * sizeof(float));
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_means, N * K * sizeof(float));
    cudaMalloc(&d_L, N * K * K * sizeof(float));
    cudaMalloc(&d_log_dets, N * sizeof(float));
    cudaMalloc(&d_obs, T * K * sizeof(float));
    cudaMalloc(&d_path, T * sizeof(int));
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
    // 5. VITERBI GPU
    // ========================================================================
    std::cout << "[GPU] Viterbi Algorithm..." << std::endl;
    
    auto viterbi_gpu_call = [&]() {
        gpu::viterbi_path_gpu(d_pi, d_A, d_means, d_L, d_log_dets, d_obs, d_path, T, N, K);
    };
    
    float time_viterbi_gpu = benchmark_kernel(viterbi_gpu_call, 3, 10);
    std::cout << "  ✓ Viterbi GPU: " << time_viterbi_gpu << " ms (avg)" << std::endl;
    
    // ========================================================================
    // 6. VITERBI CPU (Référence)
    // ========================================================================
    std::cout << "\n[CPU] Viterbi Algorithm..." << std::endl;
    
    std::vector<float> log_potentials(N + (T - 1) * N * N);
    std::vector<float> workspace(2 * K);
    std::vector<int> path_cpu(T);
    
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    // Compute potentials
    cpu::algo::compute_log_gaussian_potentials(
        data.model, data.observations, log_potentials.data(), workspace.data()
    );
    
    // Viterbi
    float log_prob_cpu = cpu::algo::viterbi_algorithm(
        log_potentials.data(), path_cpu.data(), T, N
    );
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_cpu = end_cpu - start_cpu;
    
    std::cout << "  ✓ Viterbi CPU: " << elapsed_cpu.count() << " ms" << std::endl;
    std::cout << "  Log-Probability (CPU): " << log_prob_cpu << std::endl;
    
    // ========================================================================
    // 7. VALIDATION (Récupération GPU -> Host)
    // ========================================================================
    std::cout << "\n[Validation] Comparing GPU vs CPU..." << std::endl;
    
    std::vector<int> path_gpu(T);
    cudaMemcpy(path_gpu.data(), d_path, T * sizeof(int), cudaMemcpyDeviceToHost);
    check_cuda_error("Copie D->H");
    
    // Compare Paths
    float accuracy = compute_path_accuracy(path_gpu.data(), path_cpu.data(), T);
    std::cout << "  Path Accuracy: " << (accuracy * 100.0f) << "%" << std::endl;
    
    // Assertions
    TEST_ASSERT(accuracy > 0.95f, "Viterbi paths match (>95%)");
    
    // Comparer avec états vrais (si disponibles)
    if (data.true_states) {
        float accuracy_true_gpu = compute_path_accuracy(path_gpu.data(), data.true_states, T);
        float accuracy_true_cpu = compute_path_accuracy(path_cpu.data(), data.true_states, T);
        
        std::cout << "  Accuracy vs True States (GPU): " << (accuracy_true_gpu * 100.0f) << "%" << std::endl;
        std::cout << "  Accuracy vs True States (CPU): " << (accuracy_true_cpu * 100.0f) << "%" << std::endl;
    }
    
    // ========================================================================
    // 8. SPEEDUP
    // ========================================================================
    float speedup = elapsed_cpu.count() / time_viterbi_gpu;
    
    std::cout << "\n[Performance Summary]" << std::endl;
    std::cout << "  GPU Total: " << time_viterbi_gpu << " ms" << std::endl;
    std::cout << "  CPU Total: " << elapsed_cpu.count() << " ms" << std::endl;
    std::cout << "  Speedup: " << speedup << "x" << std::endl;
    
    // ========================================================================
    // 9. Cleanup
    // ========================================================================
    cudaFree(d_pi); cudaFree(d_A); cudaFree(d_means);
    cudaFree(d_L); cudaFree(d_log_dets); cudaFree(d_obs);
    cudaFree(d_path);
    
    free_synthetic_data(data);
    
    std::cout << "\n✅ Test Viterbi PASSED\n" << std::endl;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST VITERBI GPU vs CPU" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Test 1: Small
    test_viterbi(500, 3, 4);
    
    // Test 2: Medium
    test_viterbi(1000, 4, 8);
    
    // Test 3: Large
    test_viterbi(5000, 5, 10);

    // Test 3: Large
    test_viterbi(10000, 5, 10);
    
    return print_test_summary();
}
