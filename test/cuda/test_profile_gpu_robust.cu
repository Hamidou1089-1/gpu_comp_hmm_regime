/**
 * @file test_profile_gpu_robust.cu
 * @brief Profiling GPU robuste sur datasets binaires générés par generate_benchmark_data.py
 * 
 * Usage:
 *   ./test_profile_gpu_robust <data_dir> <output_prefix> [mode]
 *   ./test_profile_gpu_robust data/bench results/gpu_benchmark
 * 
 * Modes:
 *   all         - Run all tests (default)
 *   scaling     - Only scaling tests (T and N)
 *   convergence - Only convergence tests
 *   quick       - Quick test on small datasets only
 * 
 * Génère:
 *   - <output_prefix>_results.csv
 *   - <output_prefix>_results.json
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

#include "algo_hmm_gpu.cuh"
#include "linalg_cpu.hpp"
#include "profiling_utils_gpu.cuh"


namespace fs = std::filesystem;
using namespace hmm;

// =============================================================================
// DATA LOADING
// =============================================================================

struct BenchmarkData {
    int T, N, K;
    std::string name;
    
    std::vector<float> observations;
    std::vector<float> pi_init;
    std::vector<float> A_init;
    std::vector<float> mu_init;
    std::vector<float> Sigma_init;
};

bool load_benchmark_data(const std::string& base_path, BenchmarkData& data) {
    // Load dimensions
    std::ifstream fdims(base_path + "_dims.txt");
    if (!fdims.is_open()) {
        std::cerr << "Cannot open dims file: " << base_path + "_dims.txt" << std::endl;
        return false;
    }
    fdims >> data.T >> data.N >> data.K;
    fdims.close();
    
    // Extract name from path
    data.name = fs::path(base_path).filename().string();
    
    auto load_binary = [](const std::string& path, std::vector<float>& vec, size_t expected_size) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return false;
        vec.resize(expected_size);
        f.read(reinterpret_cast<char*>(vec.data()), expected_size * sizeof(float));
        return f.good() || f.eof();
    };
    
    if (!load_binary(base_path + "_obs.bin", data.observations, data.T * data.K)) return false;
    if (!load_binary(base_path + "_pi.bin", data.pi_init, data.N)) return false;
    if (!load_binary(base_path + "_A.bin", data.A_init, data.N * data.N)) return false;
    if (!load_binary(base_path + "_mu.bin", data.mu_init, data.N * data.K)) return false;
    if (!load_binary(base_path + "_sigma.bin", data.Sigma_init, data.N * data.K * data.K)) return false;
    
    return true;
}

// =============================================================================
// GPU DATA MANAGEMENT
// =============================================================================

struct GPUData {
    float *d_obs;
    float *d_pi, *d_A;
    float *d_mu, *d_L, *d_log_dets;
    float *d_Sigma;
    float *d_alpha, *d_beta, *d_gamma;
    float *d_path;
    
    int T, N, K;
    
    void allocate(int T_, int N_, int K_) {
        T = T_; N = N_; K = K_;
        
        CUDA_CHECK(cudaMalloc(&d_obs, T * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pi, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mu, N * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_L, N * K * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_log_dets, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Sigma, N * K * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_alpha, T * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_beta, T * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gamma, T * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_path, T * N * sizeof(float)));
    }
    
    void free() {
        cudaFree(d_obs);
        cudaFree(d_pi);
        cudaFree(d_A);
        cudaFree(d_mu);
        cudaFree(d_L);
        cudaFree(d_log_dets);
        cudaFree(d_Sigma);
        cudaFree(d_alpha);
        cudaFree(d_beta);
        cudaFree(d_gamma);
        cudaFree(d_path);
    }
    
    void upload(const BenchmarkData& data, 
                const std::vector<float>& L, 
                const std::vector<float>& log_dets,
                const std::vector<float>& log_pi,
                const std::vector<float>& log_A) {
        CUDA_CHECK(cudaMemcpy(d_obs, data.observations.data(), T * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pi, log_pi.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_A, log_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mu, data.mu_init.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_L, L.data(), N * K * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_log_dets, log_dets.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Sigma, data.Sigma_init.data(), N * K * K * sizeof(float), cudaMemcpyHostToDevice));
    }
};

// Prepare initial parameters (Cholesky, log conversion)
void prepare_params(const BenchmarkData& data,
                    std::vector<float>& L,
                    std::vector<float>& log_dets,
                    std::vector<float>& log_pi,
                    std::vector<float>& log_A) {
    int N = data.N;
    int K = data.K;
    
    // Compute Cholesky decomposition
    L = data.Sigma_init;
    log_dets.resize(N);
    
    for (int i = 0; i < N; i++) {
        float* L_i = L.data() + i * K * K;
        cpu::linalg::cholesky_decomposition(L_i, K);
        log_dets[i] = cpu::linalg::log_det_from_cholesky(L_i, K);
    }
    
    // Convert to log space
    log_pi.resize(N);
    log_A.resize(N * N);
    
    for (int i = 0; i < N; i++) {
        log_pi[i] = std::log(data.pi_init[i]);
    }
    for (int i = 0; i < N * N; i++) {
        log_A[i] = std::log(data.A_init[i]);
    }
}

// =============================================================================
// PROFILING FUNCTIONS
// =============================================================================

GPUProfileResult profile_forward(const BenchmarkData& data, GPUData& gpu, int num_runs = 100) {
    GPUProfileResult result;
    result.dataset_name = data.name;
    result.algo_name = "forward";
    result.T = data.T;
    result.N = data.N;
    result.K = data.K;
    result.iterations = 1;
    get_gpu_info(result);
    
    // Warmup
    gpu::forward_gpu(gpu.d_pi, gpu.d_A, gpu.d_mu, gpu.d_L, gpu.d_log_dets,
                     gpu.d_obs, gpu.d_alpha, data.T, data.N, data.K);
    cudaDeviceSynchronize();
    
    // Benchmark
    std::vector<float> times;
    CudaTimer timer;
    
    for (int run = 0; run < num_runs; run++) {
        timer.start();
        gpu::forward_gpu(gpu.d_pi, gpu.d_A, gpu.d_mu, gpu.d_L, gpu.d_log_dets,
                         gpu.d_obs, gpu.d_alpha, data.T, data.N, data.K);
        times.push_back(timer.stop());
    }
    
    compute_timing_stats(times, result.time_ms, result.time_std_ms, 
                         result.time_min_ms, result.time_max_ms);
    
    // Get LL from last alpha
    std::vector<float> alpha(data.T * data.N);
    cudaMemcpy(alpha.data(), gpu.d_alpha, data.T * data.N * sizeof(float), cudaMemcpyDeviceToHost);
    float ll = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < data.N; i++) {
        ll = cpu::linalg::log_sum_exp(ll, alpha[(data.T - 1) * data.N + i]);
    }
    result.log_likelihood = ll;
    
    result.effective_bandwidth_GBs = estimate_hmm_bandwidth(data.T, data.N, data.K, result.time_ms, "forward");
    result.theoretical_occupancy = 0; // Would need kernel pointer
    result.gpu_memory_used_bytes = get_gpu_memory_used();
    
    return result;
}

GPUProfileResult profile_backward(const BenchmarkData& data, GPUData& gpu, int num_runs = 100) {
    GPUProfileResult result;
    result.dataset_name = data.name;
    result.algo_name = "backward";
    result.T = data.T;
    result.N = data.N;
    result.K = data.K;
    result.iterations = 1;
    get_gpu_info(result);
    
    gpu::backward_gpu(gpu.d_pi, gpu.d_A, gpu.d_mu, gpu.d_L, gpu.d_log_dets,
                      gpu.d_obs, gpu.d_beta, data.T, data.N, data.K);
    cudaDeviceSynchronize();
    
    std::vector<float> times;
    CudaTimer timer;
    
    for (int run = 0; run < num_runs; run++) {
        timer.start();
        gpu::backward_gpu(gpu.d_pi, gpu.d_A, gpu.d_mu, gpu.d_L, gpu.d_log_dets,
                          gpu.d_obs, gpu.d_beta, data.T, data.N, data.K);
        times.push_back(timer.stop());
    }
    
    compute_timing_stats(times, result.time_ms, result.time_std_ms, 
                         result.time_min_ms, result.time_max_ms);
    
    result.log_likelihood = 0;
    result.effective_bandwidth_GBs = estimate_hmm_bandwidth(data.T, data.N, data.K, result.time_ms, "backward");
    result.theoretical_occupancy = 0;
    result.gpu_memory_used_bytes = get_gpu_memory_used();
    
    return result;
}

GPUProfileResult profile_viterbi(const BenchmarkData& data, GPUData& gpu, int num_runs = 100) {
    GPUProfileResult result;
    result.dataset_name = data.name;
    result.algo_name = "viterbi";
    result.T = data.T;
    result.N = data.N;
    result.K = data.K;
    result.iterations = 1;
    get_gpu_info(result);
    
    gpu::viterbi_score_gpu(gpu.d_pi, gpu.d_A, gpu.d_mu, gpu.d_L, gpu.d_log_dets,
                          gpu.d_obs, gpu.d_path, data.T, data.N, data.K);
    cudaDeviceSynchronize();
    
    std::vector<float> times;
    CudaTimer timer;
    
    for (int run = 0; run < num_runs; run++) {
        timer.start();
        gpu::viterbi_score_gpu(gpu.d_pi, gpu.d_A, gpu.d_mu, gpu.d_L, gpu.d_log_dets,
                              gpu.d_obs, gpu.d_path, data.T, data.N, data.K);
        times.push_back(timer.stop());
    }
    
    compute_timing_stats(times, result.time_ms, result.time_std_ms, 
                         result.time_min_ms, result.time_max_ms);
    
    std::vector<float> d_delta(data.T * data.N);
    cudaMemcpy(d_delta.data(), gpu.d_path, data.T * data.N * sizeof(float), cudaMemcpyDeviceToHost);
    float ll = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < data.N; i++) {
        ll = cpu::linalg::log_sum_exp(ll, d_delta[(data.T - 1) * data.N + i]);
    }

    result.log_likelihood = ll;
    result.effective_bandwidth_GBs = estimate_hmm_bandwidth(data.T, data.N, data.K, result.time_ms, "viterbi");
    result.theoretical_occupancy = 0;
    result.gpu_memory_used_bytes = get_gpu_memory_used();
    
    return result;
}

GPUProfileResult profile_baum_welch(const BenchmarkData& data, GPUData& gpu, 
                                    int max_iter = 100, int num_runs = 5) {
    GPUProfileResult result;
    result.dataset_name = data.name;
    result.algo_name = "baum_welch_" + std::to_string(max_iter);
    result.T = data.T;
    result.N = data.N;
    result.K = data.K;
    result.iterations = max_iter;
    get_gpu_info(result);
    
    std::vector<float> times;
    float final_ll = 0;
    
    float tolerance = 1e-5;

    for (int run = 0; run < num_runs; run++) {
        // Reset parameters each run
        std::vector<float> L, log_dets, log_pi, log_A;
        prepare_params(data, L, log_dets, log_pi, log_A);
        gpu.upload(data, L, log_dets, log_pi, log_A);
        
        // Warmup
        if (run == 0) {
            gpu::baum_welch_step_gpu(gpu.d_pi, gpu.d_A, gpu.d_mu, gpu.d_L, 
                                     gpu.d_log_dets, gpu.d_Sigma, gpu.d_obs,
                                     data.T, data.N, data.K);
            cudaDeviceSynchronize();
            
            // Re-upload after warmup
            gpu.upload(data, L, log_dets, log_pi, log_A);
        }
        
        float prev_ll = -std::numeric_limits<float>::infinity();

        CudaTimer timer;
        timer.start();
        
        for (int iter = 0; iter < max_iter; iter++) {
            final_ll = gpu::baum_welch_step_gpu(gpu.d_pi, gpu.d_A, gpu.d_mu, gpu.d_L, 
                                                gpu.d_log_dets, gpu.d_Sigma, gpu.d_obs,
                                                data.T, data.N, data.K);

            if (iter > 0 && std::abs(final_ll - prev_ll) < tolerance) {
                result.log_likelihood = final_ll;
                break;
            }

            prev_ll = final_ll;
        }
        
        times.push_back(timer.stop());
    }
    
    compute_timing_stats(times, result.time_ms, result.time_std_ms, 
                         result.time_min_ms, result.time_max_ms);
    
    result.log_likelihood = final_ll;
    result.effective_bandwidth_GBs = estimate_hmm_bandwidth(data.T, data.N, data.K, 
                                                            result.time_ms / max_iter, "baum_welch");
    result.theoretical_occupancy = 0;
    result.gpu_memory_used_bytes = get_gpu_memory_used();
    
    return result;
}

GPUProfileResult profile_baum_welch_convergence(const BenchmarkData& data, GPUData& gpu,
                                                float tolerance = 1e-7f, int max_iter = 100) {
    GPUProfileResult result;
    result.dataset_name = data.name;
    result.algo_name = "baum_welch_converge";
    result.T = data.T;
    result.N = data.N;
    result.K = data.K;
    get_gpu_info(result);
    
    // Reset parameters
    std::vector<float> L, log_dets, log_pi, log_A;
    prepare_params(data, L, log_dets, log_pi, log_A);
    gpu.upload(data, L, log_dets, log_pi, log_A);
    
    CudaTimer timer;
    timer.start();
    
    float prev_ll = -std::numeric_limits<float>::infinity();
    int iter = 0;
    float ll = 0;
    
    for (iter = 0; iter < max_iter; iter++) {
        ll = gpu::baum_welch_step_gpu(gpu.d_pi, gpu.d_A, gpu.d_mu, gpu.d_L, 
                                      gpu.d_log_dets, gpu.d_Sigma, gpu.d_obs,
                                      data.T, data.N, data.K);
        
        if (iter > 0 && std::abs(ll - prev_ll) < tolerance) {
            break;
        }
        prev_ll = ll;
    }
    
    result.time_ms = timer.stop();
    result.time_std_ms = 0;
    result.time_min_ms = result.time_ms;
    result.time_max_ms = result.time_ms;
    result.iterations = iter + 1;
    result.log_likelihood = ll;
    result.effective_bandwidth_GBs = estimate_hmm_bandwidth(data.T, data.N, data.K, 
                                                            result.time_ms / result.iterations, "baum_welch");
    result.theoretical_occupancy = 0;
    result.gpu_memory_used_bytes = get_gpu_memory_used();
    
    return result;
}

// =============================================================================
// MAIN
// =============================================================================

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <data_dir> <output_prefix> [mode]\n";
    std::cout << "\n";
    std::cout << "Arguments:\n";
    std::cout << "  data_dir      Directory containing benchmark data (from generate_benchmark_data.py)\n";
    std::cout << "  output_prefix Prefix for output files (e.g., results/gpu_benchmark)\n";
    std::cout << "  mode          Optional: 'all', 'scaling', 'convergence', 'quick' (default: all)\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog << " data/bench results/gpu_benchmark\n";
    std::cout << "  " << prog << " data/bench results/gpu_scaling scaling\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string data_dir = argv[1];
    std::string output_prefix = argv[2];
    std::string mode = (argc > 3) ? argv[3] : "all";
    
    std::cout << "\n";
    print_gpu_info();
    std::cout << "\n";
    
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         GPU PROFILING - HMM ALGORITHMS (Hassan et al.)       ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Data dir: " << std::left << std::setw(47) << data_dir << "║\n";
    std::cout << "║  Output:   " << std::setw(47) << output_prefix << "║\n";
    std::cout << "║  Mode:     " << std::setw(47) << mode << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    // Find all datasets
    std::vector<std::string> dataset_bases;
    
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        std::string path = entry.path().string();
        if (path.find("_obs.bin") != std::string::npos) {
            std::string base = path.substr(0, path.find("_obs.bin"));
            
            // Filter based on mode
            if (mode == "quick") {
                BenchmarkData tmp;
                std::ifstream fdims(base + "_dims.txt");
                fdims >> tmp.T >> tmp.N >> tmp.K;
               
            }
            
            dataset_bases.push_back(base);
        }
    }
    
    std::sort(dataset_bases.begin(), dataset_bases.end());
    
    std::cout << "Found " << dataset_bases.size() << " datasets:\n";
    for (const auto& base : dataset_bases) {
        std::cout << "  - " << fs::path(base).filename().string() << "\n";
    }
    std::cout << "\n";
    
    std::vector<GPUProfileResult> all_results;
    
    // Profile each dataset
    for (const auto& base_path : dataset_bases) {
        BenchmarkData data;
        if (!load_benchmark_data(base_path, data)) {
            std::cerr << "Failed to load: " << base_path << std::endl;
            continue;
        }
        
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "Profiling: " << data.name << " (T=" << data.T 
                  << ", N=" << data.N << ", K=" << data.K << ")\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        
        // Prepare GPU data
        GPUData gpu;
        gpu.allocate(data.T, data.N, data.K);
        
        std::vector<float> L, log_dets, log_pi, log_A;
        prepare_params(data, L, log_dets, log_pi, log_A);
        gpu.upload(data, L, log_dets, log_pi, log_A);
        
        // Adjust iterations based on size
        int num_runs = (data.T > 50000) ? 100 : 135;
        int bw_runs = (data.T > 20000) ? 100 : 120;
        
        if (mode == "all" || mode == "scaling" || mode == "quick") {
            // Forward
            std::cout << "  [Forward] " << std::flush;
            GPUProfileResult r_fwd = profile_forward(data, gpu, num_runs);
            std::cout << r_fwd.time_ms << " ms (±" << r_fwd.time_std_ms << ") | " 
                      << r_fwd.effective_bandwidth_GBs << " GB/s\n";
            all_results.push_back(r_fwd);
            
            // Backward
            std::cout << "  [Backward] " << std::flush;
            GPUProfileResult r_bwd = profile_backward(data, gpu, num_runs);
            std::cout << r_bwd.time_ms << " ms (±" << r_bwd.time_std_ms << ") | " 
                      << r_bwd.effective_bandwidth_GBs << " GB/s\n";
            all_results.push_back(r_bwd);
            
            // Viterbi
            std::cout << "  [Viterbi] " << std::flush;
            GPUProfileResult r_vit = profile_viterbi(data, gpu, num_runs);
            std::cout << r_vit.time_ms << " ms (±" << r_vit.time_std_ms << ") | " 
                      << r_vit.effective_bandwidth_GBs << " GB/s\n";
            all_results.push_back(r_vit);
            
            // Baum-Welch 10 iterations
            std::cout << "  [Baum-Welch 10 iter] " << std::flush;
            GPUProfileResult r_bw10 = profile_baum_welch(data, gpu, 100, 5);
            std::cout << r_bw10.time_ms << " ms (±" << r_bw10.time_std_ms << ") | LL=" 
                      << r_bw10.log_likelihood << "\n";
            all_results.push_back(r_bw10);
        }
        
        if (mode == "all" || mode == "convergence") {
            // Only on validation dataset
            if (data.name.find("validation") != std::string::npos || 
                data.name.find("scaling_T_1000") != std::string::npos) {
                
                std::cout << "  [Baum-Welch Convergence] " << std::flush;
                GPUProfileResult r_conv = profile_baum_welch_convergence(data, gpu, 1e-4f, 10);
                std::cout << r_conv.time_ms << " ms (" << r_conv.iterations << " iters, LL=" 
                          << r_conv.log_likelihood << ")\n";
                all_results.push_back(r_conv);
            }
        }
        
        gpu.free();
        std::cout << "\n";
    }
    
    // Export results
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "                       EXPORT RESULTS                          \n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    
    // Create output directory if needed
    fs::path output_path(output_prefix);
    if (output_path.has_parent_path()) {
        fs::create_directories(output_path.parent_path());
    }
    
    export_gpu_csv(all_results, output_prefix + "_results.csv");
    export_gpu_json(all_results, output_prefix + "_results.json");
    
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    PROFILING COMPLETE                        ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Total tests: " << std::setw(45) << all_results.size() << "║\n";
    std::cout << "║  CSV output:  " << std::setw(45) << (output_prefix + "_results.csv") << "║\n";
    std::cout << "║  JSON output: " << std::setw(45) << (output_prefix + "_results.json") << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "Next steps:\n";
    std::cout << "  Analyze with Python:\n";
    std::cout << "    python analyze_benchmark_results.py --gpu-csv " << output_prefix << "_results.csv\n";
    std::cout << "\n";
    
    return 0;
}
