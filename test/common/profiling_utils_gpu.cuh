#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>

// =============================================================================
// GPU PROFILING RESULTS STRUCTURE
// =============================================================================

struct GPUProfileResult {
    std::string dataset_name;
    std::string algo_name;
    int T, N, K;
    
    // Timing
    double time_ms;
    double time_std_ms;
    double time_min_ms;
    double time_max_ms;
    
    // Convergence
    double log_likelihood;
    int iterations;
    
    // GPU Metrics
    double effective_bandwidth_GBs;
    float theoretical_occupancy;
    size_t gpu_memory_used_bytes;
    
    // Device info
    std::string device_name;
    int compute_capability_major;
    int compute_capability_minor;
};

// =============================================================================
// CUDA ERROR CHECKING
// =============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// =============================================================================
// GPU DEVICE INFO
// =============================================================================

inline void get_gpu_info(GPUProfileResult& result) {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    result.device_name = props.name;
    result.compute_capability_major = props.major;
    result.compute_capability_minor = props.minor;
}

inline void print_gpu_info() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║ GPU: " << std::left << std::setw(52) << props.name << "║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Compute Capability: " << props.major << "." << props.minor 
              << std::setw(37) << "" << "║\n";
    std::cout << "║ Total Memory: " << std::setw(6) << (props.totalGlobalMem / (1024*1024)) 
              << " MB" << std::setw(35) << "" << "║\n";
    std::cout << "║ SM Count: " << std::setw(4) << props.multiProcessorCount 
              << std::setw(44) << "" << "║\n";
    std::cout << "║ Max Threads/Block: " << std::setw(5) << props.maxThreadsPerBlock 
              << std::setw(34) << "" << "║\n";
    std::cout << "║ Warp Size: " << std::setw(3) << props.warpSize 
              << std::setw(45) << "" << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
}

// =============================================================================
// OCCUPANCY CALCULATION
// =============================================================================

template <typename KernelFunc>
float get_theoretical_occupancy(KernelFunc kernel_func, int block_size, size_t dynamic_smem_size = 0) {
    int max_active_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, 
        kernel_func, 
        block_size, 
        dynamic_smem_size
    );
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    float occupancy = (float)(max_active_blocks * block_size) / props.maxThreadsPerMultiProcessor;
    return occupancy * 100.0f;
}

// =============================================================================
// BANDWIDTH CALCULATION
// =============================================================================

inline double calculate_effective_bandwidth(
    size_t bytes_read, 
    size_t bytes_written, 
    double time_ms
) {
    // Returns GB/s
    double total_bytes = bytes_read + bytes_written;
    return (total_bytes / 1e9) / (time_ms / 1000.0);
}

// Estimation for HMM algorithms
inline double estimate_hmm_bandwidth(int T, int N, int K, double time_ms, const std::string& algo) {
    size_t bytes = 0;
    
    if (algo == "forward" || algo == "backward") {
        // Read: obs (T*K), pi (N), A (N*N), means (N*K), L (N*K*K), log_dets (N)
        // Write: alpha/beta (T*N)
        // Scan workspace: 2 * T*N*N (read/write multiple times in O(log T) passes)
        size_t input_bytes = T * K + N + N * N + N * K + N * K * K + N;
        size_t output_bytes = T * N;
        size_t scan_bytes = 2 * T * N * N * 15; // ~log2(T) passes, read+write
        bytes = (input_bytes + output_bytes) * sizeof(float) + scan_bytes * sizeof(float);
    }
    else if (algo == "viterbi") {
        // Similar to forward but with backtrack
        size_t input_bytes = T * K + N + N * N + N * K + N * K * K + N;
        size_t output_bytes = T * N + T; // delta + path
        size_t scan_bytes = 2 * T * N * N * 15;
        bytes = (input_bytes + output_bytes) * sizeof(float) + scan_bytes * sizeof(float);
    }
    else if (algo == "baum_welch" || algo == "baum_welch_step") {
        // Forward + Backward + Updates
        size_t fb_bytes = 2 * (T * K + N + N * N + N * K + N * K * K + N + T * N);
        size_t scan_bytes = 4 * T * N * N * 15; // 2 scans
        size_t update_bytes = T * N * K * 3; // gamma weighted updates
        bytes = (fb_bytes + update_bytes) * sizeof(float) + scan_bytes * sizeof(float);
    }
    
    return calculate_effective_bandwidth(bytes, 0, time_ms);
}

// =============================================================================
// GPU MEMORY TRACKING
// =============================================================================

inline size_t get_gpu_memory_used() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return total_mem - free_mem;
}

// =============================================================================
// TIMING UTILITIES
// =============================================================================

class CudaTimer {
private:
    cudaEvent_t start_event, stop_event;
    
public:
    CudaTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        cudaEventRecord(start_event);
    }
    
    float stop() {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms;
    }
};

// =============================================================================
// STATISTICS
// =============================================================================

inline void compute_timing_stats(
    const std::vector<float>& times,
    double& mean, double& std_dev, double& min_val, double& max_val
) {
    if (times.empty()) {
        mean = std_dev = min_val = max_val = 0;
        return;
    }
    
    min_val = times[0];
    max_val = times[0];
    double sum = 0;
    
    for (float t : times) {
        sum += t;
        if (t < min_val) min_val = t;
        if (t > max_val) max_val = t;
    }
    
    mean = sum / times.size();
    
    double sum_sq = 0;
    for (float t : times) {
        double diff = t - mean;
        sum_sq += diff * diff;
    }
    std_dev = (times.size() > 1) ? std::sqrt(sum_sq / (times.size() - 1)) : 0;
}

// =============================================================================
// EXPORT FUNCTIONS
// =============================================================================

inline void export_gpu_csv(const std::vector<GPUProfileResult>& results, const std::string& filename) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot write to " << filename << std::endl;
        return;
    }
    
    // Header
    f << "dataset,algo,T,N,K,time_ms,time_std_ms,time_min_ms,time_max_ms,"
      << "log_likelihood,iterations,bandwidth_GBs,occupancy_pct,gpu_mem_MB,device\n";
    
    // Data
    for (const auto& r : results) {
        f << r.dataset_name << ","
          << r.algo_name << ","
          << r.T << "," << r.N << "," << r.K << ","
          << std::fixed << std::setprecision(4) << r.time_ms << ","
          << std::setprecision(4) << r.time_std_ms << ","
          << std::setprecision(4) << r.time_min_ms << ","
          << std::setprecision(4) << r.time_max_ms << ","
          << std::setprecision(6) << r.log_likelihood << ","
          << r.iterations << ","
          << std::setprecision(2) << r.effective_bandwidth_GBs << ","
          << std::setprecision(1) << r.theoretical_occupancy << ","
          << std::setprecision(2) << (r.gpu_memory_used_bytes / (1024.0 * 1024.0)) << ","
          << "\"" << r.device_name << "\"\n";
    }
    
    f.close();
    std::cout << "✓ GPU results saved to " << filename << std::endl;
}

inline void export_gpu_json(const std::vector<GPUProfileResult>& results, const std::string& filename) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot write to " << filename << std::endl;
        return;
    }
    
    f << "{\n  \"gpu_profiling_results\": [\n";
    
    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        f << "    {\n";
        f << "      \"dataset\": \"" << r.dataset_name << "\",\n";
        f << "      \"algo\": \"" << r.algo_name << "\",\n";
        f << "      \"T\": " << r.T << ",\n";
        f << "      \"N\": " << r.N << ",\n";
        f << "      \"K\": " << r.K << ",\n";
        f << "      \"time_ms\": " << std::fixed << std::setprecision(4) << r.time_ms << ",\n";
        f << "      \"time_std_ms\": " << r.time_std_ms << ",\n";
        f << "      \"time_min_ms\": " << r.time_min_ms << ",\n";
        f << "      \"time_max_ms\": " << r.time_max_ms << ",\n";
        f << "      \"log_likelihood\": " << std::setprecision(6) << r.log_likelihood << ",\n";
        f << "      \"iterations\": " << r.iterations << ",\n";
        f << "      \"bandwidth_GBs\": " << std::setprecision(2) << r.effective_bandwidth_GBs << ",\n";
        f << "      \"occupancy_pct\": " << std::setprecision(1) << r.theoretical_occupancy << ",\n";
        f << "      \"gpu_memory_MB\": " << std::setprecision(2) << (r.gpu_memory_used_bytes / (1024.0 * 1024.0)) << ",\n";
        f << "      \"device\": \"" << r.device_name << "\",\n";
        f << "      \"compute_capability\": \"" << r.compute_capability_major << "." << r.compute_capability_minor << "\"\n";
        f << "    }";
        if (i < results.size() - 1) f << ",";
        f << "\n";
    }
    
    f << "  ]\n}\n";
    f.close();
    std::cout << "✓ GPU results saved to " << filename << std::endl;
}

// =============================================================================
// PRINT UTILITIES
// =============================================================================

inline void print_gpu_result(const GPUProfileResult& r) {
    std::cout << "┌──────────────────────────────────────────────────────────┐\n";
    std::cout << "│ " << std::left << std::setw(56) << r.dataset_name << "│\n";
    std::cout << "├──────────────────────────────────────────────────────────┤\n";
    std::cout << "│ Algo: " << std::setw(18) << r.algo_name 
              << " T=" << std::setw(7) << r.T 
              << " N=" << r.N << " K=" << r.K << "        │\n";
    std::cout << "│ Time: " << std::fixed << std::setprecision(3) << std::setw(10) << r.time_ms 
              << " ms (±" << std::setprecision(3) << std::setw(6) << r.time_std_ms << ")              │\n";
    std::cout << "│ LL:   " << std::setprecision(4) << std::setw(12) << r.log_likelihood 
              << " (" << r.iterations << " iters)                   │\n";
    std::cout << "│ BW:   " << std::setprecision(1) << std::setw(8) << r.effective_bandwidth_GBs 
              << " GB/s   Occupancy: " << std::setprecision(1) << std::setw(5) << r.theoretical_occupancy << "%     │\n";
    std::cout << "│ Mem:  " << std::setprecision(1) << std::setw(8) << (r.gpu_memory_used_bytes / (1024.0 * 1024.0)) 
              << " MB                                     │\n";
    std::cout << "└──────────────────────────────────────────────────────────┘\n";
}
