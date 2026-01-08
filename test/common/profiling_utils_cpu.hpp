#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <sys/resource.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>

// =============================================================================
// CPU PROFILING RESULTS STRUCTURE
// =============================================================================

struct CPUProfileResult {
    std::string dataset_name;
    std::string algo_name;
    int T, N, K;
    
    // Timing
    double time_ms;
    double time_std_ms;
    
    // Convergence
    double log_likelihood;
    int iterations;
    
    // Memory
    long memory_kb;
    
    // Cache metrics (filled by perf wrapper script)
    long cache_references;
    long cache_misses;
    double cache_miss_rate;
    long L1_dcache_misses;
    long LLC_misses;
    long cycles;
    long instructions;
    double ipc;  // Instructions per cycle
};

// =============================================================================
// TIMING UTILITIES
// =============================================================================

class HighResTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start_time;
        return elapsed.count();
    }
    
    double elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start_time;
        return elapsed.count();
    }
};

// =============================================================================
// MEMORY UTILITIES
// =============================================================================

inline long get_max_memory_usage_kb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // KB on Linux, bytes on macOS
}

inline long get_current_memory_kb() {
    std::ifstream statm("/proc/self/statm");
    if (!statm.is_open()) return -1;
    
    long size, resident;
    statm >> size >> resident;
    statm.close();
    
    long page_size_kb = sysconf(_SC_PAGESIZE) / 1024;
    return resident * page_size_kb;
}

// =============================================================================
// STATISTICS UTILITIES
// =============================================================================

inline double compute_mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (double v : values) sum += v;
    return sum / values.size();
}

inline double compute_std(const std::vector<double>& values, double mean) {
    if (values.size() < 2) return 0.0;
    double sum_sq = 0.0;
    for (double v : values) {
        double diff = v - mean;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / (values.size() - 1));
}

// =============================================================================
// EXPORT FUNCTIONS
// =============================================================================

inline void export_cpu_csv(const std::vector<CPUProfileResult>& results, const std::string& filename) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot write to " << filename << std::endl;
        return;
    }
    
    // Header
    f << "dataset,algo,T,N,K,time_ms,time_std_ms,log_likelihood,iterations,memory_kb,"
      << "cache_refs,cache_misses,cache_miss_rate,L1_misses,LLC_misses,cycles,instructions,ipc\n";
    
    // Data
    for (const auto& r : results) {
        f << r.dataset_name << ","
          << r.algo_name << ","
          << r.T << "," << r.N << "," << r.K << ","
          << std::fixed << std::setprecision(4) << r.time_ms << ","
          << std::setprecision(4) << r.time_std_ms << ","
          << std::setprecision(6) << r.log_likelihood << ","
          << r.iterations << ","
          << r.memory_kb << ","
          << r.cache_references << ","
          << r.cache_misses << ","
          << std::setprecision(4) << r.cache_miss_rate << ","
          << r.L1_dcache_misses << ","
          << r.LLC_misses << ","
          << r.cycles << ","
          << r.instructions << ","
          << std::setprecision(3) << r.ipc << "\n";
    }
    
    f.close();
    std::cout << "✓ CPU results saved to " << filename << std::endl;
}

inline void export_cpu_json(const std::vector<CPUProfileResult>& results, const std::string& filename) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot write to " << filename << std::endl;
        return;
    }
    
    f << "{\n  \"cpu_profiling_results\": [\n";
    
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
        f << "      \"log_likelihood\": " << std::setprecision(6) << r.log_likelihood << ",\n";
        f << "      \"iterations\": " << r.iterations << ",\n";
        f << "      \"memory_kb\": " << r.memory_kb << ",\n";
        f << "      \"cache_references\": " << r.cache_references << ",\n";
        f << "      \"cache_misses\": " << r.cache_misses << ",\n";
        f << "      \"cache_miss_rate\": " << std::setprecision(4) << r.cache_miss_rate << ",\n";
        f << "      \"L1_dcache_misses\": " << r.L1_dcache_misses << ",\n";
        f << "      \"LLC_misses\": " << r.LLC_misses << ",\n";
        f << "      \"cycles\": " << r.cycles << ",\n";
        f << "      \"instructions\": " << r.instructions << ",\n";
        f << "      \"ipc\": " << std::setprecision(3) << r.ipc << "\n";
        f << "    }";
        if (i < results.size() - 1) f << ",";
        f << "\n";
    }
    
    f << "  ]\n}\n";
    f.close();
    std::cout << "✓ CPU results saved to " << filename << std::endl;
}

// =============================================================================
// PRINT UTILITIES
// =============================================================================

inline void print_cpu_result(const CPUProfileResult& r) {
    std::cout << "┌─────────────────────────────────────────────────────┐\n";
    std::cout << "│ " << std::left << std::setw(51) << r.dataset_name << "│\n";
    std::cout << "├─────────────────────────────────────────────────────┤\n";
    std::cout << "│ Algo: " << std::setw(15) << r.algo_name 
              << " T=" << std::setw(6) << r.T 
              << " N=" << r.N << " K=" << r.K << "       │\n";
    std::cout << "│ Time: " << std::fixed << std::setprecision(2) << std::setw(10) << r.time_ms 
              << " ms (±" << std::setprecision(2) << r.time_std_ms << ")                │\n";
    std::cout << "│ LL:   " << std::setprecision(4) << std::setw(12) << r.log_likelihood 
              << " (" << r.iterations << " iters)              │\n";
    std::cout << "│ Mem:  " << std::setw(10) << r.memory_kb << " KB                          │\n";
    std::cout << "└─────────────────────────────────────────────────────┘\n";
}