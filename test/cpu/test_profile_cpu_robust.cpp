/**
 * @file test_profile_cpu_robust.cpp
 * @brief Profiling CPU robuste sur datasets binaires générés par generate_benchmark_data.py
 * 
 * Usage:
 *   ./test_profile_cpu_robust <data_dir> <output_prefix>
 *   ./test_profile_cpu_robust data/bench results/cpu_benchmark
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

#include "algo_hmm_cpu.hpp"
#include "linalg_cpu.hpp"
#include "profiling_utils_cpu.hpp"

namespace fs = std::filesystem;
using namespace hmm::cpu;
using namespace hmm::cpu::algo;

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
    
    // Load all binary files
    if (!load_binary(base_path + "_obs.bin", data.observations, data.T * data.K)) {
        std::cerr << "Cannot load observations" << std::endl;
        return false;
    }
    if (!load_binary(base_path + "_pi.bin", data.pi_init, data.N)) {
        std::cerr << "Cannot load pi" << std::endl;
        return false;
    }
    if (!load_binary(base_path + "_A.bin", data.A_init, data.N * data.N)) {
        std::cerr << "Cannot load A" << std::endl;
        return false;
    }
    if (!load_binary(base_path + "_mu.bin", data.mu_init, data.N * data.K)) {
        std::cerr << "Cannot load mu" << std::endl;
        return false;
    }
    if (!load_binary(base_path + "_sigma.bin", data.Sigma_init, data.N * data.K * data.K)) {
        std::cerr << "Cannot load Sigma" << std::endl;
        return false;
    }
    
    return true;
}

// =============================================================================
// HMM MODEL SETUP
// =============================================================================

HMMModel create_model_from_data(const BenchmarkData& data) {
    HMMModel model;
    model.T = data.T;
    model.N = data.N;
    model.K = data.K;
    
    // Allocate
    model.pi = new float[model.N];
    model.A = new float[model.N * model.N];
    model.mu = new float[model.N * model.K];
    model.Sigma = new float[model.N * model.K * model.K];
    model.L = new float[model.N * model.K * model.K];
    model.log_det = new float[model.N];
    
    // Copy initial parameters
    std::memcpy(model.pi, data.pi_init.data(), model.N * sizeof(float));
    std::memcpy(model.A, data.A_init.data(), model.N * model.N * sizeof(float));
    std::memcpy(model.mu, data.mu_init.data(), model.N * model.K * sizeof(float));
    std::memcpy(model.Sigma, data.Sigma_init.data(), model.N * model.K * model.K * sizeof(float));
    
    // Compute Cholesky
    precompute_cholesky(model);
    
    return model;
}

void free_model(HMMModel& model) {
    delete[] model.pi;
    delete[] model.A;
    delete[] model.mu;
    delete[] model.Sigma;
    delete[] model.L;
    delete[] model.log_det;
}

// =============================================================================
// PROFILING FUNCTIONS
// =============================================================================

CPUProfileResult profile_forward(const BenchmarkData& data, int num_runs = 10) {
    CPUProfileResult result;
    result.dataset_name = data.name;
    result.algo_name = "forward";
    result.T = data.T;
    result.N = data.N;
    result.K = data.K;
    result.iterations = 1;
    
    HMMModel model = create_model_from_data(data);
    
    // Allocate
    int pot_size = data.N + (data.T - 1) * data.N * data.N;
    std::vector<float> log_potentials(pot_size);
    std::vector<float> alpha(data.T * data.N);
    std::vector<float> workspace(2 * data.K);
    
    // Precompute potentials
    compute_log_gaussian_potentials(model, data.observations.data(), 
                                    log_potentials.data(), workspace.data());
    
    // Warmup
    forward_algorithm(log_potentials.data(), alpha.data(), data.T, data.N);
    
    // Benchmark
    std::vector<double> times;
    HighResTimer timer;
    
    for (int run = 0; run < num_runs; run++) {
        timer.start();
        float ll = forward_algorithm(log_potentials.data(), alpha.data(), data.T, data.N);
        times.push_back(timer.elapsed_ms());
        result.log_likelihood = ll;
    }
    
    result.time_ms = compute_mean(times);
    result.time_std_ms = compute_std(times, result.time_ms);
    result.memory_kb = get_max_memory_usage_kb();
    
    // Cache metrics initialized to 0 (filled by perf wrapper)
    result.cache_references = 0;
    result.cache_misses = 0;
    result.cache_miss_rate = 0;
    result.L1_dcache_misses = 0;
    result.LLC_misses = 0;
    result.cycles = 0;
    result.instructions = 0;
    result.ipc = 0;
    
    free_model(model);
    return result;
}

CPUProfileResult profile_backward(const BenchmarkData& data, int num_runs = 10) {
    CPUProfileResult result;
    result.dataset_name = data.name;
    result.algo_name = "backward";
    result.T = data.T;
    result.N = data.N;
    result.K = data.K;
    result.iterations = 1;
    
    HMMModel model = create_model_from_data(data);
    
    int pot_size = data.N + (data.T - 1) * data.N * data.N;
    std::vector<float> log_potentials(pot_size);
    std::vector<float> beta(data.T * data.N);
    std::vector<float> workspace(2 * data.K);
    
    compute_log_gaussian_potentials(model, data.observations.data(), 
                                    log_potentials.data(), workspace.data());
    
    backward_algorithm(log_potentials.data(), beta.data(), data.T, data.N);
    
    std::vector<double> times;
    HighResTimer timer;
    
    for (int run = 0; run < num_runs; run++) {
        timer.start();
        backward_algorithm(log_potentials.data(), beta.data(), data.T, data.N);
        times.push_back(timer.elapsed_ms());
    }
    
    result.time_ms = compute_mean(times);
    result.time_std_ms = compute_std(times, result.time_ms);
    result.memory_kb = get_max_memory_usage_kb();
    result.log_likelihood = 0; // Not computed in backward alone
    
    result.cache_references = 0;
    result.cache_misses = 0;
    result.cache_miss_rate = 0;
    result.L1_dcache_misses = 0;
    result.LLC_misses = 0;
    result.cycles = 0;
    result.instructions = 0;
    result.ipc = 0;
    
    free_model(model);
    return result;
}

CPUProfileResult profile_viterbi(const BenchmarkData& data, int num_runs = 10) {
    CPUProfileResult result;
    result.dataset_name = data.name;
    result.algo_name = "viterbi";
    result.T = data.T;
    result.N = data.N;
    result.K = data.K;
    result.iterations = 1;
    
    HMMModel model = create_model_from_data(data);
    
    int pot_size = data.N + (data.T - 1) * data.N * data.N;
    std::vector<float> log_potentials(pot_size);
    std::vector<int> path(data.T);
    std::vector<float> workspace(2 * data.K);
    
    compute_log_gaussian_potentials(model, data.observations.data(), 
                                    log_potentials.data(), workspace.data());
    
    viterbi_algorithm(log_potentials.data(), path.data(), data.T, data.N);
    
    std::vector<double> times;
    HighResTimer timer;
    
    for (int run = 0; run < num_runs; run++) {
        timer.start();
        float score = viterbi_algorithm(log_potentials.data(), path.data(), data.T, data.N);
        times.push_back(timer.elapsed_ms());
        result.log_likelihood = score;
    }
    
    result.time_ms = compute_mean(times);
    result.time_std_ms = compute_std(times, result.time_ms);
    result.memory_kb = get_max_memory_usage_kb();
    
    result.cache_references = 0;
    result.cache_misses = 0;
    result.cache_miss_rate = 0;
    result.L1_dcache_misses = 0;
    result.LLC_misses = 0;
    result.cycles = 0;
    result.instructions = 0;
    result.ipc = 0;
    
    free_model(model);
    return result;
}

CPUProfileResult profile_baum_welch(const BenchmarkData& data, int max_iter = 100, int num_runs = 10) {
    CPUProfileResult result;
    result.dataset_name = data.name;
    result.algo_name = "baum_welch_" + std::to_string(max_iter);
    result.T = data.T;
    result.N = data.N;
    result.K = data.K;
    result.iterations = max_iter;
    
    std::vector<double> times;
    HighResTimer timer;
    
    for (int run = 0; run < num_runs; run++) {
        // Fresh model each run
        HMMModel model = create_model_from_data(data);
        std::vector<float> workspace(2 * data.K);
        
        timer.start();
        float ll = baum_welch_train(model, data.observations.data(), 
                                    max_iter, 1e-4f, workspace.data());
        times.push_back(timer.elapsed_ms());
        result.log_likelihood = ll;
        
        free_model(model);
    }
    
    result.time_ms = compute_mean(times);
    result.time_std_ms = compute_std(times, result.time_ms);
    result.memory_kb = get_max_memory_usage_kb();
    
    result.cache_references = 0;
    result.cache_misses = 0;
    result.cache_miss_rate = 0;
    result.L1_dcache_misses = 0;
    result.LLC_misses = 0;
    result.cycles = 0;
    result.instructions = 0;
    result.ipc = 0;
    
    return result;
}

CPUProfileResult profile_baum_welch_convergence(const BenchmarkData& data, 
                                                 float tolerance = 1e-2f, 
                                                 int max_iter = 100) {
    CPUProfileResult result;
    result.dataset_name = data.name;
    result.algo_name = "baum_welch_converge";
    result.T = data.T;
    result.N = data.N;
    result.K = data.K;
    
    HMMModel model = create_model_from_data(data);
    std::vector<float> workspace(2 * data.K);
    
    HighResTimer timer;
    timer.start();
    
    // Manual iteration to count convergence
    std::vector<float> gamma(data.T * data.N);
    std::vector<float> xi((data.T - 1) * data.N * data.N);
    
    float prev_ll = -std::numeric_limits<float>::infinity();
    int iter = 0;
    
    for (iter = 0; iter < max_iter; iter++) {
        float ll = baum_welch_e_step(model, data.observations.data(), 
                                     gamma.data(), xi.data(), workspace.data());
        
        if (iter > 0 && std::abs(ll - prev_ll) < tolerance) {
            result.log_likelihood = ll;
            break;
        }
        
        baum_welch_m_step(model, data.observations.data(), gamma.data(), xi.data());
        prev_ll = ll;
        result.log_likelihood = ll;
    }
    
    result.time_ms = timer.elapsed_ms();
    result.time_std_ms = 0;
    result.iterations = iter + 1;
    result.memory_kb = get_max_memory_usage_kb();
    
    result.cache_references = 0;
    result.cache_misses = 0;
    result.cache_miss_rate = 0;
    result.L1_dcache_misses = 0;
    result.LLC_misses = 0;
    result.cycles = 0;
    result.instructions = 0;
    result.ipc = 0;
    
    free_model(model);
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
    std::cout << "  output_prefix Prefix for output files (e.g., results/cpu_benchmark)\n";
    std::cout << "  mode          Optional: 'all', 'scaling', 'convergence' (default: all)\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog << " data/bench results/cpu_benchmark\n";
    std::cout << "  " << prog << " data/bench results/cpu_scaling scaling\n";
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
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         CPU PROFILING - HMM ALGORITHMS (Hassan et al.)       ║\n";
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
            dataset_bases.push_back(base);
        }
    }
    
    std::sort(dataset_bases.begin(), dataset_bases.end());
    
    std::cout << "Found " << dataset_bases.size() << " datasets:\n";
    for (const auto& base : dataset_bases) {
        std::cout << "  - " << fs::path(base).filename().string() << "\n";
    }
    std::cout << "\n";
    
    std::vector<CPUProfileResult> all_results;
    
    // Profile each dataset
    for (const auto& base_path : dataset_bases) {
        BenchmarkData data;
        if (!load_benchmark_data(base_path, data)) {
            std::cerr << "Failed to load: " << base_path << std::endl;
            continue;
        }
        
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "Profiling: " << data.name << " (T=" << data.T 
                  << ", N=" << data.N << ", K=" << data.K << ")\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        
        // Adjust iterations based on size
        int num_runs = 40;
        int bw_runs = 40;
        
        if (mode == "all" || mode == "scaling") {
            // Forward
            std::cout << "  [Forward] " << std::flush;
            CPUProfileResult r_fwd = profile_forward(data, num_runs);
            std::cout << r_fwd.time_ms << " ms (±" << r_fwd.time_std_ms << ")\n";
            all_results.push_back(r_fwd);
            
            // Backward
            std::cout << "  [Backward] " << std::flush;
            CPUProfileResult r_bwd = profile_backward(data, num_runs);
            std::cout << r_bwd.time_ms << " ms (±" << r_bwd.time_std_ms << ")\n";
            all_results.push_back(r_bwd);
            
            // Viterbi
            std::cout << "  [Viterbi] " << std::flush;
            CPUProfileResult r_vit = profile_viterbi(data, num_runs);
            std::cout << r_vit.time_ms << " ms (±" << r_vit.time_std_ms << ")\n";
            all_results.push_back(r_vit);
            
            // Baum-Welch 10 iterations
            std::cout << "  [Baum-Welch 10 iter] " << std::flush;
            CPUProfileResult r_bw10 = profile_baum_welch(data, 10, 40);
            std::cout << r_bw10.time_ms << " ms (±" << r_bw10.time_std_ms << ")\n";
            all_results.push_back(r_bw10);
        }
        
        if (mode == "all" || mode == "convergence") {
            // Only on validation dataset
            if (data.name.find("validation") != std::string::npos || 
                data.name.find("scaling_T_1000") != std::string::npos) {
                
                std::cout << "  [Baum-Welch Convergence] " << std::flush;
                CPUProfileResult r_conv = profile_baum_welch_convergence(data, 1e-4f, 10);
                std::cout << r_conv.time_ms << " ms (" << r_conv.iterations << " iters, LL=" 
                          << r_conv.log_likelihood << ")\n";
                all_results.push_back(r_conv);
            }
        }
        
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
    
    export_cpu_csv(all_results, output_prefix + "_results.csv");
    export_cpu_json(all_results, output_prefix + "_results.json");
    
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
    std::cout << "  1. Run with perf for cache metrics:\n";
    std::cout << "     perf stat -e cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses,cycles,instructions \\\n";
    std::cout << "       ./" << argv[0] << " " << data_dir << " " << output_prefix << "\n";
    std::cout << "  2. Analyze with Python:\n";
    std::cout << "     python analyze_benchmark_results.py --cpu-csv " << output_prefix << "_results.csv\n";
    std::cout << "\n";
    
    return 0;
}       