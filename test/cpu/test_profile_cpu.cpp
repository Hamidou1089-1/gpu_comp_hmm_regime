// test_profile_cpu.cpp
// Profiling complet des algorithmes HMM CPU


#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <sstream>

#include "test_utils.hpp"
#include "linalg_cpu.hpp"
#include "algo_hmm_cpu.hpp"

using namespace hmm::cpu::linalg;
using namespace hmm::cpu;
using namespace hmm::cpu::algo;

// ==============================================================================
// STRUCTURES DE DONNÉES
// ==============================================================================

struct ProfilingConfig {
    int N;        // Nombre d'états
    int T;        // Pas de temps
    int K;        // Dimensions observations
    std::string label;
};

struct ProfilingResult {
    std::string algo_name;
    std::string config_label;
    int N, T, K;
    double time_ms;
    size_t memory_estimate_mb;
    double flops_estimate;  // Pour certains algos
};

// ==============================================================================
// CONFIGURATIONS DE TEST
// ==============================================================================

std::vector<ProfilingConfig> get_test_configurations() {
    return {
        // Medium
        {3, 500, 50, "medium"},
        {5, 1000, 100, "medium_large"},
        
        // Large
        {3, 2500, 200, "large"},
        {5, 3000, 400, "large_xl"},
        
        // XLarge (extrême)
        {3, 4000, 200, "xlarge"},
        {5, 5000, 400, "xlarge_max"},
        
    };
}

// ==============================================================================
// UTILITAIRES PROFILING
// ==============================================================================

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start_time;
        return elapsed.count();
    }
};

size_t estimate_memory_mb(int N, int T, int K) {
    size_t bytes = 0;
    bytes += N * N * sizeof(float);           // A
    bytes += N * sizeof(float);               // pi
    bytes += N * K * sizeof(float);           // mu
    bytes += N * K * K * sizeof(float);       // Sigma
    bytes += T * N * sizeof(float);           // alpha/beta
    bytes += T * K * sizeof(float);           // observations
    bytes += T * N * sizeof(float);           // log_potentials (vecteur part)
    bytes += (T - 1) * N * N * sizeof(float); // log_potentials (matrices part)
    
    return bytes / (1024 * 1024);
}

// ==============================================================================
// PROFILING INDIVIDUEL PAR ALGO
// ==============================================================================

ProfilingResult profile_cholesky(const ProfilingConfig& cfg) {
    std::cout << "  [Cholesky] N=" << cfg.N << ", K=" << cfg.K << "..." << std::flush;
    
    std::vector<float> Sigma(cfg.K * cfg.K);
    std::vector<float> L(cfg.K * cfg.K);
    generate_random_pd_matrix(Sigma.data(), cfg.K);
    
    Timer timer;
    timer.start();
    
    // Répéter pour avoir un timing stable
    int repeats = std::max(1, 100 / cfg.K);
    for (int r = 0; r < repeats; r++) {
        cholesky_decomposition(Sigma.data(), cfg.K);
    }
    
    double time_ms = timer.elapsed_ms() / repeats;
    
    // Complexité O(K^3)
    double flops = (cfg.K * cfg.K * cfg.K) / 3.0;
    
    std::cout << " " << time_ms << "ms\n";
    
    return {
        "cholesky",
        cfg.label,
        cfg.N, cfg.T, cfg.K,
        time_ms,
        estimate_memory_mb(cfg.N, cfg.T, cfg.K),
        flops
    };
}

ProfilingResult profile_forward(const ProfilingConfig& cfg) {
    std::cout << "  [Forward] N=" << cfg.N << ", T=" << cfg.T << ", K=" << cfg.K << "..." << std::flush;
    
    // Générer données synthétiques
    auto data = generate_synthetic_hmm_sequence(cfg.T, cfg.K, cfg.N);
    
    
    std::vector<float> log_potentials(cfg.N + (cfg.T - 1)*cfg.N*cfg.N);
    std::vector<float> workspace(2*cfg.K);
    compute_log_gaussian_potentials(data.model, data.observations,
                                     log_potentials.data(), workspace.data());
    
    std::vector<float> log_alpha(cfg.T * cfg.N);
    
    Timer timer;
    timer.start();
    forward_algorithm(log_potentials.data(), log_alpha.data(), cfg.T, cfg.N);
    double time_ms = timer.elapsed_ms();
    
    std::cout << " " << time_ms << "ms\n";
    
    
    free_synthetic_data(data);
    
    double flops = (cfg.T * cfg.N * cfg.N) / 1.0;

    return {
        "forward",
        cfg.label,
        cfg.N, cfg.T, cfg.K,
        time_ms,
        estimate_memory_mb(cfg.N, cfg.T, cfg.K),
        flops 
    };
}

ProfilingResult profile_backward(const ProfilingConfig& cfg) {
    std::cout << "  [Backward] N=" << cfg.N << ", T=" << cfg.T << ", K=" << cfg.K << "..." << std::flush;
    
    auto data = generate_synthetic_hmm_sequence(cfg.T, cfg.K, cfg.N);
    
    std::vector<float> log_potentials(cfg.N + (cfg.T - 1)*cfg.N*cfg.N);
    std::vector<float> workspace(2*cfg.K);
    compute_log_gaussian_potentials(data.model, data.observations,
                                     log_potentials.data(), workspace.data());
    
    std::vector<float> log_beta(cfg.T * cfg.N);
    
    Timer timer;
    timer.start();
    backward_algorithm(log_potentials.data(), log_beta.data(), cfg.T, cfg.N);
    double time_ms = timer.elapsed_ms();
    
    std::cout << " " << time_ms << "ms\n";
    
    
    free_synthetic_data(data);
    
    double flops = (cfg.T * cfg.N * cfg.N) / 1.0;

    return {
        "backward",
        cfg.label,
        cfg.N, cfg.T, cfg.K,
        time_ms,
        estimate_memory_mb(cfg.N, cfg.T, cfg.K),
        flops
    };
}

ProfilingResult profile_smoothing(const ProfilingConfig& cfg) {
    std::cout << "  [Smoothing] N=" << cfg.N << ", T=" << cfg.T << ", K=" << cfg.K << "..." << std::flush;
    
    auto data = generate_synthetic_hmm_sequence(cfg.T, cfg.K, cfg.N);
    
    std::vector<float> log_potentials(cfg.N + (cfg.T - 1)*cfg.N*cfg.N);
    std::vector<float> workspace(2*cfg.K);
    compute_log_gaussian_potentials(data.model, data.observations,
                                     log_potentials.data(), workspace.data());
    
    std::vector<float> log_alpha(cfg.T * cfg.N);
    std::vector<float> log_beta(cfg.T * cfg.N);
    std::vector<float> log_gamma(cfg.T * cfg.N);
    std::vector<float> log_xi((cfg.T - 1) * cfg.N * cfg.N);
    
    forward_algorithm(log_potentials.data(), log_alpha.data(), cfg.T, cfg.N);
    backward_algorithm(log_potentials.data(), log_beta.data(), cfg.T, cfg.N);
    
    Timer timer;
    timer.start();
    compute_gamma(log_alpha.data(),
                    log_beta.data(),
                    log_gamma.data(),
                    cfg.T,
                    cfg.N);
    double time_ms = timer.elapsed_ms();
    
    std::cout << " " << time_ms << "ms\n";
    
    
    free_synthetic_data(data);
    
    double flops = (cfg.T * cfg.N * cfg.N) / 1.0;

    return {
        "smoothing",
        cfg.label,
        cfg.N, cfg.T, cfg.K,
        time_ms,
        estimate_memory_mb(cfg.N, cfg.T, cfg.K),
        flops
    };
}

ProfilingResult profile_viterbi(const ProfilingConfig& cfg) {
    std::cout << "  [Viterbi] N=" << cfg.N << ", T=" << cfg.T << ", K=" << cfg.K << "..." << std::flush;
    
    auto data = generate_synthetic_hmm_sequence(cfg.T, cfg.K, cfg.N);
    
    std::vector<float> log_potentials(cfg.N + (cfg.T - 1)*cfg.N*cfg.N);
    std::vector<float> workspace(2*cfg.K);
    compute_log_gaussian_potentials(data.model, data.observations,
                                     log_potentials.data(), workspace.data());
    
    std::vector<int> best_path(cfg.T);
    
    
    Timer timer;
    timer.start();
    viterbi_algorithm(log_potentials.data(), best_path.data(), cfg.T, cfg.N);
    double time_ms = timer.elapsed_ms();
    
    std::cout << " " << time_ms << "ms\n";
    
    
    free_synthetic_data(data);
    
    double flops = (cfg.T * cfg.N * cfg.N) / 1.0;

    return {
        "viterbi",
        cfg.label,
        cfg.N, cfg.T, cfg.K,
        time_ms,
        estimate_memory_mb(cfg.N, cfg.T, cfg.K),
        flops
    };
}

ProfilingResult profile_em_fixed_iters(const ProfilingConfig& cfg, int num_iters = 100) {
    std::cout << "  [EM " << num_iters << " iters] N=" << cfg.N << ", T=" << cfg.T 
              << ", K=" << cfg.K << "..." << std::flush;
    
    auto data = generate_synthetic_hmm_sequence(cfg.T, cfg.K, cfg.N);
    
    // Paramètres à estimer (initialisation aléatoire)
    std::vector<float> A(cfg.N * cfg.N);
    std::vector<float> pi(cfg.N);
    std::vector<float> mu(cfg.N * cfg.K);
    std::vector<float> Sigma(cfg.N * cfg.K * cfg.K);
    std::vector<float> workspace(2*cfg.K);
    
    generate_transition_matrix(A.data(), cfg.N, 0.7f);
    generate_initial_distribution(pi.data(), cfg.N);
    generate_separated_means(mu.data(), cfg.N, cfg.K, 3.0f);
    generate_covariance_matrices(Sigma.data(), cfg.N, cfg.K, 1.0f, 0.3f);


    
    Timer timer;
    timer.start();
    
    // EM avec nombre fixe d'itérations
    baum_welch_train(data.model, data.observations, num_iters, 1e-7, workspace.data());
    
    double time_ms = timer.elapsed_ms();
    
    std::cout << " " << time_ms << "ms (" << (time_ms / num_iters) << "ms/iter)\n";
    
    free_synthetic_data(data);
    
    double flops = ( num_iters * cfg.T * cfg.N * cfg.N * cfg.K) / 1.0;

    return {
        "em_" + std::to_string(num_iters) + "_iters",
        cfg.label,
        cfg.N, cfg.T, cfg.K,
        time_ms,
        estimate_memory_mb(cfg.N, cfg.T, cfg.K),
        flops 
    };
}

// ==============================================================================
// EXPORT RÉSULTATS
// ==============================================================================

void export_csv(const std::vector<ProfilingResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Erreur : impossible d'ouvrir " << filename << std::endl;
        return;
    }
    
    // Header
    file << "algo,config,N,T,K,time_ms,memory_mb,flops_estimate\n";
    
    // Data
    for (const auto& r : results) {
        file << r.algo_name << ","
             << r.config_label << ","
             << r.N << ","
             << r.T << ","
             << r.K << ","
             << std::fixed << std::setprecision(6) << r.time_ms << ","
             << r.memory_estimate_mb << ","
             << std::scientific << r.flops_estimate << "\n";
    }
    
    file.close();
    std::cout << "\n[INFO] Résultats exportés : " << filename << std::endl;
}

void export_json(const std::vector<ProfilingResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Erreur : impossible d'ouvrir " << filename << std::endl;
        return;
    }
    
    file << "{\n";
    file << "  \"profiling_results\": [\n";
    
    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        file << "    {\n";
        file << "      \"algo\": \"" << r.algo_name << "\",\n";
        file << "      \"config\": \"" << r.config_label << "\",\n";
        file << "      \"N\": " << r.N << ",\n";
        file << "      \"T\": " << r.T << ",\n";
        file << "      \"K\": " << r.K << ",\n";
        file << "      \"time_ms\": " << std::fixed << std::setprecision(6) << r.time_ms << ",\n";
        file << "      \"memory_mb\": " << r.memory_estimate_mb << ",\n";
        file << "      \"flops_estimate\": " << std::scientific << r.flops_estimate << "\n";
        file << "    }";
        if (i < results.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "  ]\n";
    file << "}\n";
    
    file.close();
    std::cout << "[INFO] Résultats exportés : " << filename << std::endl;
}

// ==============================================================================
// MAIN
// ==============================================================================

int main(int argc, char* argv[]) {
    bool export_csv_flag = (argc > 1 && std::string(argv[1]) == "export_csv");
    bool export_json_flag = (argc > 1 && std::string(argv[1]) == "export_json");
    
    std::cout << "\n";
    std::cout << "=======================================================\n";
    std::cout << "   PROFILING CPU - ALGORITHMES HMM\n";
    std::cout << "=======================================================\n\n";
    
    auto configs = get_test_configurations();
    std::vector<ProfilingResult> all_results;
    
    // Profiling de chaque algo sur toutes les configs
    std::cout << "--- CHOLESKY DECOMPOSITION ---\n";
    for (const auto& cfg : configs) {
        all_results.push_back(profile_cholesky(cfg));
    }
    
    std::cout << "\n--- FORWARD ALGORITHM ---\n";
    for (const auto& cfg : configs) {
        all_results.push_back(profile_forward(cfg));
    }
    
    std::cout << "\n--- BACKWARD ALGORITHM ---\n";
    for (const auto& cfg : configs) {
        all_results.push_back(profile_backward(cfg));
    }
    
    std::cout << "\n--- SMOOTHING (Forward-Backward) ---\n";
    for (const auto& cfg : configs) {
        all_results.push_back(profile_smoothing(cfg));
    }
    
    std::cout << "\n--- VITERBI ALGORITHM ---\n";
    for (const auto& cfg : configs) {
        all_results.push_back(profile_viterbi(cfg));
    }
    
    std::cout << "\n--- EM ALGORITHM (10 iterations) ---\n";
    // Seulement pour configs petites/moyennes (EM est coûteux)
    for (size_t i = 0; i < std::min(size_t(6), configs.size()); i++) {
        all_results.push_back(profile_em_fixed_iters(configs[i], 10));
    }
    
    // Export
    if (export_csv_flag || export_json_flag) {
        if (export_csv_flag) {
            export_csv(all_results, "profiling_results.csv");
        }
        if (export_json_flag) {
            export_json(all_results, "profiling_results.json");
        }
    } else {
        export_csv(all_results, "profiling_results.csv");
        export_json(all_results, "profiling_results.json");
    }
    
    std::cout << "\n=======================================================\n";
    std::cout << "   PROFILING TERMINÉ\n";
    std::cout << "=======================================================\n";
    std::cout << "\nUtilisez 'perf record/report' pour analyse détaillée.\n";
    std::cout << "Exemple: perf record -g ./test_profile_cpu\n";
    std::cout << "         perf report -g 'graph,0.5,caller'\n\n";
    
    return 0;
}