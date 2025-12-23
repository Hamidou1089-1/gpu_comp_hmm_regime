#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include "test_utils.hpp"
#include "low_level_linear_algebra.hpp"
#include "algo_hmm.hpp"

struct BenchResult {
    int N;
    double time_ms;
};

// Fonction helper pour le benchmark
double run_cholesky_bench(int size) {
    std::vector<float> Sigma(size * size);
    std::vector<float> L(size * size);
    generate_random_pd_matrix(Sigma.data(), size);

    auto start = std::chrono::high_resolution_clock::now();
    bool success = choleskyDecomposition(Sigma.data(), L.data(), size);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (!success) std::cerr << "Erreur: Cholesky a echoue pour N=" << size << std::endl;

    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}

void run_complexity_analysis() {
    std::cout << "\n=======================================================\n";
    std::cout << "   BENCHMARK COMPLEXITE CHOLESKY (O(N^3)) \n";
    std::cout << "=======================================================\n";
    std::cout << std::setw(10) << "Taille(N)" 
              << std::setw(15) << "Temps(ms)" 
              << std::setw(20) << "Ratio (T/N^3)" 
              << std::endl;
    std::cout << "-------------------------------------------------------\n";

    std::vector<BenchResult> results;
    
    // On teste de 100 a 1000 par pas de 100
    for (int n = 100; n <= 1000; n += 100) {
        double t = run_cholesky_bench(n);
        results.push_back({n, t});

        // Calcul du ratio : Temps / N^3
        // On multiplie par 1e6 pour avoir un chiffre lisible
        double ratio = t / (pow(n, 3)) * 1e6;

        std::cout << std::setw(10) << n 
                  << std::setw(15) << std::fixed << std::setprecision(2) << t 
                  << std::setw(20) << std::setprecision(4) << ratio 
                  << std::endl;
    }

    // Export CSV
    std::ofstream file("benchmark_results.csv");
    file << "N,Time_ms\n";
    for (const auto& r : results) {
        file << r.N << "," << r.time_ms << "\n";
    }
    file.close();
    
    std::cout << "\n[INFO] Resultats exportes dans 'build/benchmark_results.csv'" << std::endl;
    std::cout << "[INFO] Si le Ratio se stabilise, la complexite est bien cubique." << std::endl;
}

int main() {
    run_complexity_analysis();
    return 0;
}