/**
 * @file test_finance_cpu.cpp
 * @brief Benchmark Finance : Version CPU Optimisée (Baseline C++)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>

#include "algo_hmm_cpu.hpp" // Ton header CPU existant
#include "linalg_cpu.hpp"

// Structure de données simple
struct FinancialData {
    int T, N, K;
    std::vector<float> obs;
};

FinancialData load_data(const std::string& bin_path, const std::string& dims_path) {
    FinancialData d;
    std::ifstream fdim(dims_path);
    if(!fdim) throw std::runtime_error("Dims file not found");
    fdim >> d.T >> d.N >> d.K;
    
    d.obs.resize(d.T * d.K);
    std::ifstream fbin(bin_path, std::ios::binary);
    fbin.read(reinterpret_cast<char*>(d.obs.data()), d.obs.size() * sizeof(float));
    return d;
}

int main(int argc, char** argv) {
    

    std::string data_dir = "data/finance";
    std::string out_dir = "results/finance";
    std::string bin_path = data_dir + "/sp500_obs.bin";
    std::string dims_path = data_dir + "/sp500_obs_dims.txt";
    
    try {
        std::cout << "--- CPU FINANCIAL HMM (Baseline) ---" << std::endl;
        
        // 1. Load Data
        auto data = load_data(bin_path, dims_path);
        int T = data.T; int N = 3; int K = data.K;
        std::cout << "Loaded S&P500: T=" << T << ", K=" << K << ", N=" << N << std::endl;

        // 2. Init Params (Même Seed 42 que GPU pour comparaison équitable)
        hmm::cpu::algo::HMMModel model;
        model.N = N; model.K = K;
        model.T = T;
        model.N = N;
        model.K = K;
        model.pi = new float[N];
        model.A = new float[N * N];
        model.mu = new float[N * K];
        model.Sigma = new float[N * K * K];
        model.L = new float[N * K * K];
        model.log_det = new float[N];

        std::mt19937 gen(42);
        std::normal_distribution<float> d_mu(0.0f, 0.5f);
        std::uniform_real_distribution<float> d_A(0.1f, 1.0f);

        for(int i=0; i<N; ++i) {
            model.pi[i] = 1.0f / N;
        }

        // Initialiser mu
        for(int i=0; i<N*K; ++i) {
            model.mu[i] = d_mu(gen);
        }
        
        
        for(int i=0; i<N; ++i) {
            float row_sum = 0;
            // Générer les valeurs
            for(int j=0; j<N; ++j) {
                float val = (i==j) ? 1.0f : d_A(gen);
                model.A[i*N+j] = val;
                row_sum += val;
            }
             for(int j=0; j<N; ++j) {
                model.A[i*N+j] = model.A[i*N+j] / row_sum;
            }
            


            for(int k1=0; k1<K; ++k1) {
                for(int k2=0; k2<K; ++k2) {
                    model.Sigma[i*K*K + k1*K + k2] = (k1 == k2) ? 1.0f : 0.0f;
                }
            }
        }
        hmm::cpu::algo::precompute_cholesky(model);

        
        
        
        // 3. Training (Baum-Welch CPU)
        std::cout << "Training (100 iterations)..." << std::flush;
        auto start_train = std::chrono::high_resolution_clock::now();
        
        
        
        std::vector<float> workspace_train(2*K);
        float loglikelihood = hmm::cpu::algo::baum_welch_train(
            model, data.obs.data(), 20, 1e-4, workspace_train.data()
        );
        


        
        auto end_train = std::chrono::high_resolution_clock::now();
        double train_ms = std::chrono::duration<double, std::milli>(end_train - start_train).count();
        std::cout << " Done in " << train_ms << " ms (" << train_ms/20.0 << " ms/iter)" << std::endl;
        std::cout << "Final log-likelihood: " << loglikelihood << std::endl;

        // 4. Inference (Viterbi CPU)
        std::cout << "Decoding (Viterbi)..." << std::flush;
        auto start_infer = std::chrono::high_resolution_clock::now();
        
        

        std::vector<float> workspace_infer(2*K);
        std::vector<float> log_potentials(N + (T-1)*N*N); 
        hmm::cpu::algo::compute_log_gaussian_potentials(model, data.obs.data(), log_potentials.data(),workspace_infer.data());

        std::vector<int> path(T);
        float score =  hmm::cpu::algo::viterbi_algorithm(log_potentials.data(), path.data(), T, N);
        
        auto end_infer = std::chrono::high_resolution_clock::now();
        double infer_ms = std::chrono::duration<double, std::milli>(end_infer - start_infer).count();
        std::cout << " Done in " << infer_ms << " ms" << std::endl;
        std::cout << "Viterbi score: " << score << std::endl;

        // 5. Sauvegarde (pour vérifier que CPU et GPU trouvent la même chose)
        std::string out_file = out_dir + "/sp500_path_cpu.bin";
        std::ofstream fout(out_file, std::ios::binary);
        fout.write(reinterpret_cast<char*>(path.data()), T*sizeof(int));
        std::cout << "Path saved to " << out_file << std::endl;


        delete[] model.pi;
        delete[] model.A;
        delete[] model.mu;
        delete[] model.Sigma;
        delete[] model.L;
        delete[] model.log_det;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}