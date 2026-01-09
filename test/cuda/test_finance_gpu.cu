/**
 * @file test_finance_gpu.cu
 * @brief Application Financière : Entraînement et Inférence sur données réelles S&P500
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <filesystem>

#include <cuda_runtime.h>
#include "algo_hmm_gpu.cuh"
#include "hmm_primitives.cuh" 

namespace fs = std::filesystem;

// Structure simple pour les données
struct FinancialData {
    int T, N, K;
    std::vector<float> obs;
};

template<typename T>
void save_bin(const std::string& filename, const std::vector<T>& data) {
    std::ofstream f(filename, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
}

// Chargeur manuel simple
FinancialData load_data(const std::string& bin_path, const std::string& dims_path) {
    FinancialData d;
    std::cout << dims_path << std::endl;
    std::ifstream fdim(dims_path);
    if(!fdim) throw std::runtime_error("Dims file not found");
    fdim >> d.T >> d.N >> d.K;
    
    d.obs.resize(d.T * d.K);
    std::ifstream fbin(bin_path, std::ios::binary);
    fbin.read(reinterpret_cast<char*>(d.obs.data()), d.obs.size() * sizeof(float));
    return d;
}

int main(int argc, char** argv) {
    // if (argc < 3) {
    //     std::cerr << "Usage: " << argv[0] << " <data_dir> <output_dir>" << std::endl;
    //     return 1;
    // }

    std::string data_dir = "../../data/finance";
    std::string out_dir = "../../results/finance";
    std::string bin_path = data_dir + "/sp500_obs.bin";
    std::string dims_path = data_dir + "/sp500_obs_dims.txt";
    
    try {
        std::cout << "--- GPU FINANCIAL HMM ---" << std::endl;
        
        // 1. Load Data
        auto data = load_data(bin_path, dims_path);
        int T = data.T; int N = 3; int K = data.K; // Force N=3 régimes
        std::cout << "Loaded S&P500: T=" << T << ", K=" << K << ", N=" << N << std::endl;

        
        std::vector<float> pi(N, 1.0f/N); // Uniforme
        std::vector<float> A(N*N);
        std::vector<float> mu(N*K);
        std::vector<float> Sigma(N*K*K); // Sera converti en L (Cholesky)
        std::vector<float> L(N*K*K);
        std::vector<float> log_dets(N);

        // Init aléatoire déterministe
        std::mt19937 gen(42);
        std::normal_distribution<float> dmu(0.0f, 0.5f); 
        std::uniform_real_distribution<float> dA(0.1f, 1.0f);

        for(int i=0; i<N*K; ++i) mu[i] = dmu(gen);
        
        // Matrice A (diagonale dominante pour persistance des régimes)
        for(int i=0; i<N; ++i) {
            float row_sum = 0;
            for(int j=0; j<N; ++j) {
                float val = (i==j) ? 5.0f : dA(gen); // Persistance forte
                A[i*N+j] = val;
                row_sum += val;
            }
            for(int j=0; j<N; ++j) A[i*N+j] = std::log(A[i*N+j] / row_sum);
            pi[i] = std::log(pi[i]);
        }

        // Sigma Identité pour commencer
        for(int i=0; i<N; ++i) {
            for(int k=0; k<K; ++k) {
                Sigma[i*K*K + k*K + k] = 1.0f;
                L[i*K*K + k*K + k] = 1.0f;
            }
            log_dets[i] = 0.0f;
        }

        // 3. Allocations GPU
        float *d_obs, *d_pi, *d_A, *d_mu, *d_L, *d_dets, *d_Sigma_buf;
        cudaMalloc(&d_obs, T*K*sizeof(float));
        cudaMalloc(&d_pi, N*sizeof(float));
        cudaMalloc(&d_A, N*N*sizeof(float));
        cudaMalloc(&d_mu, N*K*sizeof(float));
        cudaMalloc(&d_L, N*K*K*sizeof(float));
        cudaMalloc(&d_dets, N*sizeof(float));
        cudaMalloc(&d_Sigma_buf, N*K*K*sizeof(float)); // Buffer temporaire

        cudaMemcpy(d_obs, data.obs.data(), T*K*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pi, pi.data(), N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, A.data(), N*N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mu, mu.data(), N*K*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_L, L.data(), N*K*K*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dets, log_dets.data(), N*sizeof(float), cudaMemcpyHostToDevice);

        // 4. Training (Baum-Welch)
        std::cout << "Training (20 iterations)..." << std::flush;
        auto start_train = std::chrono::high_resolution_clock::now();
        
        for(int i=0; i<20; ++i) {
            hmm::gpu::baum_welch_step_gpu(d_pi, d_A, d_mu, d_L, d_dets, d_Sigma_buf, d_obs, T, N, K);
        }
        cudaDeviceSynchronize();
        
        auto end_train = std::chrono::high_resolution_clock::now();
        double train_ms = std::chrono::duration<double, std::milli>(end_train - start_train).count();
        std::cout << " Done in " << train_ms << " ms (" << train_ms/20.0 << " ms/iter)" << std::endl;

        
        
        std::cout << "Decoding (Viterbi)..." << std::flush;
        auto start_infer = std::chrono::high_resolution_clock::now();

        
        int* d_path;
        cudaMalloc(&d_path, T*sizeof(int));
        
        hmm::gpu::viterbi_path_gpu(d_pi, d_A, d_mu, d_L, d_dets, d_obs, d_path, T, N, K);
        cudaDeviceSynchronize();
        
        auto end_infer = std::chrono::high_resolution_clock::now();
        double infer_ms = std::chrono::duration<double, std::milli>(end_infer - start_infer).count();
        std::cout << " Done in " << infer_ms << " ms" << std::endl;

        // 6. Sauvegarde Résultats
        std::vector<int> h_path(T);
        cudaMemcpy(h_path.data(), d_path, T*sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(mu.data(), d_mu, N*K*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Sigma.data(), d_L, N*K*K*sizeof(float), cudaMemcpyDeviceToHost);

        save_bin(out_dir + "/gpu_states.bin", h_path);
        save_bin(out_dir + "/gpu_means.bin", mu);
        save_bin(out_dir + "/gpu_sigma.bin", Sigma);

        
        // Cleanup
        cudaFree(d_obs); cudaFree(d_pi); cudaFree(d_A); cudaFree(d_mu);
        cudaFree(d_L); cudaFree(d_dets); cudaFree(d_Sigma_buf); cudaFree(d_path);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}