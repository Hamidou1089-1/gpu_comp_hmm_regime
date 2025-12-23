
/**
 * @file algo_hmm.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2025-12-22
 * 
 * @copyright Copyright (c) 2025
 * 
 * 
 * Observations = [r1(t), ..., rK(t)] the returns of my K assets in T horizon of time
 * 
 */

#include "algo_hmm.hpp"
#include <cmath>
#include <cstring>
#include <limits>
#include <iostream>


const float NEG_INF = -std::numeric_limits<float>::infinity();



void precomputeGaussianParams(
    GaussianParams& params,
    const float* mu,
    const float* Sigma,
    int N, int K
) {
    params.N  = N;
    params.K  = K;
    params.mu = new float[N * K];
    params.L  = new float[N * K * K];
    params.logDetSigma = new float[N];  

    memcpy(params.mu, mu, N * K * sizeof(float));

    for (int i = 0; i < N; i++) {
        const float* Sigma_i = Sigma + i * K * K;
        float* L_i = params.L + i * K * K;

        if (!choleskyDecomposition(Sigma_i, L_i, K)) {
            params.logDetSigma[i] = -INFINITY;
        } else {
            params.logDetSigma[i] = logDeterminantCholesky(L_i, K);
        }
    }

}



void freeGaussianParams(GaussianParams& params) {
    delete[] params.mu;
    delete[] params.L;
    delete[] params.logDetSigma;
}


float log_multivariate_normal_pdf_cholesky(
    const float* x,
    const float* mu,
    const float* L,
    float logDetSigma,
    int K,
    float* workspace
) {
    float* diff = workspace;
    float* y = workspace + K;


    for (int i = 0; i < K; i++) {
        diff[i] = x[i] - mu[i];
    }

    forwardSubstitution(L, diff, y, K);

    float mahalanobis_squared = 0.0f;
    for (int i = 0; i < K; i++) {
        mahalanobis_squared += y[i] * y[i];
    }

    const float LOG_2PI = 1.8378770664093454835606594728112f;
    return -0.5f * (K * LOG_2PI + logDetSigma + mahalanobis_squared);
}




float log_sum_exp(float log_a, float log_b) {
    if (std::isinf(log_a) && log_a < 0) return log_b;
    if (std::isinf(log_b) && log_b < 0) return log_a;
    
    if (log_a > log_b) {
        return log_a + log1p(exp(log_b - log_a));
    } else {
        return log_b + log1p(exp(log_a - log_b));
    }
}

float log_sum_exp_array(const float* log_values, int n) {
    float max_val = log_values[0];
    for (int i = 1; i < n; i++) {
        if (log_values[i] > max_val) {
            max_val = log_values[i];
        }
    }
    
    if (std::isinf(max_val) && max_val < 0) {
        return NEG_INF;
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += exp(log_values[i] - max_val);
    }
    
    return max_val + log(sum);
}



void compute_log_gaussian_potentials(
    float*& log_potentials,
    const float* A,
    const GaussianParams& params,
    const float* pi,
    const float* observations,
    int T, int K, int N
) {
    int totalSize = N + (T - 1) * N * N;
    log_potentials = new float[totalSize];
    
    float* workspace = new float[2 * K];
    
    // ψ_1(x_1) = p(y_1 | x_1) × π(x_1)
    for (int i = 0; i < N; i++) {
        const float* y_0 = observations;
        const float* mu_i = params.mu + i * K;
        const float* L_i = params.L + i * K * K;
        
        float log_pdf = log_multivariate_normal_pdf_cholesky(
            y_0, mu_i, L_i, params.logDetSigma[i], K, workspace
        );
        
        log_potentials[i] = log_pdf + log(pi[i]);
    }
    
    // ψ_t(x_{t-1}, x_t) = p(y_t | x_t) × A(x_{t-1}, x_t)
    for (int t = 1; t < T; t++) {
        int matrixIndex = t;
        
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                const float* y_t = observations + t * K;
                const float* mu_j = params.mu + j * K;
                const float* L_j = params.L + j * K * K;
                
                float log_pdf = log_multivariate_normal_pdf_cholesky(
                    y_t, mu_j, L_j, params.logDetSigma[j], K, workspace
                );
                
                int idx = getMatrixElement(matrixIndex, i, j, N, N);
                log_potentials[idx] = log_pdf + log(A[i * N + j]);
            }
        }
    }
    
    delete[] workspace;
}

void freePotentials(float* potentials) {
    delete[] potentials;
}


void forward_algorithm_log(
    float* log_alpha,
    const float* log_potentials,
    int T, int N
) {
    // Initialisation : log α_1(i) = log ψ_1(i)
    for (int i = 0; i < N; i++) {
        log_alpha[0 * N + i] = log_potentials[i];
    }
    
    // Récursion : log α_t(j) = log_sum_exp_i [ log α_{t-1}(i) + log ψ_t(i,j) ]
    float* log_terms = new float[N];
    
    for (int t = 1; t < T; t++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                int psi_idx = getMatrixElement(t, i, j, N, N);
                log_terms[i] = log_alpha[(t-1) * N + i] + log_potentials[psi_idx];
            }
            log_alpha[t * N + j] = log_sum_exp_array(log_terms, N);
        }
    }
    
    delete[] log_terms;
}



void backward_algorithm_log(
    float* log_beta,
    const float* log_potentials,
    int T, int N
) {
    // Initialisation : log β_T(i) = 0 (car β_T(i) = 1)
    for (int i = 0; i < N; i++) {
        log_beta[(T-1) * N + i] = 0.0f;
    }
    
    // Récursion backward : log β_t(i) = log_sum_exp_j [ log ψ_{t+1}(i,j) + log β_{t+1}(j) ]
    float* log_terms = new float[N];
    
    for (int t = T - 2; t >= 0; t--) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int psi_idx = getMatrixElement(t+1, i, j, N, N);
                log_terms[j] = log_potentials[psi_idx] + log_beta[(t+1) * N + j];
            }
            log_beta[t * N + i] = log_sum_exp_array(log_terms, N);
        }
    }
    
    delete[] log_terms;
}



void forward_backward_smoothing(
    float* log_gamma,
    float* log_xi,
    const float* log_alpha,
    const float* log_beta,
    const float* log_potentials,
    int T, int N
) {
    // Calculer log γ_t(i) = log α_t(i) + log β_t(i) - log Z_t
    for (int t = 0; t < T; t++) {
        float* log_unnormalized = new float[N];
        
        // log γ_t(i) non normalisé
        for (int i = 0; i < N; i++) {
            log_unnormalized[i] = log_alpha[t * N + i] + log_beta[t * N + i];
        }
        
        // Normaliser
        float log_Z = log_sum_exp_array(log_unnormalized, N);
        for (int i = 0; i < N; i++) {
            log_gamma[t * N + i] = log_unnormalized[i] - log_Z;
        }
        
        delete[] log_unnormalized;
    }
    
    // Calculer log ξ_t(i,j) = log α_t(i) + log ψ_{t+1}(i,j) + log β_{t+1}(j) - log Z
    for (int t = 0; t < T - 1; t++) {
        float* log_unnormalized = new float[N * N];
        
        // log ξ_t(i,j) non normalisé
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int psi_idx = getMatrixElement(t+1, i, j, N, N);
                log_unnormalized[i * N + j] = 
                    log_alpha[t * N + i] + 
                    log_potentials[psi_idx] + 
                    log_beta[(t+1) * N + j];
            }
        }
        
        // Normaliser
        float log_Z = log_sum_exp_array(log_unnormalized, N * N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                log_xi[t * N * N + i * N + j] = log_unnormalized[i * N + j] - log_Z;
            }
        }
        
        delete[] log_unnormalized;
    }
}



void compute_marginals(
    float* marginals,
    const float* log_gamma,
    int T, int N
) {
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < N; i++) {
            marginals[t * N + i] = exp(log_gamma[t * N + i]);
        }
    }
}


void viterbi_log(
    int* best_path,
    float* log_probability,
    const float* log_potentials,
    int T, int N
) {
    float* log_delta = new float[T * N];
    int* psi = new int[T * N];
    
    // Initialisation : log δ_1(i) = log ψ_1(i)
    for (int i = 0; i < N; i++) {
        log_delta[0 * N + i] = log_potentials[i];
        psi[0 * N + i] = -1;
    }
    
    // Récursion : log δ_t(j) = max_i [ log δ_{t-1}(i) + log ψ_t(i,j) ]
    for (int t = 1; t < T; t++) {
        for (int j = 0; j < N; j++) {
            float max_val = NEG_INF;
            int max_idx = 0;
            
            for (int i = 0; i < N; i++) {
                int psi_idx = getMatrixElement(t, i, j, N, N);
                float val = log_delta[(t-1) * N + i] + log_potentials[psi_idx];
                
                if (val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
            }
            
            log_delta[t * N + j] = max_val;
            psi[t * N + j] = max_idx;
        }
    }
    
    // Terminaison : trouver l'état final optimal
    float max_val = NEG_INF;
    int max_idx = 0;
    for (int i = 0; i < N; i++) {
        if (log_delta[(T-1) * N + i] > max_val) {
            max_val = log_delta[(T-1) * N + i];
            max_idx = i;
        }
    }
    
    *log_probability = max_val;
    
    // Backtracking
    best_path[T-1] = max_idx;
    for (int t = T - 2; t >= 0; t--) {
        best_path[t] = psi[(t+1) * N + best_path[t+1]];
    }
    
    delete[] log_delta;
    delete[] psi;
}


float compute_log_likelihood(const float* log_alpha, int T, int N) {
    const float* log_alpha_T = log_alpha + (T-1) * N;
    return log_sum_exp_array(log_alpha_T, N);
}



void em_algorithm(
    float* A, float* pi, float* mu, float* Sigma,
    const float* observations,
    int T, int K, int N,
    int max_iterations ,
    float tolerance
) {
    float prev_log_likelihood = -std::numeric_limits<float>::infinity();
    
    for (int iter = 0; iter < max_iterations; iter++) {
        
        
        
        GaussianParams params;
        precomputeGaussianParams(params, mu, Sigma, N, K);
        
        float* log_potentials;
        compute_log_gaussian_potentials(log_potentials, A, params, 
                                        pi, observations, T, K, N);
        
        float* log_alpha = new float[T * N];
        float* log_beta = new float[T * N];
        forward_algorithm_log(log_alpha, log_potentials, T, N);
        backward_algorithm_log(log_beta, log_potentials, T, N);
        
        float* log_gamma = new float[T * N];
        float* log_xi = new float[(T-1) * N * N];
        forward_backward_smoothing(log_gamma, log_xi, log_alpha, log_beta, 
                                   log_potentials, T, N);
        
        // Calculer log-vraisemblance pour vérifier convergence
        float log_likelihood = compute_log_likelihood(log_alpha, T, N);
        
        std::cout << "Iteration " << iter << " : log-likelihood = " 
                  << log_likelihood << std::endl;
        
        // Vérifier convergence
        if (std::abs(log_likelihood - prev_log_likelihood) < tolerance) {
            std::cout << "Convergence atteinte après " << iter << " iterations" << std::endl;
            
            // Cleanup
            delete[] log_alpha;
            delete[] log_beta;
            delete[] log_gamma;
            delete[] log_xi;
            freePotentials(log_potentials);
            freeGaussianParams(params);
            break;
        }
        prev_log_likelihood = log_likelihood;
        
        
        float* gamma = new float[T * N];
        float* xi = new float[(T-1) * N * N];
        
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < N; i++) {
                gamma[t * N + i] = exp(log_gamma[t * N + i]);
            }
        }
        
        for (int t = 0; t < T - 1; t++) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    xi[t * N * N + i * N + j] = exp(log_xi[t * N * N + i * N + j]);
                }
            }
        }
        
        
        for (int i = 0; i < N; i++) {
            pi[i] = gamma[0 * N + i];  
        }
        
        
        for (int i = 0; i < N; i++) {
            
            float denom = 0.0f;
            for (int t = 0; t < T - 1; t++) {
                denom += gamma[t * N + i];
            }
            
            
            for (int j = 0; j < N; j++) {
                
                float numer = 0.0f;
                for (int t = 0; t < T - 1; t++) {
                    numer += xi[t * N * N + i * N + j];
                }
                
                A[i * N + j] = numer / (denom + 1e-10f);  
            }
        }
        
        
        for (int i = 0; i < N; i++) {
            
            float denom = 0.0f;
            for (int t = 0; t < T; t++) {
                denom += gamma[t * N + i];
            }
            
            
            for (int k = 0; k < K; k++) {
                
                float numer = 0.0f;
                for (int t = 0; t < T; t++) {
                    numer += gamma[t * N + i] * observations[t * K + k];
                }
            
                mu[i * K + k] = numer / (denom + 1e-10f);
            }
        }
        
        
        for (int i = 0; i < N; i++) {
             
            float denom = 0.0f;
            for (int t = 0; t < T; t++) {
                denom += gamma[t * N + i];
            }
            
            
            float* Sigma_i = Sigma + i * K * K;
            memset(Sigma_i, 0, K * K * sizeof(float));
            
            
            for (int t = 0; t < T; t++) {
               
                float* diff = new float[K];
                for (int k = 0; k < K; k++) {
                    diff[k] = observations[t * K + k] - mu[i * K + k];
                }
                
                
                
                for (int r = 0; r < K; r++) {
                    for (int c = 0; c < K; c++) {
                        Sigma_i[r * K + c] += gamma[t * N + i] * diff[r] * diff[c];
                    }
                }
                
                delete[] diff;
            }
            
            
            for (int r = 0; r < K; r++) {
                for (int c = 0; c < K; c++) {
                    Sigma_i[r * K + c] /= (denom + 1e-10f);
                }
            }
            
            
            for (int k = 0; k < K; k++) {
                Sigma_i[k * K + k] += 1e-6f;
            }
        }
        
        
        delete[] gamma;
        delete[] xi;
        delete[] log_alpha;
        delete[] log_beta;
        delete[] log_gamma;
        delete[] log_xi;
        freePotentials(log_potentials);
        freeGaussianParams(params);
    }
}


void em_algorithm_with_history(
    float* A, float* pi, float* mu, float* Sigma,
    const float* observations,
    int T, int K, int N,
    float* log_likelihood_history,  
    int* num_iterations,           
    int max_iterations,
    float tolerance
) {
    float prev_log_likelihood = -std::numeric_limits<float>::infinity();
    
    for (int iter = 0; iter < max_iterations; iter++) {
        
        // ====================================================================
        // E-STEP
        // ====================================================================
        
        GaussianParams params;
        precomputeGaussianParams(params, mu, Sigma, N, K);
        
        float* log_potentials;
        compute_log_gaussian_potentials(log_potentials, A, params, 
                                        pi, observations, T, K, N);
        
        float* log_alpha = new float[T * N];
        float* log_beta = new float[T * N];
        forward_algorithm_log(log_alpha, log_potentials, T, N);
        backward_algorithm_log(log_beta, log_potentials, T, N);
        
        float* log_gamma = new float[T * N];
        float* log_xi = new float[(T-1) * N * N];
        forward_backward_smoothing(log_gamma, log_xi, log_alpha, log_beta, 
                                   log_potentials, T, N);
        
        // Calculer log-vraisemblance
        float log_likelihood = compute_log_likelihood(log_alpha, T, N);
        
        // ✅ Stocker dans l'historique
        if (log_likelihood_history != nullptr) {
            log_likelihood_history[iter] = log_likelihood;
        }
        
        std::cout << "Iteration " << iter << " : log-likelihood = " 
                  << log_likelihood << std::endl;
        
        // Vérifier convergence
        if (iter > 0 && std::abs(log_likelihood - prev_log_likelihood) < tolerance) {
            std::cout << "Convergence atteinte après " << iter + 1 << " iterations" << std::endl;
            
            // ✅ Retourner le nombre d'itérations
            if (num_iterations != nullptr) {
                *num_iterations = iter + 1;
            }
            
            // Cleanup
            delete[] log_alpha;
            delete[] log_beta;
            delete[] log_gamma;
            delete[] log_xi;
            freePotentials(log_potentials);
            freeGaussianParams(params);
            break;
        }
        prev_log_likelihood = log_likelihood;
        
        // ====================================================================
        // M-STEP
        // ====================================================================
        
        float* gamma = new float[T * N];
        float* xi = new float[(T-1) * N * N];
        
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < N; i++) {
                gamma[t * N + i] = exp(log_gamma[t * N + i]);
            }
        }
        
        for (int t = 0; t < T - 1; t++) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    xi[t * N * N + i * N + j] = exp(log_xi[t * N * N + i * N + j]);
                }
            }
        }
        
        // Mise à jour π
        memcpy(pi, gamma, N * sizeof(float));
        
        // Mise à jour A
        for (int i = 0; i < N; i++) {
            float denom = 0.0f;
            for (int t = 0; t < T - 1; t++) {
                denom += gamma[t * N + i];
            }
            
            for (int j = 0; j < N; j++) {
                float numer = 0.0f;
                for (int t = 0; t < T - 1; t++) {
                    numer += xi[t * N * N + i * N + j];
                }
                A[i * N + j] = numer / (denom + 1e-10f);
            }
        }
        
        // Mise à jour μ
        for (int i = 0; i < N; i++) {
            float denom = 0.0f;
            for (int t = 0; t < T; t++) {
                denom += gamma[t * N + i];
            }
            
            for (int k = 0; k < K; k++) {
                float numer = 0.0f;
                for (int t = 0; t < T; t++) {
                    numer += gamma[t * N + i] * observations[t * K + k];
                }
                mu[i * K + k] = numer / (denom + 1e-10f);
            }
        }
        
        // Mise à jour Σ
        for (int i = 0; i < N; i++) {
            float denom = 0.0f;
            for (int t = 0; t < T; t++) {
                denom += gamma[t * N + i];
            }
            
            float* Sigma_i = Sigma + i * K * K;
            memset(Sigma_i, 0, K * K * sizeof(float));
            
            for (int t = 0; t < T; t++) {
                float* diff = new float[K];
                for (int k = 0; k < K; k++) {
                    diff[k] = observations[t * K + k] - mu[i * K + k];
                }
                
                for (int r = 0; r < K; r++) {
                    for (int c = 0; c < K; c++) {
                        Sigma_i[r * K + c] += gamma[t * N + i] * diff[r] * diff[c];
                    }
                }
                
                delete[] diff;
            }
            
            for (int r = 0; r < K; r++) {
                for (int c = 0; c < K; c++) {
                    Sigma_i[r * K + c] /= (denom + 1e-10f);
                }
            }
            
            // Régularisation
            for (int k = 0; k < K; k++) {
                Sigma_i[k * K + k] += 1e-6f;
            }
        }
        
        // Si on arrive au max sans converger
        if (iter == max_iterations - 1 && num_iterations != nullptr) {
            *num_iterations = max_iterations;
        }
        
        delete[] gamma;
        delete[] xi;
        delete[] log_alpha;
        delete[] log_beta;
        delete[] log_gamma;
        delete[] log_xi;
        freePotentials(log_potentials);
        freeGaussianParams(params);
    }
}