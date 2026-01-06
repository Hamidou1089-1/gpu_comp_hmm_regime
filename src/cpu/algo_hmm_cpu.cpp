#include "algo_hmm_cpu.hpp"
#include "linalg_cpu.hpp"
#include <cmath>
#include <limits>
#include <cstring>

namespace hmm {
namespace cpu {
namespace algo {

using namespace linalg;

// ============================================================================
// Potentials (Hassan formulation)
// ============================================================================

void compute_log_gaussian_potentials(
    const HMMModel& model,
    const float* observations,
    float* log_potentials,
    float* workspace
) {
    const int N = model.N;
    const int K = model.K;
    const int T = model.T;
    
    // t=1: ψ₁(x₁) = log π(x₁) + log p(y₁|x₁)
    const float* y_1 = observations;
    for (int i = 0; i < N; i++) {
        const float* mu_i = model.mu + i * K;
        const float* L_i = model.L + i * K * K;
        
        float log_pdf = log_multivariate_normal_pdf(
            y_1, mu_i, L_i, model.log_det[i], K, workspace
        );
        
        log_potentials[i] = std::log(model.pi[i]) + log_pdf;
    }
    
    // t=2..T: ψₜ(xₜ₋₁,xₜ) = log A(xₜ₋₁,xₜ) + log p(yₜ|xₜ)
    for (int t = 2; t <= T; t++) {
        const float* y_t = observations + (t - 1) * K;
        int base_idx = N + (t - 2) * N * N;
        
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                const float* mu_j = model.mu + j * K;
                const float* L_j = model.L + j * K * K;
                
                float log_pdf = log_multivariate_normal_pdf(
                    y_t, mu_j, L_j, model.log_det[j], K, workspace
                );
                
                log_potentials[base_idx + i * N + j] = 
                    std::log(model.A[i * N + j]) + log_pdf;
            }
        }
    }
}

// ============================================================================
// Forward Algorithm
// ============================================================================

float forward_algorithm(
    const float* log_potentials,
    float* alpha,
    int T,
    int N
) {
    // t=1: α₁(i) = ψ₁(i)
    for (int i = 0; i < N; i++) {
        alpha[i] = log_potentials[i];
    }
    
    // t=2..T: αₜ(j) = log-sum-exp_i [αₜ₋₁(i) + ψₜ(i,j)]
    for (int t = 2; t <= T; t++) {
        int pot_base = N + (t - 2) * N * N;
        
        for (int j = 0; j < N; j++) {
            float log_sum = -std::numeric_limits<float>::infinity();
            
            for (int i = 0; i < N; i++) {
                float val = alpha[(t - 2) * N + i] + log_potentials[pot_base + i * N + j];
                log_sum = log_sum_exp(log_sum, val);
            }
            
            alpha[(t - 1) * N + j] = log_sum;
        }
    }
    
    // Log-likelihood
    float log_likelihood = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < N; i++) {
        log_likelihood = log_sum_exp(log_likelihood, alpha[(T - 1) * N + i]);
    }
    
    return log_likelihood;
}

// ============================================================================
// Backward Algorithm
// ============================================================================

void backward_algorithm(
    const float* log_potentials,
    float* beta,
    int T,
    int N
) {
    // t=T: βₜ(i) = 0 (log 1)
    for (int i = 0; i < N; i++) {
        beta[(T - 1) * N + i] = 0.0f;
    }
    
    // t=T-1..1: βₜ(i) = log-sum-exp_j [ψₜ₊₁(i,j) + βₜ₊₁(j)]
    for (int t = T - 1; t >= 1; t--) {
        int pot_base = N + (t - 1) * N * N;
        
        for (int i = 0; i < N; i++) {
            float log_sum = -std::numeric_limits<float>::infinity();
            
            for (int j = 0; j < N; j++) {
                float val = log_potentials[pot_base + i * N + j] + beta[t * N + j];
                log_sum = log_sum_exp(log_sum, val);
            }
            
            beta[(t - 1) * N + i] = log_sum;
        }
    }
}

// ============================================================================
// Posteriors
// ============================================================================

void compute_gamma(
    const float* alpha,
    const float* beta,
    float* gamma,
    int T,
    int N
) {

    float* log_gamma = new float[N];
    for (int t = 0; t < T; t++) {
        
        
        for (int i = 0; i < N; i++) {
            log_gamma[i] = alpha[t * N + i] + beta[t * N + i];
        }
        
        normalize_from_log(log_gamma, gamma + t * N, N);
    }

    delete[] log_gamma;
}

void compute_xi(
    const float* log_potentials,
    const float* alpha,
    const float* beta,
    float* xi,
    int T,
    int N
) {

    float* log_xi = new float[N*N];
    for (int t = 1; t < T; t++) {
        int pot_base = N + (t - 1) * N * N;
        
        
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                log_xi[i * N + j] = alpha[(t - 1) * N + i] +
                                    log_potentials[pot_base + i * N + j] +
                                    beta[t * N + j];
            }
        }
        
        normalize_from_log(log_xi, xi + (t - 1) * N * N, N * N);
    }

    delete[] log_xi;
}

// ============================================================================
// Viterbi
// ============================================================================

float viterbi_algorithm(
    const float* log_potentials,
    int* path,
    int T,
    int N
) {
    float* delta = new float[T * N];
    int* psi = new int[T * N];
    
    // t=1: δ₁(i) = ψ₁(i)
    for (int i = 0; i < N; i++) {
        delta[i] = log_potentials[i];
        psi[i] = 0;
    }
    
    // t=2..T: δₜ(j) = max_i [δₜ₋₁(i) + ψₜ(i,j)]
    for (int t = 2; t <= T; t++) {
        int pot_base = N + (t - 2) * N * N;
        
        for (int j = 0; j < N; j++) {
            float max_val = -std::numeric_limits<float>::infinity();
            int max_idx = 0;
            
            for (int i = 0; i < N; i++) {
                float val = delta[(t - 2) * N + i] + log_potentials[pot_base + i * N + j];
                if (val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
            }
            
            delta[(t - 1) * N + j] = max_val;
            psi[(t - 1) * N + j] = max_idx;
        }
    }
    
    // Best final state
    float best_score = -std::numeric_limits<float>::infinity();
    int best_state = 0;
    for (int i = 0; i < N; i++) {
        if (delta[(T - 1) * N + i] > best_score) {
            best_score = delta[(T - 1) * N + i];
            best_state = i;
        }
    }
    
    // Backtrack
    path[T - 1] = best_state;
    for (int t = T - 2; t >= 0; t--) {
        path[t] = psi[(t + 1) * N + path[t + 1]];
    }
    
    delete[] delta;
    delete[] psi;
    
    return best_score;
}

// ============================================================================
// Baum-Welch
// ============================================================================

float baum_welch_e_step(
    const HMMModel& model,
    const float* observations,
    float* gamma,
    float* xi,
    float* workspace
) {
    const int T = model.T;
    const int N = model.N;
    
    int pot_size = N + (T - 1) * N * N;
    float* log_potentials = new float[pot_size];
    float* alpha = new float[T * N];
    float* beta = new float[T * N];
    
    compute_log_gaussian_potentials(model, observations, log_potentials, workspace);
    
    float log_likelihood = forward_algorithm(log_potentials, alpha, T, N);
    backward_algorithm(log_potentials, beta, T, N);
    
    compute_gamma(alpha, beta, gamma, T, N);
    compute_xi(log_potentials, alpha, beta, xi, T, N);
    
    delete[] log_potentials;
    delete[] alpha;
    delete[] beta;
    
    return log_likelihood;
}



void baum_welch_m_step(
    HMMModel& model,
    const float* observations,
    const float* gamma,
    const float* xi
) {
    const int N = model.N;
    const int T = model.T;
    const int K = model.K;
    
    // Update π
    for (int i = 0; i < N; i++) {
        model.pi[i] = gamma[i];
    }
    
    // ========================================================================
    // Update A (Transition Matrix)
    // ========================================================================
    for (int i = 0; i < N; i++) {
        // Denominator: sum_t γₜ(i) for t=0..T-2
        float denom = 0.0f;
        for (int t = 0; t < T - 1; t++) {
            denom += gamma[t * N + i];
        }
        
        // Numerator: sum_t ξₜ(i,j) for t=0..T-2
        for (int j = 0; j < N; j++) {
            float numer = 0.0f;
            for (int t = 0; t < T - 1; t++) {
                numer += xi[t * N * N + i * N + j];
            }
            model.A[i * N + j] = (denom > 1e-10f) ? numer / denom : 1.0f / N;
        }
    }
    
    // ========================================================================
    // Update μ (Means)
    // ========================================================================
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
            model.mu[i * K + k] = (denom > 1e-10f) ? numer / denom : 0.0f;
        }
    }
    
    // ========================================================================
    // Update Σ (Covariances)
    // ========================================================================
    const float reg = 1e-4f; // 
    
    for (int i = 0; i < N; i++) {
        
        float denom = 0.0f;
        for (int t = 0; t < T; t++) {
            denom += gamma[t * N + i];
        }
        
        float* Sigma_i = model.Sigma + i * K * K;
        std::memset(Sigma_i, 0, K * K * sizeof(float));
        
        for (int t = 0; t < T; t++) {
            float w = gamma[t * N + i];
            const float* mu_i = model.mu + i * K;
            
            for (int k1 = 0; k1 < K; k1++) {
                float diff1 = observations[t * K + k1] - mu_i[k1];
                for (int k2 = 0; k2 < K; k2++) {
                    float diff2 = observations[t * K + k2] - mu_i[k2];
                    Sigma_i[k1 * K + k2] += w * diff1 * diff2;
                }
            }
        }
        
        if (denom > 1e-10f) {
            for (int k = 0; k < K * K; k++) {
                Sigma_i[k] /= denom;
            }
            // ✅ Régularisation Ridge (diagonal + epsilon)
            for (int k = 0; k < K; k++) {
                Sigma_i[k * K + k] += reg;
            }
        } else {
            
            for (int r = 0; r < K; r++) {
                for (int c = 0; c < K; c++) {
                    Sigma_i[r * K + c] = (r == c) ? 1.0f : 0.0f;
                }
            }
        }
    }
    
    
    bool success = precompute_cholesky(model);
    if (!success) {
        // Si Cholesky échoue, réinitialiser avec identité
        for (int i = 0; i < model.N; i++) {
            float* Sigma_i = model.Sigma + i * K * K;
            for (int r = 0; r < K; r++) {
                for (int c = 0; c < K; c++) {
                    Sigma_i[r * K + c] = (r == c) ? 1.5f : 0.0f;
                }
            }
        }
        precompute_cholesky(model); // Refaire (doit marcher sur identité)
    }
}


float baum_welch_train(
    HMMModel& model,
    const float* observations,
    int max_iterations,
    float tolerance,
    float* workspace
) {
    const int T = model.T;
    const int N = model.N;
    
    float* gamma = new float[T * N];
    float* xi = new float[(T - 1) * N * N];
    
    float prev_ll = -std::numeric_limits<float>::infinity();
    
    for (int iter = 0; iter < max_iterations; iter++) {
        float ll = baum_welch_e_step(model, observations, gamma, xi, workspace);
        
        if (iter > 0 && ll - prev_ll < tolerance) {
            delete[] gamma;
            delete[] xi;
            return ll;
        }
        
        baum_welch_m_step(model, observations, gamma, xi);
        prev_ll = ll;
    }
    
    delete[] gamma;
    delete[] xi;
    return prev_ll;
}

// ============================================================================
// Utilities
// ============================================================================

bool precompute_cholesky(HMMModel& model) {
    for (int i = 0; i < model.N; i++) {
        float* L_i = model.L + i * model.K * model.K;
        std::memcpy(L_i, model.Sigma + i * model.K * model.K, model.K * model.K * sizeof(float));
        
        if (!cholesky_decomposition(L_i, model.K)) return false;
        model.log_det[i] = log_det_from_cholesky(L_i, model.K);
    }
    return true;
}

void initialize_random(HMMModel& model, unsigned int seed) {
    for (int i = 0; i < model.N; i++) model.pi[i] = 1.0f / model.N;
    for (int i = 0; i < model.N * model.N; i++) model.A[i] = 1.0f / model.N;
}

bool validate_model(HMMModel& model) {
    if (std::fabs(array_sum(model.pi, model.N) - 1.0f) > 1e-3f) return false;
    
    for (int i = 0; i < model.N; i++) {
        if (std::fabs(array_sum(model.A + i * model.N, model.N) - 1.0f) > 1e-3f) return false;
    }
    
    return precompute_cholesky(model);
}

} // namespace algo
} // namespace cpu
} // namespace hmm