#pragma once

#include <cstddef>

namespace hmm {
namespace cpu {
namespace algo {

// ============================================================================
// HMM Model
// ============================================================================

struct HMMModel {
    int N;              // States
    int K;              // Observation dimension
    int T;              // Sequence length
    
    float* pi;          // Initial [N]
    float* A;           // Transition [N × N]
    float* mu;          // Means [N × K]
    float* Sigma;       // Covariances [N × K × K]
    float* L;           // Cholesky [N × K × K]
    float* log_det;     // Log determinants [N]
};

// ============================================================================
// Potentials (Hassan et al.)
// ============================================================================

/**
 * @brief Compute log-space Gaussian potentials
 * 
 * Returns array with layout:
 * - [0..N-1]: ψ₁(x₁) = log π(x₁) + log p(y₁|x₁)
 * - [N..(N+(T-1)N²-1)]: ψₜ(xₜ₋₁,xₜ) = log A(xₜ₋₁,xₜ) + log p(yₜ|xₜ)
 * 
 * Total size: N + (T-1)×N²
 * 
 * @param model HMM model
 * @param observations [T × K]
 * @param log_potentials Output [N + (T-1)×N²] (allocated by caller)
 * @param workspace [2K] for PDF computation
 */
void compute_log_gaussian_potentials(
    const HMMModel& model,
    const float* observations,
    float* log_potentials,
    float* workspace
);

/**
 * @brief Helper to get index in potential array
 * 
 * @param t Time step (1-indexed: t=1 for first vector, t>1 for matrices)
 * @param i Previous state (for t>1)
 * @param j Current state
 * @param N Number of states
 * @return Index in log_potentials array
 */
inline int get_potential_index(int t, int i, int j, int N) {
    if (t == 1) {
        return i;  // First N elements are ψ₁(x₁)
    } else {
        return N + (t - 2) * N * N + i * N + j;
    }
}

// ============================================================================
// Forward-Backward (Sequential)
// ============================================================================

/**
 * @brief Forward algorithm with potentials
 * 
 * Uses associative operator ⊕ = log-sum-exp
 * 
 * @param log_potentials [N + (T-1)×N²]
 * @param alpha Output [T × N] (log-space)
 * @param T Sequence length
 * @param N Number of states
 * @return Log-likelihood
 */
float forward_algorithm(
    const float* log_potentials,
    float* alpha,
    int T,
    int N
);

/**
 * @brief Backward algorithm with potentials
 */
void backward_algorithm(
    const float* log_potentials,
    float* beta,
    int T,
    int N
);

/**
 * @brief Compute posteriors
 */
void compute_gamma(
    const float* alpha,
    const float* beta,
    float* gamma,
    int T,
    int N
);

/**
 * @brief Compute two-slice posteriors
 */
void compute_xi(
    const float* log_potentials,
    const float* alpha,
    const float* beta,
    float* xi,
    int T,
    int N
);

// ============================================================================
// Viterbi
// ============================================================================

/**
 * @brief Viterbi with potentials (⊕ = max)
 */
float viterbi_algorithm(
    const float* log_potentials,
    int* path,
    int T,
    int N
);

// ============================================================================
// Baum-Welch
// ============================================================================

float baum_welch_e_step(
    const HMMModel& model,
    const float* observations,
    float* gamma,
    float* xi,
    float* workspace
);

void baum_welch_m_step(
    HMMModel& model,
    const float* observations,
    const float* gamma,
    const float* xi
);

float baum_welch_train(
    HMMModel& model,
    const float* observations,
    int max_iterations,
    float tolerance,
    float* workspace
);

// ============================================================================
// Utilities
// ============================================================================

bool precompute_cholesky(HMMModel& model);
void initialize_random(HMMModel& model, unsigned int seed);
bool validate_model(HMMModel& model);

} // namespace algo
} // namespace cpu
} // namespace hmm