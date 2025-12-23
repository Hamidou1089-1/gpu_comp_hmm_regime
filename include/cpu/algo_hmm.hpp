#pragma once

#include "low_level_linear_algebra.hpp"

struct GaussianParams {
    float* mu;
    float* L;
    float* logDetSigma;
    int N, K;
};

inline int getVectorOffset() {
    return 0;
}

inline int getMatrixOffset(int matrixIndex, int vectorSize, int N) {
    return vectorSize + (matrixIndex - 1) * N * N;
}

inline int getMatrixElement(int matrixIndex, int i, int j, int vectorSize, int N) {
    return getMatrixOffset(matrixIndex, vectorSize, N) + i * N + j;
}

void precomputeGaussianParams(
    GaussianParams& params,
    const float* mu,
    const float* Sigma,
    int N, int K
);

void freeGaussianParams(GaussianParams& params);

float log_multivariate_normal_pdf_cholesky(
    const float* x,
    const float* mu,
    const float* L,
    float logDetSigma,
    int K,
    float* workspace
);

void compute_log_gaussian_potentials(
    float*& log_potentials,
    const float* A,
    const GaussianParams& params,
    const float* pi,
    const float* observations,
    int T, int K, int N
);

void freePotentials(float* potentials);

void forward_algorithm_log(
    float* log_alpha,
    const float* log_potentials,
    int T, int N
);

void backward_algorithm_log(
    float* log_beta,               
    const float* log_potentials,
    int T, int N
);


void forward_backward_smoothing(
    float* log_gamma,              // Sortie : T × N (log marginales)
    float* log_xi,                 // Sortie : (T-1) × N × N (log paires)
    const float* log_alpha,
    const float* log_beta,
    const float* log_potentials,
    int T, int N
);


void compute_marginals(
    float* marginals,              // Sortie : T × N (probabilités)
    const float* log_gamma,
    int T, int N
);

float compute_log_likelihood(const float* log_alpha, int T, int N);


void viterbi_log(
    int* best_path,                // Sortie : T (indices d'états)
    float* log_probability,        // Sortie : log P(chemin optimal)
    const float* log_potentials,
    int T, int N
);
float log_sum_exp(float log_a, float log_b);
float log_sum_exp_array(const float* log_values, int n);


void em_algorithm(
    float* A, float* pi, float* mu, float* Sigma,  // Paramètres à estimer
    const float* observations,                      // Observations
    int T, int K, int N,
    int max_iterations = 100,
    float tolerance = 1e-4
);

void em_algorithm_with_history(
    float* A, float* pi, float* mu, float* Sigma,
    const float* observations,
    int T, int K, int N,
    float* log_likelihood_history,  
    int* num_iterations,           
    int max_iterations = 150,
    float tolerance = 1e-6
);