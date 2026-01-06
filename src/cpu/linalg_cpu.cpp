#include "linalg_cpu.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace hmm {
namespace cpu {
namespace linalg {

// ============================================================================
// BLAS Level 1: Vector Operations
// ============================================================================

float dot_product(const float* x, const float* y, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

void element_wise_mult(const float* x, const float* y, float* z, int N) {
    for (int i = 0; i < N; i++) {
        z[i] = x[i] * y[i];
    }
}

void element_wise_add(const float* x, const float* y, float* z, int N) {
    for (int i = 0; i < N; i++) {
        z[i] = x[i] + y[i];
    }
}

void vector_copy(const float* x, float* y, int N) {
    for (int i = 0; i < N; i++) {
        y[i] = x[i];
    }
}

void vector_scale(float alpha, const float* x, float* y, int N) {
    for (int i = 0; i < N; i++) {
        y[i] = alpha * x[i];
    }
}

// ============================================================================
// BLAS Level 2: Matrix-Vector Operations
// ============================================================================

void matrix_vector_mult(const float* A, const float* x, float* y, int M, int N) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

void matrix_vector_mult_transpose(const float* A, const float* x, float* y, int M, int N) {
    // y = A^T * x, where A is [M × N], x is [M], y is [N]
    for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int i = 0; i < M; i++) {
            sum += A[i * N + j] * x[i];
        }
        y[j] = sum;
    }
}

// ============================================================================
// BLAS Level 3: Matrix-Matrix Operations
// ============================================================================

void matrix_matrix_mult(const float* A, const float* B, float* C, int M, int N, int K) {
    // C = A * B, where A is [M × K], B is [K × N], C is [M × N]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Specialized Operations for HMM
// ============================================================================

bool cholesky_decomposition(float* A, int N) {
    // Cholesky-Banachiewicz algorithm
    // A = L * L^T, where L is lower triangular
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = A[i * N + j];
            
            for (int k = 0; k < j; k++) {
                sum -= A[i * N + k] * A[j * N + k];
            }
            
            if (i == j) {
                if (sum <= 0.0f) {
                    return false;  // Not positive-definite
                }
                A[i * N + j] = std::sqrt(sum);
            } else {
                A[i * N + j] = sum / A[j * N + j];
            }
        }
        
        // Zero out upper triangle
        for (int j = i + 1; j < N; j++) {
            A[i * N + j] = 0.0f;
        }
    }
    
    return true;
}

void forward_substitution(const float* L, const float* b, float* y, int N) {
    // Solve L * y = b for y, where L is lower triangular
    for (int i = 0; i < N; i++) {
        float sum = b[i];
        for (int j = 0; j < i; j++) {
            sum -= L[i * N + j] * y[j];
        }
        y[i] = sum / L[i * N + i];
    }
}

float log_det_from_cholesky(const float* L, int N) {
    // log(det(A)) = log(det(L * L^T)) = 2 * log(det(L))
    // log(det(L)) = sum(log(L[i,i]))
    float log_det = 0.0f;
    for (int i = 0; i < N; i++) {
        log_det += std::log(L[i * N + i]);
    }
    return 2.0f * log_det;
}

float log_multivariate_normal_pdf(
    const float* x,
    const float* mu,
    const float* L,
    float log_det_sigma,
    int K,
    float* workspace
) {
    // Compute log N(x | mu, Sigma) = -0.5 * (K*log(2π) + log|Σ| + (x-μ)^T Σ^-1 (x-μ))
    // Using Cholesky: Σ = L*L^T, so Σ^-1 = (L^T)^-1 * L^-1
    // Mahalanobis distance: (x-μ)^T Σ^-1 (x-μ) = ||L^-1 (x-μ)||^2
    
    float* diff = workspace;        // [K]
    float* z = workspace + K;       // [K]
    
    // diff = x - mu
    for (int i = 0; i < K; i++) {
        diff[i] = x[i] - mu[i];
    }
    
    // Solve L * z = diff for z (forward substitution)
    forward_substitution(L, diff, z, K);
    
    // Compute ||z||^2 = z^T * z
    float mahalanobis_sq = dot_product(z, z, K);
    
    // log N(x | mu, Sigma)
    const float log_2pi = 1.8378770664093454835606594728112f;  // log(2π)
    float log_prob = -0.5f * (K * log_2pi + log_det_sigma + mahalanobis_sq);
    
    return log_prob;
}

// ============================================================================
// Utility Functions
// ============================================================================

float log_sum_exp(float log_a, float log_b) {
    // Compute log(exp(log_a) + exp(log_b)) numerically stable
    if (std::isinf(log_a) && log_a < 0) return log_b;
    if (std::isinf(log_b) && log_b < 0) return log_a;
    
    float max_val = std::max(log_a, log_b);
    return max_val + std::log(std::exp(log_a - max_val) + std::exp(log_b - max_val));
}

float array_max(const float* arr, int N) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < N; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}

float array_sum(const float* arr, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += arr[i];
    }
    return sum;
}

void normalize_inplace(float* arr, int N) {
    float sum = array_sum(arr, N);
    if (sum > 0.0f) {
        for (int i = 0; i < N; i++) {
            arr[i] /= sum;
        }
    }
}

void normalize_from_log(const float* log_arr, float* arr, int N) {
    // Find max for numerical stability
    float max_log = array_max(log_arr, N);
    
    // Compute exp(log_arr - max_log)
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        arr[i] = std::exp(log_arr[i] - max_log);
        sum += arr[i];
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (int i = 0; i < N; i++) {
            arr[i] /= sum;
        }
    }
}

} // namespace linalg
} // namespace cpu
} // namespace hmm