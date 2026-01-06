#pragma once

#include <cstddef>

/**
 * @file linalg_cpu.hpp
 * @brief CPU implementation of linear algebra operations for HMM
 * 
 * All operations are implemented from scratch (no external libraries except standard library).
 * Optimized with -O3 -march=native for maximum CPU performance.
 */

namespace hmm {
namespace cpu {
namespace linalg {

// ============================================================================
// BLAS Level 1: Vector Operations
// ============================================================================

/**
 * @brief Dot product: result = x^T * y
 * @param x Vector [N]
 * @param y Vector [N]
 * @param N Vector size
 * @return Scalar result
 */
float dot_product(const float* x, const float* y, int N);

/**
 * @brief Element-wise multiplication: z = x ⊙ y
 * @param x Vector [N]
 * @param y Vector [N]
 * @param z Vector [N] (output)
 * @param N Vector size
 */
void element_wise_mult(const float* x, const float* y, float* z, int N);

/**
 * @brief Element-wise addition: z = x + y
 * @param x Vector [N]
 * @param y Vector [N]
 * @param z Vector [N] (output)
 * @param N Vector size
 */
void element_wise_add(const float* x, const float* y, float* z, int N);

/**
 * @brief Vector copy: y = x
 * @param x Source vector [N]
 * @param y Destination vector [N]
 * @param N Vector size
 */
void vector_copy(const float* x, float* y, int N);

/**
 * @brief Vector scale: y = alpha * x
 * @param alpha Scalar
 * @param x Vector [N]
 * @param y Vector [N] (output)
 * @param N Vector size
 */
void vector_scale(float alpha, const float* x, float* y, int N);

// ============================================================================
// BLAS Level 2: Matrix-Vector Operations
// ============================================================================

/**
 * @brief Matrix-vector multiplication: y = A * x
 * @param A Matrix [M × N] in row-major order
 * @param x Vector [N]
 * @param y Vector [M] (output)
 * @param M Number of rows in A
 * @param N Number of columns in A
 */
void matrix_vector_mult(const float* A, const float* x, float* y, int M, int N);

/**
 * @brief Matrix-vector multiplication with transpose: y = A^T * x
 * @param A Matrix [M × N] in row-major order
 * @param x Vector [M]
 * @param y Vector [N] (output)
 * @param M Number of rows in A
 * @param N Number of columns in A
 */
void matrix_vector_mult_transpose(const float* A, const float* x, float* y, int M, int N);

// ============================================================================
// BLAS Level 3: Matrix-Matrix Operations
// ============================================================================

/**
 * @brief Matrix-matrix multiplication: C = A * B
 * @param A Matrix [M × K] in row-major order
 * @param B Matrix [K × N] in row-major order
 * @param C Matrix [M × N] in row-major order (output)
 * @param M Rows of A
 * @param N Columns of B
 * @param K Columns of A / Rows of B
 */
void matrix_matrix_mult(const float* A, const float* B, float* C, int M, int N, int K);

// ============================================================================
// Specialized Operations for HMM
// ============================================================================

/**
 * @brief Cholesky decomposition: A = L * L^T (in-place)
 * 
 * Computes the Cholesky decomposition of a symmetric positive-definite matrix.
 * This is used for Gaussian PDF evaluation in HMM.
 * 
 * @param A Input: symmetric positive-definite matrix [N × N] (row-major)
 *          Output: lower triangular matrix L (in-place)
 * @param N Matrix dimension
 * @return true if successful, false if matrix is not positive-definite
 */
bool cholesky_decomposition(float* A, int N);

/**
 * @brief Forward substitution: solve L * y = b for y
 * 
 * Solves a lower triangular system. Used in combination with Cholesky
 * decomposition for computing Mahalanobis distance.
 * 
 * @param L Lower triangular matrix [N × N] (row-major)
 * @param b Right-hand side vector [N]
 * @param y Solution vector [N] (output)
 * @param N Matrix/vector dimension
 */
void forward_substitution(const float* L, const float* b, float* y, int N);

/**
 * @brief Compute log determinant from Cholesky factor
 * 
 * Given L from Cholesky decomposition A = L*L^T, compute log(det(A))
 * 
 * @param L Lower triangular Cholesky factor [N × N] (row-major)
 * @param N Matrix dimension
 * @return log(det(A))
 */
float log_det_from_cholesky(const float* L, int N);

/**
 * @brief Log-multivariate normal PDF using Cholesky decomposition
 * 
 * Computes log N(x | mu, Sigma) efficiently using pre-computed Cholesky factor.
 * 
 * @param x Observation vector [K]
 * @param mu Mean vector [K]
 * @param L Cholesky factor of covariance matrix [K × K] (row-major)
 * @param log_det_sigma Log determinant of Sigma (pre-computed)
 * @param K Dimension
 * @param workspace Workspace array [2*K] for intermediate computations
 * @return log probability density
 */
float log_multivariate_normal_pdf(
    const float* x,
    const float* mu,
    const float* L,
    float log_det_sigma,
    int K,
    float* workspace
);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Compute log(exp(a) + exp(b)) numerically stable
 * 
 * LogSumExp trick for numerical stability in log-space computations
 * 
 * @param log_a First log value
 * @param log_b Second log value
 * @return log(exp(log_a) + exp(log_b))
 */
float log_sum_exp(float log_a, float log_b);

/**
 * @brief Find maximum value in array
 * @param arr Array [N]
 * @param N Array size
 * @return Maximum value
 */
float array_max(const float* arr, int N);

/**
 * @brief Compute sum of array
 * @param arr Array [N]
 * @param N Array size
 * @return Sum of elements
 */
float array_sum(const float* arr, int N);

/**
 * @brief Normalize array in-place (divide by sum)
 * @param arr Array [N] (modified in-place)
 * @param N Array size
 */
void normalize_inplace(float* arr, int N);

/**
 * @brief Normalize array in log-space
 * 
 * Given log probabilities, normalize them: exp(log_p) / sum(exp(log_p))
 * Returns result in linear space.
 * 
 * @param log_arr Log-space array [N]
 * @param arr Output normalized array [N] (linear space)
 * @param N Array size
 */
void normalize_from_log(const float* log_arr, float* arr, int N);

} // namespace linalg
} // namespace cpu
} // namespace hmm