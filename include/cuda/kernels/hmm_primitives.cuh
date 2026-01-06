#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

namespace hmm {
namespace gpu {
namespace primitives {

// Constante
#ifndef LOG_2PI
#define LOG_2PI 1.8378770664093453f
#endif

// ============================================================================
// 1. MATHS DE BASE
// ============================================================================

__device__ __forceinline__ float log_sum_exp(float a, float b) {
    if (isinf(a) && a < 0.0f) return b;
    if (isinf(b) && b < 0.0f) return a;
    float m = fmaxf(a, b);
    return m + logf(expf(a - m) + expf(b - m));
}

// ============================================================================
// 2. ALGÈBRE LINÉAIRE (Optimisé pour K <= 16)
// ============================================================================

// Résolution système triangulaire inf L*z = x (Forward Sub)
// Z est stocké dans x ou un buffer à part. Ici z est un buffer temp.
__device__ __forceinline__ void cholesky_solve_device(
    const float* L,  // [K*K]
    const float* x,  // [K] (Diff y - mu)
    float* z,        // [K] (Sortie)
    int K
) {
    // Note: Pour K petit (ex 4, 8, 16), on pourrait dérouler la boucle (#pragma unroll)
    for (int i = 0; i < K; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < i; ++j) {
            sum += L[i * K + j] * z[j];
        }
        z[i] = (x[i] - sum) / L[i * K + i];
    }
}

// Calcul Log-PDF Gaussienne Multivariée
__device__ __forceinline__ float compute_log_gaussian_device(
    const float* y,      // [K] Observation
    const float* mu,     // [K] Moyenne
    const float* L,      // [K*K] Cholesky
    float log_det,       // Log déterminant de Sigma (2 * sum(log(diag(L))))
    int K
) {
    float diff[16]; // Buffer registre pour K <= 16
    float z[16];    // Buffer registre
    
    // 1. Diff = y - mu
    for (int k = 0; k < K; ++k) {
        diff[k] = y[k] - mu[k];
    }

    // 2. Solve L*z = diff
    cholesky_solve_device(L, diff, z, K);

    // 3. Mahalanobis = z^T * z
    float mahalanobis = 0.0f;
    for (int k = 0; k < K; ++k) {
        mahalanobis += z[k] * z[k];
    }

    // 4. Log PDF
    return -0.5f * (K * LOG_2PI + mahalanobis) - 0.5f * log_det; 
    // Note: log_det passé est souvent celui de Sigma. Si c'est celui de L, multiplier par 2.
    // Dans algo_hmm_cpu, vous calculez log_det(Sigma). Donc -0.5 * log_det est correct.
}

// ============================================================================
// 3. OPÉRATION MATRICIELLE ABSTRAITE (Pour le Scan)
// ============================================================================

// Calcule UNE cellule de la matrice résultat C = A (op) B
template <typename Semiring>
__device__ __forceinline__ float compute_matrix_cell_device(
    const float* A, // Matrice Gauche
    const float* B, // Matrice Droite
    int row, 
    int col, 
    int N
) {
    float acc = Semiring::zero();
    
    // Produit scalaire généralisé
    for (int k = 0; k < N; ++k) {
        // A_rk (op_mul) B_kc
        float val = Semiring::times(A[row * N + k], B[k * N + col]);
        // Acc (op_add) val
        acc = Semiring::plus(acc, val);
    }
    return acc;
}

} // namespace primitives
} // namespace gpu
} // namespace hmm