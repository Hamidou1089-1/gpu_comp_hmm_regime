#pragma once
#include "hmm_primitives.cuh"

namespace hmm {
namespace gpu {
namespace semiring {

// ============================================================================
// SEMIRING TROPICAL (MAX-PLUS) -> Pour Viterbi
// ============================================================================
struct MaxSum {
    // Élément neutre de l'addition (Max(-inf, x) = x)
    __device__ __forceinline__ static float zero() { return -INFINITY; }
    
    // Élément neutre de la multiplication (0 + x = x)
    __device__ __forceinline__ static float one()  { return 0.0f; }

    // "Addition" = Maximum
    __device__ __forceinline__ static float plus(float a, float b) {
        return fmaxf(a, b);
    }

    // "Multiplication" = Somme arithmétique
    __device__ __forceinline__ static float times(float a, float b) {
        return a + b;
    }
};

// ============================================================================
// SEMIRING LOGARITHMIQUE (LOG-SUM-EXP) -> Pour Forward/Backward
// ============================================================================
struct LogSum {
    // Élément neutre de l'addition (log(0) = -inf)
    __device__ __forceinline__ static float zero() { return -INFINITY; }
    
    // Élément neutre de la multiplication (log(1) = 0)
    __device__ __forceinline__ static float one()  { return 0.0f; }

    // "Addition" = LogSumExp
    __device__ __forceinline__ static float plus(float a, float b) {
        return primitives::log_sum_exp(a, b);
    }

    // "Multiplication" = Somme arithmétique
    __device__ __forceinline__ static float times(float a, float b) {
        return a + b;
    }
};

} // namespace semiring
} // namespace gpu
} // namespace hmm