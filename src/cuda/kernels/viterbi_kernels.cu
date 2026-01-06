#include "viterbi_kernels.cuh"
#include <cuda_runtime.h>
#include <cfloat>

namespace hmm {
namespace gpu {
namespace kernels {

// Kernel séquentiel (1 thread) : Tres rapide car T*N tient en cache L2 GPU
__global__ void viterbi_backtrack_kernel(
    const float* __restrict__ delta, // [T*N] Calculé par Scan MaxSum (Forward)
    const float* __restrict__ A,     // [N*N] Transition
    int* __restrict__ path_out,      // [T] Résultat
    int T, int N
) {
    // Un seul thread exécute le backtrack complet pour la séquence
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // 1. Trouver le meilleur état final q_T
    float best_val = -INFINITY;
    int best_state = 0;

    const float* delta_T = delta + (T - 1) * N;
    for (int i = 0; i < N; ++i) {
        if (delta_T[i] > best_val) {
            best_val = delta_T[i];
            best_state = i;
        }
    }
    path_out[T - 1] = best_state;

    // 2. Remonter le temps : q_t = argmax_i ( delta_t(i) + log(A_i,q_{t+1}) )
    for (int t = T - 2; t >= 0; --t) {
        int next_state = path_out[t + 1];
        
        best_val = -INFINITY;
        best_state = 0;
        
        const float* delta_t = delta + t * N;
        
        for (int i = 0; i < N; ++i) {
            // Score = Score accumulé à t + Transition vers (t+1)
            float score = delta_t[i] + A[i * N + next_state];
            
            if (score > best_val) {
                best_val = score;
                best_state = i;
            }
        }
        path_out[t] = best_state;
    }
}

void launch_viterbi_backtrack(const float* delta, const float* A, int* path, int T, int N) {
    // 1 bloc, 1 thread
    viterbi_backtrack_kernel<<<1, 1>>>(delta, A, path, T, N);
}

} // namespace kernels
} // namespace gpu
} // namespace hmm