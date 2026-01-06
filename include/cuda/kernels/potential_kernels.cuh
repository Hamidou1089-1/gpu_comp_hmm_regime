#pragma once

namespace hmm {
namespace gpu {
namespace kernels {

// Prépare les matrices pour le Forward (Standard)
void launch_prepare_forward_inputs(
    const float* obs, const float* means, const float* L, const float* dets,
    const float* pi, const float* A, float* workspace,
    int T, int N, int K
);

// Prépare les matrices pour le Backward (Transposée + Inversion temps)
void launch_prepare_backward_inputs(
    const float* obs, const float* means, const float* L, const float* dets,
    const float* pi, const float* A, float* workspace,
    int T, int N, int K
);

void launch_compute_emissions(
    const float* obs, const float* means,
     const float* L, const float* dets, 
     float* emissions,
    int T, int N, int K
);

} // namespace kernels
} // namespace gpu
} // namespace hmm