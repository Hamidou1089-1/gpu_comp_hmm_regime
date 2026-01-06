#include "scan_kernels.cuh"
#include "hmm_semirings.cuh"
#include "hmm_primitives.cuh"
#include <cstdio>

namespace hmm {
namespace gpu {
namespace kernels {

// Kernel générique (Hillis-Steele)
template <typename Semiring>
__global__ void scan_step_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int offset, int T, int N
) {
    int idx_mat = blockIdx.x * blockDim.x + threadIdx.x;
    int t       = blockIdx.y;

    if (t >= T || idx_mat >= N * N) return;
    int idx_global = t * (N * N) + idx_mat;

    if (t < offset) {
        output[idx_global] = input[idx_global];
    } else {
        int r = idx_mat / N;
        int c = idx_mat % N;
        
        const float* MatA = input + (t - offset) * (N * N);
        const float* MatB = input + t * (N * N);
        
        // Utilise la primitive device définie dans hmm_primitives
        output[idx_global] = primitives::compute_matrix_cell_device<Semiring>(MatA, MatB, r, c, N);
    }
}

template <typename Semiring>
void run_parallel_scan(float* d_data, float* d_temp, int T, int N) {
    dim3 block(256);
    dim3 grid((N * N + block.x - 1) / block.x, T);

    float* in = d_data;
    float* out = d_temp;

    // Scan logarithmique
    for (int offset = 1; offset < T; offset *= 2) {
        scan_step_kernel<Semiring><<<grid, block>>>(in, out, offset, T, N);
        cudaDeviceSynchronize();
        float* tmp = in; in = out; out = tmp;
    }

    if (in != d_data) {
        cudaMemcpy(d_data, d_temp, T * N * N * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

// Instanciations Explicites
template void run_parallel_scan<semiring::LogSum>(float*, float*, int, int);
template void run_parallel_scan<semiring::MaxSum>(float*, float*, int, int);

} // namespace kernels
} // namespace gpu
} // namespace hmm