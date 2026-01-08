#pragma once

#include <cuda_runtime.h>


namespace hmm {
namespace gpu {
namespace kernels {

    

    void launch_compute_gamma(
        const float* a, 
        const float* b, 
        float* g,
        int T, int N
    );

    void launch_update_transition(
        const float* a, 
        const float* b, 
        const float* e, 
        const float* oldA, 
        float* newA, 
        float ll, 
        int T, int N
    );

    void launch_normalize_A(float* A, int N);

    void launch_update_gaussian(
        const float* obs, 
        const float* gam, 
        float* mu, 
        float* sig, 
        float* sum_gam, 
        float ll, 
        int T, int N, int K
    );

    void launch_compute_log_likelihood(
        const float* alpha, 
        float* d_ll_out, 
        int T, int N
    );

}
}
}