#pragma once



namespace hmm {
namespace gpu {
namespace kernels {




    void launch_viterbi_backtrack(
        const float* delta, 
        const float* A, 
        int* path, 
        int T, int N
    );

}
}
}