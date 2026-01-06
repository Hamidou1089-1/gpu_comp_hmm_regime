#include "baum_welch_kernels.cuh"
#include "hmm_primitives.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace hmm {
namespace gpu {
namespace kernels {

// ============================================================================
// E-STEP : Gamma
// ============================================================================
__global__ void compute_log_gamma_kernel(
    const float* alpha, 
    const float* beta, 
    float* log_gamma, // [T*N]
    int T, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < T * N) {
        // Log probabilité jointe non normalisée
        log_gamma[idx] = alpha[idx] + beta[idx]; 
    }
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================================
// M-STEP A : Accumulation Numérateur Transition (Xi)
// ============================================================================
__global__ void update_transition_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    const float* __restrict__ emissions, 
    const float* __restrict__ old_A,     // Log Space
    float* __restrict__ new_A_log,       // Sortie en Log Space direct
    float log_likelihood,
    int T, int N
) {
    // Grid (N, N) -> 1 bloc par élément de la matrice A
    int j = blockIdx.x; // To
    int i = blockIdx.y; // From
    int tid = threadIdx.x;

    if (i >= N || j >= N) return;

    // Accumulation en Log-Space (stable)
    float local_log_sum = -INFINITY;
    float log_A_ij = old_A[i * N + j];

    // Boucle parallèle (Stride) : plusieurs threads traitent T
    for (int t = tid; t < T - 1; t += blockDim.x) {
        // log xi = alpha + A + B + beta - LL
        float log_val = alpha[t * N + i] 
                      + log_A_ij 
                      + emissions[(t + 1) * N + j] 
                      + beta[(t + 1) * N + j]
                      - log_likelihood;
        
        // Somme logarithmique : log(exp(a) + exp(b))
        local_log_sum = primitives::log_sum_exp(local_log_sum, log_val);
    }

    // Réduction Shared Memory (pour sommer les résultats des threads)
    __shared__ float sdata[256];
    sdata[tid] = local_log_sum;
    __syncthreads();

    // Réduction classique en arbre
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = primitives::log_sum_exp(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Le chef de bloc écrit le numérateur (Log-Space)
    if (tid == 0) {
        new_A_log[i * N + j] = sdata[0];
    }
}

__global__ void update_mean_kernel_optimized(
    const float* __restrict__ obs,       // [T*K]
    const float* __restrict__ log_gamma, // [T*N]
    float* __restrict__ new_mu,          // [N*K] Accumulateur
    float* __restrict__ sum_gamma,       // [N] Accumulateur (Dénominateur)
    float log_likelihood,
    int T, int N, int K
) {
    int i = blockIdx.x; // État
    int k = blockIdx.y; // Dimension (Feature)
    
    // Accumulateurs locaux (Registres -> Rapide)
    float local_num = 0.0f;
    float local_denom = 0.0f;

    // Grid-Stride Loop : Les threads se partagent T
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        float log_w = log_gamma[t * N + i] - log_likelihood;
        
        // Optimisation : Skip si poids négligeable (log(1e-10) ~ -23)
        if (log_w > -23.0f) {
            float w = expf(log_w);
            local_denom += w;
            local_num += w * obs[t * K + k];
        }
    }

    // Réduction Block (Shared Memory)
    // On utilise les intrinsics Warp Shuffle pour aller plus vite que la Shared Mem classique
    local_num = warp_reduce_sum(local_num);
    local_denom = warp_reduce_sum(local_denom);

    // Seul le premier thread de chaque warp a la somme partielle
    // Il faut maintenant réduire les warps entre eux via Shared Memory
    __shared__ float s_num[8];   // Max 256 threads / 32 = 8 warps
    __shared__ float s_denom[8];

    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;

    if (lane == 0) {
        s_num[warp] = local_num;
        s_denom[warp] = local_denom;
    }
    __syncthreads();

    // Le premier warp réduit les résultats des autres warps
    if (warp == 0) {
        local_num = (lane < blockDim.x / warpSize) ? s_num[lane] : 0.0f;
        local_denom = (lane < blockDim.x / warpSize) ? s_denom[lane] : 0.0f;
        
        local_num = warp_reduce_sum(local_num);
        local_denom = warp_reduce_sum(local_denom);

        // ECITURE GLOBALE ATOMIQUE (1 SEULE PAR BLOC !)
        if (lane == 0) {
            atomicAdd(&new_mu[i * K + k], local_num);
            // Pour le dénominateur, tous les blocs (i, k=0..K) calculent la même chose.
            // On ne l'ajoute que si k==0 pour éviter de le sommer K fois.
            if (k == 0) {
                atomicAdd(&sum_gamma[i], local_denom);
            }
        }
    }
}


__global__ void update_cov_kernel_optimized(
    const float* __restrict__ obs,
    const float* __restrict__ log_gamma,
    float* __restrict__ new_Sigma,       // [N*K*K] Accumulateur
    float log_likelihood,
    int T, int N, int K
) {
    int i = blockIdx.x; // État
    int idx_cov = blockIdx.y; // Index aplati de covariance 0..(K*K-1)
    
    int r = idx_cov / K;
    int c = idx_cov % K;

    // Optimisation symétrique : on ne calcule que si c <= r (triangle inf)
    // Mais pour simplifier l'écriture mémoire, on calcule tout ici.

    float local_cov = 0.0f;

    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        float log_w = log_gamma[t * N + i] - log_likelihood;
        if (log_w > -23.0f) {
            float w = expf(log_w);
            // Terme E[X * X^T]
            local_cov += w * (obs[t * K + r] * obs[t * K + c]);
        }
    }

    // Réduction Warp
    local_cov = warp_reduce_sum(local_cov);

    // Réduction Block
    __shared__ float s_cov[8];
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;

    if (lane == 0) s_cov[warp] = local_cov;
    __syncthreads();

    if (warp == 0) {
        local_cov = (lane < blockDim.x / warpSize) ? s_cov[lane] : 0.0f;
        local_cov = warp_reduce_sum(local_cov);

        // Ecriture Atomique Unique
        if (lane == 0) {
            atomicAdd(&new_Sigma[i * K * K + r * K + c], local_cov);
        }
    }
}

__global__ void finalize_stats_kernel(float* mu, float* Sigma, const float* denom, int N, int K) {
    int i = blockIdx.x;
    if (i >= N) return;
    
    float d = denom[i];
    if (d < 1e-10f) return;

    // Mu = Sum / Denom
    for(int k=0; k<K; ++k) mu[i*K+k] /= d;

    // Sigma = E[XX] - E[X]E[X]
    for(int k=0; k<K*K; ++k) {
        int r = k/K; int c = k%K;
        float exx = Sigma[i*K*K + k] / d;
        float ex_sq = mu[i*K+r] * mu[i*K+c];
        Sigma[i*K*K + k] = exx - ex_sq;
    }
}

// Kernel pour repasser A en Log-Prob après normalisation
__global__ void normalize_log_A_kernel(float* A_log, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Ligne i
    if (i >= N) return;

    // 1. Calculer la somme de la ligne (Dénominateur)
    float log_sum = -INFINITY;
    for (int j = 0; j < N; ++j) {
        log_sum = primitives::log_sum_exp(log_sum, A_log[i * N + j]);
    }

    // 2. Normaliser : log(Num/Denom) = log(Num) - log(Denom)
    for (int j = 0; j < N; ++j) {
        A_log[i * N + j] -= log_sum;
    }
}

__global__ void compute_total_loglikelihood_kernel(const float* alpha, float* ll_out, int T, int N) {
    if (threadIdx.x != 0) return;
    float acc = -INFINITY;
    const float* alpha_last = alpha + (T - 1) * N;
    for (int i = 0; i < N; ++i) {
        acc = primitives::log_sum_exp(acc, alpha_last[i]);
    }
    *ll_out = acc;
}
// Wrappers
void launch_compute_log_likelihood(const float* alpha, float* d_ll_out, int T, int N) {
    compute_total_loglikelihood_kernel<<<1, 1>>>(alpha, d_ll_out, T, N);
}


void launch_compute_gamma(const float* a, const float* b, float* g, int T, int N) {
    dim3 bl(256); dim3 gr((T*N+255)/256);
    compute_log_gamma_kernel<<<gr, bl>>>(a, b, g, T, N);
}

void launch_update_transition(const float* a, const float* b, const float* e, const float* oldA, float* newA, float ll, int T, int N) {
    // Configuration de la grille :
    // - Grid : N x N blocs (un bloc par cellule de la matrice A)
    // - Block : 256 threads par bloc (pour paralléliser la boucle T)
    dim3 grid(N, N); 
    dim3 block(256); 
    
    // Mémoire partagée implicite statique (déjà dans le kernel)
    update_transition_kernel<<<grid, block>>>(a, b, e, oldA, newA, ll, T, N);
}

void launch_normalize_A(float* A, int N) {
    normalize_log_A_kernel<<<(N+255)/256, 256>>>(A, N);
}

// void launch_update_gaussian(const float* obs, const float* gam, float* mu, float* sig, float* sum_gam, float ll, int T, int N, int K) {
//     // 1 bloc par état, un seul thread par bloc (simple et sûr pour K petit)
//     update_gaussian_stats_kernel<<<N, 1>>>(obs, gam, mu, sig, sum_gam, ll, T, N, K);
// }

void launch_update_gaussian(const float* obs, const float* gam, float* mu, float* sig, float* sum_gam, float ll, int T, int N, int K) {
    // Reset buffers
    cudaMemset(mu, 0, N*K*sizeof(float));
    cudaMemset(sig, 0, N*K*K*sizeof(float));
    cudaMemset(sum_gam, 0, N*sizeof(float));

    // 1. Update Moyennes
    // Grid (N, K), Block 256
    dim3 grid_mu(N, K);
    update_mean_kernel_optimized<<<grid_mu, 256>>>(obs, gam, mu, sum_gam, ll, T, N, K);
    
    // 2. Update Covariances
    // Grid (N, K*K), Block 256
    dim3 grid_cov(N, K*K);
    update_cov_kernel_optimized<<<grid_cov, 256>>>(obs, gam, sig, ll, T, N, K);

    // 3. Finalisation
    finalize_stats_kernel<<<N, 1>>>(mu, sig, sum_gam, N, K);
}
} // kernels
} // gpu
} // hmm