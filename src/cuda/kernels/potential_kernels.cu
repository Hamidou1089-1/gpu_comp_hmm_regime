#include "potential_kernels.cuh"
#include "hmm_primitives.cuh"

namespace hmm {
namespace gpu {
namespace kernels {

// ============================================================================
// KERNEL FORWARD : Calcul Emission + Embedding Matrice
// ============================================================================
// Grid: X = N*N, Y = T
__global__ void prepare_forward_kernel(
    const float* __restrict__ obs,
    const float* __restrict__ means,
    const float* __restrict__ L,
    const float* __restrict__ log_dets,
    const float* __restrict__ pi,
    const float* __restrict__ A,
    float* __restrict__ workspace, // [T*N*N]
    int T, int N, int K
) {
    int idx_mat = blockIdx.x * blockDim.x + threadIdx.x;
    int t       = blockIdx.y;

    if (t >= T || idx_mat >= N * N) return;

    int r = idx_mat / N; // Ligne (État départ)
    int c = idx_mat % N; // Colonne (État arrivée / émetteur)

    // Calcul de l'émission pour l'état Cible 'c' au temps 't'
    // Optimisation : Seulement si nécessaire (si r=0 à t=0, ou tout le temps si t>0)
    // Pour éviter divergence, on calcule.
    
    float log_emission = primitives::compute_log_gaussian_device(
        obs + t * K,
        means + c * K,
        L + c * K * K,
        log_dets[c],
        K
    );

    float val = -INFINITY;

    if (t == 0) {
        // T=0 : Embedding du vecteur Pi dans la ligne 0
        if (r == 0) {
            val = pi[c] + log_emission;
        } else {
            val = -INFINITY;
        }
    } else {
        // T>0 : M_ij = A_ij + B_t(j)
        val = A[r * N + c] + log_emission;
    }

    workspace[t * N * N + idx_mat] = val;
}

// ============================================================================
// KERNEL BACKWARD : Calcul Emission + Transposée + Inversion Temps
// ============================================================================
__global__ void prepare_backward_kernel(
    const float* __restrict__ obs,
    const float* __restrict__ means,
    const float* __restrict__ L,
    const float* __restrict__ log_dets,
    const float* __restrict__ pi,
    const float* __restrict__ A,
    float* __restrict__ workspace, // [T*N*N]
    int T, int N, int K
) {
    int idx_mat = blockIdx.x * blockDim.x + threadIdx.x;
    int k       = blockIdx.y; // Index Scan (0..T-1)

    if (k >= T || idx_mat >= N * N) return;

    // Mapping Temps Réel
    // Le scan backward k=0 correspond à la transition de T vers T-1
    // On veut calculer Beta_{T-1-k}.
    int t_real = T - 1 - k;

    int r = idx_mat / N;
    int c = idx_mat % N;

    // Matrice Transposée pour Backward : M'_{ji} = A_{ij} + B_{t+1}(j)
    // Ici r=j (destination backward), c=i (source backward)
    // Donc on charge A[c][r] (transposée)
    // Emission : dépend de l'état 'r' (celui qui a émis à t+1)
    
    float val = -INFINITY;

    if (k == 0) {
        // Init Beta_T (k=0 correspond à t=T-1)
        // Beta_T = 1 (log 0).
        // Beta_{T-1}(i) = sum_j A_{ij} * B_T(j) * 1
        // Donc M_init(i, j) = A_{ij} + B_T(j)
        // Mais notre Scan Backward fait v * M.
        // On veut beta_{t} = beta_{t+1} * M'
        // M'(r, c) avec r=état j (t+1), c=état i (t)
        // M'(r, c) = A_{cr} + B_{t+1}(r)  <-- Correction indices
        
        // Calcul Emission au temps T (dernier temps) pour l'état r
        float log_emission = primitives::compute_log_gaussian_device(
            obs + (T - 1) * K,
            means + r * K,
            L + r * K * K,
            log_dets[r],
            K
        );
        
        // A[c][r] : Transition de c vers r
        val = A[c * N + r] + log_emission;
        
    } else {
        // Pas général t < T-1
        // On vise beta_{t_real}. On vient de beta_{t_real+1}.
        // Emission à t_real + 1
        
        float log_emission = primitives::compute_log_gaussian_device(
            obs + (t_real + 1) * K,
            means + r * K,
            L + r * K * K,
            log_dets[r],
            K
        );

        val = A[c * N + r] + log_emission;
    }

    workspace[k * N * N + idx_mat] = val;
}

// ============================================================================
// CALCUL ÉMISSIONS SEULES (Pour Baum-Welch)
// ============================================================================
__global__ void compute_emissions_only_kernel(
    const float* __restrict__ obs,
    const float* __restrict__ means,
    const float* __restrict__ L,
    const float* __restrict__ log_dets,
    float* __restrict__ emissions_out, // [T*N]
    int T, int N, int K
) {
    int t = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x; // État

    if (t >= T || i >= N) return;

    // Calcul Gaussienne Log-Space
    float val = primitives::compute_log_gaussian_device(
        obs + t * K,
        means + i * K,
        L + i * K * K,
        log_dets[i],
        K
    );
    
    emissions_out[t * N + i] = val;
}

// Wrapper
void launch_compute_emissions(const float* obs, const float* means, const float* L, const float* dets, float* emissions, int T, int N, int K) {
    dim3 block(256);
    dim3 grid((N + 255) / 256, T);
    compute_emissions_only_kernel<<<grid, block>>>(obs, means, L, dets, emissions, T, N, K);
}


void launch_prepare_forward_inputs(const float* obs, const float* means, const float* L, const float* dets, const float* pi, const float* A, float* workspace, int T, int N, int K) {
    dim3 block(256);
    dim3 grid((N*N + 255)/256, T);
    prepare_forward_kernel<<<grid, block>>>(obs, means, L, dets, pi, A, workspace, T, N, K);
}

void launch_prepare_backward_inputs(const float* obs, const float* means, const float* L, const float* dets, const float* pi, const float* A, float* workspace, int T, int N, int K) {
    dim3 block(256);
    dim3 grid((N*N + 255)/256, T);
    prepare_backward_kernel<<<grid, block>>>(obs, means, L, dets, pi, A, workspace, T, N, K);
}

} // namespace kernels
} // namespace gpu
} // namespace hmm