#pragma once

namespace hmm {
namespace gpu {

/**
 * @brief Exécute l'algorithme Forward Parallèle (Hassan 2021).
 * * @param d_pi         [N] Log-probabilités initiales
 * @param d_A          [N*N] Log-probabilités de transition
 * @param d_means      [N*K] Moyennes des Gaussiennes
 * @param d_L          [N*K*K] Matrices Cholesky (triangulaire inf)
 * @param d_log_dets   [N] Log-déterminants des covariances
 * @param d_obs        [T*K] Séquence d'observations
 * @param d_alpha_out  [T*N] (Sortie) Alpha (Log-Likelihood forward)
 * @param T            Longueur séquence
 * @param N            Nombre d'états
 * @param K            Dimension observation
 */
void forward_gpu(
    const float* d_pi, const float* d_A, 
    const float* d_means, const float* d_L, const float* d_log_dets,
    const float* d_obs,
    float* d_alpha_out,
    int T, int N, int K
);

/**
 * @brief Exécute l'algorithme Viterbi Parallèle (Score Max seulement).
 * Note: Pour récupérer le chemin (argmax), il faudrait étendre le scan.
 * Ici on calcule delta_T (le score du meilleur chemin).
 */
void viterbi_score_gpu(
    const float* d_pi, const float* d_A, 
    const float* d_means, const float* d_L, const float* d_log_dets,
    const float* d_obs,
    float* d_delta_out,
    int T, int N, int K
);

void viterbi_path_gpu(
    const float* d_pi, const float* d_A, 
    const float* d_means, const float* d_L, const float* d_log_dets,
    const float* d_obs, 
    int* d_path_out, // [T]
    int T, int N, int K
);

void backward_gpu(
    const float* d_pi, const float* d_A, 
    const float* d_means, const float* d_L, const float* d_log_dets,
    const float* d_obs, float* d_beta_out, int T, int N, int K
);

// Smoothing (Gamma) 
// Calcule log(P(state | obs)) non normalisé (Joint Log Prob)
void smoothing_gpu(
    const float* d_alpha, const float* d_beta, float* d_gamma_out, int T, int N
);

float baum_welch_step_gpu(
    float* d_pi, float* d_A, 
    float* d_means, float* d_L, float* d_log_dets, // In/Out (Update)
    float* d_Sigma, // Stockage Covariance Linear Space
    const float* d_obs,
    int T, int N, int K
);

} // namespace gpu
} // namespace hmm