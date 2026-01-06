#pragma once

namespace hmm {
namespace gpu {
namespace kernels {

/**
 * @brief Lance le Scan Parallèle (Hillis-Steele) sur des matrices NxN.
 * * @tparam Semiring Le type d'algèbre (LogSum pour Forward, MaxSum pour Viterbi)
 * @param d_data [In/Out] Données initiales (T*N*N). Contient le résultat à la fin.
 * @param d_temp [Temp] Buffer temporaire de même taille (T*N*N).
 * @param T Longueur de la séquence temporelle.
 * @param N Nombre d'états.
 */
template <typename Semiring>
void run_parallel_scan(float* d_data, float* d_temp, int T, int N);

} // namespace kernels
} // namespace gpu
} // namespace hmm