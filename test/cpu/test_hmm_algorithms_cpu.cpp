// test_hmm_algorithms.cpp

#include "test_utils.hpp"
#include "algo_hmm.hpp"

void test_forward_backward_consistency() {
    std::cout << "\n--- Test Cohérence Forward-Backward ---\n";
    
    // Données synthétiques simples
    int T = 1000, K = 100, N = 3;
    auto data = generate_synthetic_hmm_sequence(T, K, N);
    
    // Précalcul
    GaussianParams params;
    precomputeGaussianParams(params, data.mu, data.Sigma, N, K);
    
    // Potentiels
    float* log_potentials;
    compute_log_gaussian_potentials(log_potentials, data.A, params, 
                                     data.pi, data.observations, T, K, N);
    
    // Forward
    std::vector<float> log_alpha(T * N);
    forward_algorithm_log(log_alpha.data(), log_potentials, T, N);
    
    // Backward
    std::vector<float> log_beta(T * N);
    backward_algorithm_log(log_beta.data(), log_potentials, T, N);
    
    // TEST : log P(y) calculé par forward doit être cohérent avec smoothing
    float log_likelihood_forward = compute_log_likelihood(log_alpha.data(), T, N);
    
    // Calculer via α[0] + β[0]
    std::vector<float> log_terms(N);
    for (int i = 0; i < N; i++) {
        log_terms[i] = log_alpha[0 * N + i] + log_beta[0 * N + i];
    }
    float log_likelihood_smoothing = log_sum_exp_array(log_terms.data(), N);
    
    TEST_ASSERT_FLOAT_EQ(log_likelihood_forward, log_likelihood_smoothing, 1e-3,
        "Log-likelihood cohérent entre forward et smoothing");
    
    // Cleanup
    freePotentials(log_potentials);
    freeGaussianParams(params);
    free_synthetic_data(data);
}

void test_viterbi_quality() {
    std::cout << "\n--- Test Qualité Viterbi ---\n";
    
    // Données avec états bien séparés
    int T = 100, K = 3, N = 3;
    auto data = generate_synthetic_hmm_sequence(T, K, N, 0.9f, 5.0f);
    
    // Algorithme
    GaussianParams params;
    precomputeGaussianParams(params, data.mu, data.Sigma, N, K);
    
    float* log_potentials;
    compute_log_gaussian_potentials(log_potentials, data.A, params,
                                     data.pi, data.observations, T, K, N);
    
    std::vector<int> best_path(T);
    float log_prob;
    viterbi_log(best_path.data(), &log_prob, log_potentials, T, N);
    
    // Mesurer la précision : % d'états correctement identifiés
    int correct = 0;
    for (int t = 0; t < T; t++) {
        if (best_path[t] == data.true_states[t]) {
            correct++;
        }
    }
    float accuracy = (float)correct / T;
    
    std::cout << "Précision Viterbi : " << accuracy * 100 << "%" << std::endl;
    TEST_ASSERT(accuracy > 0.7f, "Viterbi doit avoir >70% de précision sur données idéales");
    
    // Cleanup
    freePotentials(log_potentials);
    freeGaussianParams(params);
    free_synthetic_data(data);
}

int main()  {
    test_forward_backward_consistency();
    test_viterbi_quality();
    return 0;
}