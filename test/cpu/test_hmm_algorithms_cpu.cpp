// test_hmm_algorithms.cpp

#include "test_utils.hpp"
#include "algo_hmm_cpu.hpp"

using namespace hmm::cpu;

void test_forward_backward_consistency() {
    std::cout << "\n--- Test Cohérence Forward-Backward ---\n";
    
    int T = 1000, K = 100, N = 3;
    auto data = generate_synthetic_hmm_sequence(T, K, N);
    
    // Compute potentials
    int pot_size = N + (T - 1) * N * N;
    std::vector<float> log_potentials(pot_size);
    std::vector<float> workspace(2 * K);
    
    algo::compute_log_gaussian_potentials(
        data.model, data.observations, log_potentials.data(), workspace.data()
    );
    
    // Forward
    std::vector<float> alpha(T * N);
    float ll_forward = algo::forward_algorithm(log_potentials.data(), alpha.data(), T, N);
    
    // Backward
    std::vector<float> beta(T * N);
    algo::backward_algorithm(log_potentials.data(), beta.data(), T, N);
    
    // Check consistency: log P(y) from forward = α[0] + β[0]
    std::vector<float> log_terms(N);
    for (int i = 0; i < N; i++) {
        log_terms[i] = alpha[i] + beta[i];
    }
    float ll_smoothing = linalg::array_max(log_terms.data(), N);
    for (int i = 1; i < N; i++) {
        ll_smoothing = linalg::log_sum_exp(ll_smoothing, log_terms[i]);
    }
    std::cout << ll_forward << " " << ll_smoothing << std::endl;
    
    TEST_ASSERT_FLOAT_EQ(ll_forward, ll_smoothing, 1e-7,
        "Log-likelihood cohérent entre forward et smoothing");
    
    free_synthetic_data(data);
}



void test_viterbi_quality() {
    std::cout << "\n--- Test Qualité Viterbi ---\n";
    
    int T = 100, K = 3, N = 3;
    auto data = generate_synthetic_hmm_sequence(T, K, N, 0.9f, 5.0f);
    
    int pot_size = N + (T - 1) * N * N;
    std::vector<float> log_potentials(pot_size);
    std::vector<float> workspace(2 * K);
    
    algo::compute_log_gaussian_potentials(
        data.model, data.observations, log_potentials.data(), workspace.data()
    );
    
    std::vector<int> path(T);
    float log_prob = algo::viterbi_algorithm(log_potentials.data(), path.data(), T, N);
    
    int correct = 0;
    for (int t = 0; t < T; t++) {
        if (path[t] == data.true_states[t]) correct++;
    }
    float accuracy = (float)correct / T;
    
    std::cout << "Précision Viterbi : " << accuracy * 100 << "%" << std::endl;
    TEST_ASSERT(accuracy > 0.7f, "Viterbi doit avoir >70% de précision");
    
    free_synthetic_data(data);
}

int main() {
    test_forward_backward_consistency();
    test_viterbi_quality();
    return print_test_summary();
}