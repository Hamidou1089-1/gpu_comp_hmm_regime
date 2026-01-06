#include <iostream>
#include "test_utils.hpp"
#include "algo_hmm_cpu.hpp"
#include "linalg_cpu.hpp"

using namespace hmm::cpu::algo;
using namespace hmm::cpu::linalg;


void test_log_sum_exp() {
    std::cout << "\n--- Test LogSumExp ---\n";
    // log(e^0 + e^0) = log(1 + 1) = log(2) approx 0.693147
    float val = log_sum_exp(0.0f, 0.0f);
    TEST_ASSERT_FLOAT_EQ(0.693147f, val, 1e-4, "log_sum_exp(0, 0)");
}

void test_gaussian_pdf() {
    std::cout << "\n--- Test Gaussian PDF (Cholesky) ---\n";
    
    // Cas simple 1D : Normale standard N(0, 1)
    // PDF au point x=0 est 1/sqrt(2*pi) approx 0.3989
    // Log PDF est log(0.3989) approx -0.9189
    
    int K = 1;
    float x[] = {0.0f};
    float mu[] = {0.0f};
    // Sigma = 1 -> L = 1 -> logDet = log(1^2) = 0
    float L[] = {1.0f}; 
    float logDet = 0.0f;
    float workspace[2]; // Besoin de 2*K

    float res = log_multivariate_normal_pdf(x, mu, L, logDet, K, workspace);
    
    // Valeur théorique : -0.5 * (1 * log(2pi) + 0 + 0) = -0.5 * 1.83787...
    float expected = -0.9189385f;
    
    TEST_ASSERT_FLOAT_EQ(expected, res, 1e-4, "Log PDF N(0,1) a x=0");
}

int main() {
    test_log_sum_exp();
    test_gaussian_pdf();
    return print_test_summary();
}