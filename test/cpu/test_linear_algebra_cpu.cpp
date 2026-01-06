#include <iostream>
#include <vector>
#include <cstring>
#include "test_utils.hpp"
#include "linalg_cpu.hpp"

using namespace hmm::cpu::linalg;

void test_dot_product() {
    std::cout << "\n--- Test Dot Product ---\n";
    float v1[] = {1.0f, 2.0f, 3.0f};
    float v2[] = {4.0f, 5.0f, 6.0f};
    float res = dot_product(v1, v2, 3);
    TEST_ASSERT_FLOAT_EQ(32.0f, res, 1e-5, "Produit scalaire v1.v2");
}

void test_cholesky_valid() {
    std::cout << "\n--- Test Cholesky (Cas Valide) ---\n";
    float Sigma[] = {4.0f, 1.0f, 2.0f, 1.0f, 5.0f, 1.0f, 2.0f, 1.0f, 6.0f};
    // Matrice 3x3 définie positive
    // 4 1 2
    // 1 5 1
    // 2 1 6
    float L[9]; 
    std::memcpy(L, Sigma, 9 * sizeof(float));
    
    bool success = cholesky_decomposition(L, 3);
    
    // Verification manuelle de L (L * L^T doit redonner Sigma)
    // L attendu théorique (environ) :
    // 2.0  0    0
    // 0.5  2.18 0
    // 1.0  0.22 2.22
    
    TEST_ASSERT(success, "Cholesky doit reussir pour une matrice def. positive");
    TEST_ASSERT_FLOAT_EQ(2.0f, L[0], 1e-4, "L(0,0) doit etre 2.0"); 
    TEST_ASSERT_FLOAT_EQ(0.0f, L[1], 1e-4, "L(0,1) doit etre 0.0"); 
}

void test_cholesky_invalid() {
    std::cout << "\n--- Test Cholesky (Cas Invalide) ---\n";
    float Sigma[] = {-4.0f, 0.0f, 0.0f, 4.0f};
    float L[4];
    std::memcpy(L, Sigma, 4 * sizeof(float));
    bool success = cholesky_decomposition(L, 2);
    TEST_ASSERT(!success, "Cholesky doit echouer pour matrice negative");
}

void test_forward_substitution() {
    std::cout << "\n--- Test Forward Substitution ---\n";
    float L[] = {2.0f, 0.0f, 0.0f,
                 1.0f, 3.0f, 0.0f,
                 2.0f, 1.0f, 4.0f};
    float b[] = {4.0f, 10.0f, 20.0f};
    float y[3];
    
    forward_substitution(L, b, y, 3);
    
    TEST_ASSERT_FLOAT_EQ(2.0f, y[0], 1e-5, "y[0] = 2");
    TEST_ASSERT_FLOAT_EQ(8.0f/3.0f, y[1], 1e-5, "y[1] = 8/3");
    TEST_ASSERT_FLOAT_EQ(10.0f/3.0f, y[2], 1e-5, "y[2] = 10/3");
}

void test_cholesky_reconstruction() {
    std::cout << "\n--- Test Reconstruction Sigma = L * L^T ---\n";
    
    int n = 40;
    std::vector<float> Sigma(n * n);
    std::vector<float> L(n * n);
    std::vector<float> Reconstructed(n * n);
    
    generate_random_pd_matrix(Sigma.data(), n);
    std::memcpy(L.data(), Sigma.data(), n * n * sizeof(float));
    bool success = cholesky_decomposition(L.data(), n);
    TEST_ASSERT(success, "Cholesky doit réussir");
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
            Reconstructed[i * n + j] = sum;
        }
    }
    
    for (int i = 0; i < n * n; i++) {
        TEST_ASSERT_FLOAT_EQ(Sigma[i], Reconstructed[i], 1e-3, 
            "Reconstruction Sigma = L*L^T");
    }
}

int main() {
    test_dot_product();
    test_cholesky_valid();
    test_cholesky_invalid();
    test_cholesky_reconstruction();
    test_forward_substitution();
    return print_test_summary();
}