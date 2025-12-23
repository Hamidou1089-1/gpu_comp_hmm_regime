#include <iostream>
#include <vector>
#include "test_utils.hpp"
#include "low_level_linear_algebra.hpp" // Chemin relatif vers le header

void test_dot_product() {
    std::cout << "\n--- Test Dot Product ---\n";
    float v1[] = {1.0f, 2.0f, 3.0f};
    float v2[] = {4.0f, 5.0f, 6.0f};
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    float res = dotProduct(v1, v2, 3);
    TEST_ASSERT_FLOAT_EQ(32.0f, res, 1e-5, "Produit scalaire v1.v2");
}

void test_cholesky_valid() {
    std::cout << "\n--- Test Cholesky (Cas Valide) ---\n";
    // Matrice 3x3 définie positive
    // 4 1 2
    // 1 5 1
    // 2 1 6
    float Sigma[] = {4.0f, 1.0f, 2.0f, 1.0f, 5.0f, 1.0f, 2.0f, 1.0f, 6.0f};
    float L[9]; 
    int n = 3;

    bool success = choleskyDecomposition(Sigma, L, n);
    TEST_ASSERT(success, "Cholesky doit reussir pour une matrice def. positive");

    // Verification manuelle de L (L * L^T doit redonner Sigma)
    // L attendu théorique (environ) :
    // 2.0  0    0
    // 0.5  2.18 0
    // 1.0  0.22 2.22
    
    TEST_ASSERT_FLOAT_EQ(2.0f, L[0], 1e-4, "L(0,0) doit etre 2.0"); 
    TEST_ASSERT_FLOAT_EQ(0.0f, L[1], 1e-4, "L(0,1) doit etre 0.0"); 
}

void test_cholesky_invalid() {
    std::cout << "\n--- Test Cholesky (Cas Invalide) ---\n";
    // Matrice non définie positive (ex: diagonale négative)
    float Sigma[] = {-4.0f, 0.0f, 0.0f, 4.0f}; // 2x2
    float L[4];
    bool success = choleskyDecomposition(Sigma, L, 2);
    TEST_ASSERT(!success, "Cholesky doit echouer pour matrice negative");
}


// test_linear_algebra.cpp - À ajouter

void test_forward_substitution() {
    std::cout << "\n--- Test Forward Substitution ---\n";
    
    // Système simple : L * y = b
    // [2  0  0]   [y1]   [4]
    // [1  3  0] * [y2] = [10]
    // [2  1  4]   [y3]   [20]
    
    float L[] = {2.0f, 0.0f, 0.0f,
                 1.0f, 3.0f, 0.0f,
                 2.0f, 1.0f, 4.0f};
    float b[] = {4.0f, 10.0f, 20.0f};
    float y[3];
    
    forwardSubstitution(L, b, y, 3);
    
    // Solution attendue : y1=2, y2=8/3, y3=3
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
    bool success = choleskyDecomposition(Sigma.data(), L.data(), n);
    TEST_ASSERT(success, "Cholesky doit réussir");
    
    // Reconstruire : Reconstructed = L * L^T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
            Reconstructed[i * n + j] = sum;
        }
    }
    
    // Vérifier que Reconstructed ≈ Sigma
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