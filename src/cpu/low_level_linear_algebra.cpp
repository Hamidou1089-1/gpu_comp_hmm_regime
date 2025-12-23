#include "low_level_linear_algebra.hpp"
#include <cmath>
#include <cstring>


void matrixVectorMult(const float* M, const float* v, float* result, 
                      int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0f;
        for (int j = 0; j < cols; j++) {
            result[i] += M[i * cols + j] * v[j];
        }
    }
}

float dotProduct(const float* v1, const float* v2, int size) {
    float result = 0.0f;
    for (int i = 0; i < size; i++) {
        result += v1[i] * v2[i];
    }
    return result;
}

bool choleskyDecomposition(const float* Sigma, float* L, int n) {
    memset(L, 0, n * n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            
            if (j == i) {
                for (int k = 0; k < j; k++) {
                    sum += L[j * n + k] * L[j * n + k];
                }
                float val = Sigma[j * n + j] - sum;
                if (val <= 0.0f) {
                    return false;
                }
                L[j * n + j] = sqrt(val);
            } else {
                for (int k = 0; k < j; k++) {
                    sum += L[i * n + k] * L[j * n + k];
                }
                L[i * n + j] = (Sigma[i * n + j] - sum) / L[j * n + j];
            }
        }
    }
    return true;
}

void forwardSubstitution(const float* L, const float* b, float* y, int n) {
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < i; j++) {
            sum += L[i * n + j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i * n + i];
    }
}

void backwardSubstitution(const float* L, const float* y, float* x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        float sum = 0.0f;
        for (int j = i + 1; j < n; j++) {
            sum += L[j * n + i] * x[j];
        }
        x[i] = (y[i] - sum) / L[i * n + i];
    }
}

float logDeterminantCholesky(const float* L, int n) {
    float logDet = 0.0f;
    for (int i = 0; i < n; i++) {
        logDet += log(L[i * n + i]);
    }
    return 2.0f * logDet;
}

