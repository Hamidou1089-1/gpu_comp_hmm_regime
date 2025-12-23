#pragma once

#include <cmath>

void matrixVectorMult(const float* M, const float* v, float* result, 
                      int rows, int cols);

float dotProduct(const float* v1, const float* v2, int size);

bool choleskyDecomposition(const float* Sigma, float* L, int n);

void forwardSubstitution(const float* L, const float* b, float* y, int n);

void backwardSubstitution(const float* L, const float* y, float* x, int n);

float logDeterminantCholesky(const float* L, int n);