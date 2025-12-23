# GPU-Accelerated HMM for Financial Regime Detection

High-performance C++/CUDA implementation of Hidden Markov Models for detecting market regimes in financial time series, achieving up to 1000× speedup through GPU acceleration.

## Overview

This project implements GPU-accelerated Hidden Markov Models based on **Hassan & Sévil (2021)** for financial regime detection. The system provides both CPU baseline and CUDA-optimized implementations of classical HMM algorithms with custom linear algebra operations tailored for numerical stability and performance.

**Key Components:**
- Forward-Backward, Viterbi, and Baum-Welch (EM) algorithms
- Log-space stable computations for numerical robustness
- Custom linear algebra library (matrix operations, Cholesky decomposition)
- Multivariate Gaussian emission models
- CPU and CUDA dual implementation

## Status

**Current:** ✅ CPU implementation complete with full test suite  
**In Progress:** 🔄 CUDA kernel development and optimization  
**Next:** 📋 Performance benchmarking and cluster deployment

## Project Structure

```
├── src/
│   ├── cpu/              # CPU baseline implementation
│   └── cuda/             # CUDA-accelerated kernels
├── include/              # Public headers
├── tests/                # Unit and integration tests
├── scripts/              # Data processing and cluster utilities
└── data/                 # Financial time series datasets
```

## Quick Start

**Prerequisites:** CMake ≥ 3.18, C++17 compiler, CUDA Toolkit ≥ 11.0 (optional)

```bash
# Build (CPU only)
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Build with CUDA
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=ON ..
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

## Usage

```cpp
#include "algo_hmm.hpp"

// Initialize 3-state HMM with 2D observations
int N = 3, K = 2, T = 100;
GaussianParams params = initializeGaussianParams(mu, Sigma, N, K);

// Forward-backward algorithm
forward_algorithm_log(log_alpha, log_potentials, T, N);
backward_algorithm_log(log_beta, log_potentials, T, N);

// Compute posterior marginals
forward_backward_smoothing(log_gamma, log_xi, log_alpha, 
                          log_beta, log_potentials, T, N);
```

## References

- **Hassan, M. R., & Sévil, A. (2021).** *GPU Computing for 1000-fold Acceleration of HMM Likelihood Calculation*
- Course: GPU Computing - Ensimag 2024-2025

## Author

**Hamidou Diallo** - Ensimag  
Academic Project - GPU Computing Fall 2024

---
*Last Updated: December 2024*