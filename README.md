# GPU-Accelerated HMM for Financial Regime Detection

High-performance C++/CUDA implementation of Hidden Markov Models for detecting market regimes in financial time series.

## Overview

This project implements GPU-accelerated Hidden Markov Models based on **Hassan & Sévil (2021)** for financial regime detection. The system provides both CPU baseline and CUDA-optimized implementations of classical HMM algorithms with custom linear algebra operations tailored for numerical stability and performance.

**Key Components:**

- Forward-Backward, Viterbi, and Baum-Welch (EM) algorithms
- Log-space stable computations for numerical robustness
- Custom linear algebra library (matrix operations, Cholesky decomposition)
- Multivariate Gaussian emission models
- CPU and CUDA dual implementation

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

## References

- **Hassan, M. R., & Sévil, A. (2021).** *Temporal Parallelization of Inference in Hidden
  Markov Models*
- Course: GPU Computing - Ensimag 2025-2026

## Author

**Hamidou Diallo** - Ensimag
Academic Project - GPU Computing

---

*Last Updated: Janvier 2026*
