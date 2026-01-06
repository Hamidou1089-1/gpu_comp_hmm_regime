#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <functional>

/**
 * @file cuda_timing.cuh
 * @brief Simple macros and helpers for GPU timing with CUDA events
 */

// ============================================================================
// Simple CUDA Event Timing Macros
// ============================================================================

#define CUDA_TIME_START(start_event) \
    cudaEvent_t start_event; \
    cudaEventCreate(&start_event); \
    cudaEventRecord(start_event)

#define CUDA_TIME_END(start_event, stop_event, time_ms) \
    cudaEvent_t stop_event; \
    cudaEventCreate(&stop_event); \
    cudaEventRecord(stop_event); \
    cudaEventSynchronize(stop_event); \
    cudaEventElapsedTime(&time_ms, start_event, stop_event); \
    cudaEventDestroy(start_event); \
    cudaEventDestroy(stop_event)

// ============================================================================
// Benchmark Helper Function
// ============================================================================

/**
 * @brief Benchmark a CUDA kernel call with warmup and averaging
 * 
 * @param kernel_call Lambda or function to execute (should call kernel + sync if needed)
 * @param warmup Number of warmup iterations (default: 3)
 * @param iters Number of timed iterations (default: 10)
 * @return Average execution time in milliseconds
 * 
 * @example
 * float avg_time = benchmark_kernel([&]() {
 *     my_kernel<<<grid, block>>>(args);
 *     cudaDeviceSynchronize();
 * }, 3, 10);
 */
inline float benchmark_kernel(std::function<void()> kernel_call, int warmup=3, int iters=10) {
    // Warmup phase (to avoid cold start effects)
    for(int i=0; i<warmup; ++i) {
        kernel_call();
        cudaDeviceSynchronize(); // Ensure completion
    }
    
    // Timed phase
    float total_ms = 0.0f;
    for(int i=0; i<iters; ++i) {
        float time_ms;
        CUDA_TIME_START(start);
        kernel_call();
        cudaDeviceSynchronize(); // Ensure completion before timing
        CUDA_TIME_END(start, stop, time_ms);
        total_ms += time_ms;
    }
    
    return total_ms / iters;
}

