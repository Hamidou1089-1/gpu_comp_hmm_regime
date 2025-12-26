# ==============================================================================
# Test Suite - All tests are OPTIONAL
# ==============================================================================

# Options for fine-grained test control
option(BUILD_TESTS_CPU "Build CPU tests" ON)
option(BUILD_TESTS_CUDA "Build CUDA tests" OFF)
option(BUILD_TESTS_BENCHMARK "Build benchmark tests" OFF)

# Helper function for CPU tests
function(add_cpu_test test_name)
    if(BUILD_TESTS_CPU)
        add_executable(${test_name} cpu/${test_name}.cpp)
        target_link_libraries(${test_name} PRIVATE hmm_cpu)
        target_include_directories(${test_name} PRIVATE 
            ${PROJECT_SOURCE_DIR}/include/cpu           # Pour les headers publics
            ${PROJECT_SOURCE_DIR}/test/common       # Pour test_utils.hpp
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endif()
endfunction()

# Helper function for CUDA tests
function(add_cuda_test test_name)
    if(BUILD_TESTS_CUDA AND BUILD_CUDA)
        add_executable(${test_name} cuda/${test_name}.cu)
        target_link_libraries(${test_name} PRIVATE hmm_cpu hmm_cuda)
        target_include_directories(${test_name} PRIVATE 
            ${PROJECT_SOURCE_DIR}/include/cuda
            ${PROJECT_SOURCE_DIR}/test/common
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endif()
endfunction()

# ==============================================================================
# Register Tests
# ==============================================================================

# CPU Tests
if(BUILD_TESTS_CPU)
    message(STATUS "CPU tests enabled")
    add_cpu_test(test_linear_algebra_cpu)
    add_cpu_test(test_hmm_algorithms_cpu)
    add_cpu_test(test_real_data_cpu)
    add_cpu_test(test_stats_cpu)
    add_cpu_test(test_stress_cpu)
    add_cpu_test(test_profile_cpu)
endif()

if(BUILD_TESTS_CUDA AND BUILD_CUDA)
    message(STATUS "CUDA tests enabled")
    
    # Test 1: Scan primitives
    add_executable(test_scan_primitives_cuda cuda/test_scan_primitives.cu)
    target_link_libraries(test_scan_primitives_cuda PRIVATE 
        hmm_kernels  # Teste juste les kernels
        CUDA::cudart
    )
    target_include_directories(test_scan_primitives_cuda PRIVATE
        ${PROJECT_SOURCE_DIR}/include/cuda
        ${PROJECT_SOURCE_DIR}/test/common
    )
    add_test(NAME test_scan_primitives_cuda COMMAND test_scan_primitives_cuda)
    
    # Test 2: Linear algebra GPU
    add_executable(test_linear_algebra_cuda cuda/test_linear_algebra_cuda.cu)
    target_link_libraries(test_linear_algebra_cuda PRIVATE 
        hmm_cuda
        hmm_cpu
        CUDA::cudart
    )
    target_include_directories(test_linear_algebra_cuda PRIVATE
        ${PROJECT_SOURCE_DIR}/include/cuda
        ${PROJECT_SOURCE_DIR}/test/common
    )
    add_test(NAME test_linear_algebra_cuda COMMAND test_linear_algebra_cuda)
    
    # Test 3: Forward-Backward GPU
    add_executable(test_forward_backward_cuda cuda/test_forward_backward_cuda.cu)
    target_link_libraries(test_forward_backward_cuda PRIVATE 
        hmm_cuda
        hmm_cpu
        CUDA::cudart
    )
    target_include_directories(test_forward_backward_cuda PRIVATE
        ${PROJECT_SOURCE_DIR}/include/cuda
        ${PROJECT_SOURCE_DIR}/test/common
    )
    add_test(NAME test_forward_backward_cuda COMMAND test_forward_backward_cuda)
    
    # Test 4: Viterbi GPU
    add_executable(test_viterbi_cuda cuda/test_viterbi_cuda.cu)
    target_link_libraries(test_viterbi_cuda PRIVATE 
        hmm_cuda
        hmm_cpu
        CUDA::cudart
    )
    target_include_directories(test_viterbi_cuda PRIVATE
        ${PROJECT_SOURCE_DIR}/include/cuda
        ${PROJECT_SOURCE_DIR}/test/common
    )
    add_test(NAME test_viterbi_cuda COMMAND test_viterbi_cuda)
    
    # Benchmark CPU vs GPU
    if(BUILD_BENCHMARKS)
        add_executable(benchmark_gpu_vs_cpu cuda/benchmark_gpu_vs_cpu.cu)
        target_link_libraries(benchmark_gpu_vs_cpu PRIVATE 
            hmm_cuda
            hmm_cpu
            CUDA::cudart
        )
        target_include_directories(benchmark_gpu_vs_cpu PRIVATE
            ${PROJECT_SOURCE_DIR}/include/cuda
            ${PROJECT_SOURCE_DIR}/test/common
        )
        add_test(NAME benchmark_gpu_vs_cpu COMMAND benchmark_gpu_vs_cpu)
        
        message(STATUS "  - benchmark_gpu_vs_cpu")
    endif()
    
    message(STATUS "  - test_scan_primitives_cuda")
    message(STATUS "  - test_linear_algebra_cuda")
    message(STATUS "  - test_forward_backward_cuda")
    message(STATUS "  - test_viterbi_cuda")
endif()


# ==============================================================================
# Custom test targets
# ==============================================================================

if(BUILD_TESTS_CPU)
    add_custom_target(run_tests_cpu
        COMMAND ${CMAKE_CTEST_COMMAND} -R "cpu" --output-on-failure
        COMMENT "Running CPU tests..."
    )
endif()

if(BUILD_TESTS_CUDA AND BUILD_CUDA)
    add_custom_target(run_tests_gpu
        COMMAND ${CMAKE_CTEST_COMMAND} -R "cuda" --output-on-failure
        COMMENT "Running GPU tests..."
    )
endif()