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

# CUDA Tests
if(BUILD_TESTS_CUDA AND BUILD_CUDA)
    message(STATUS "CUDA tests enabled")
    add_cuda_test(test_kernels_cuda)
    add_cuda_test(test_hmm_algorithms_cuda)
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