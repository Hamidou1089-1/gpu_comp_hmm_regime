# ==============================================================================
# Library CUDA Kernels (primitives bas niveau)
# ==============================================================================

if(NOT BUILD_CUDA)
    message(STATUS "Skipping CUDA kernels - CUDA support disabled")
    return()
endif()

# Sources des kernels
set(KERNEL_SOURCES
    matrix_kernels.cu
    reduction_kernels.cu
    scan_kernels.cu
    potential_kernels.cu
    forward_backward_kernels.cu
    viterbi_kernels.cu
    baum_welch_kernels.cu  
)

# Créer la library
add_library(hmm_kernels STATIC ${KERNEL_SOURCES})

# ==============================================================================
# Configuration CUDA
# ==============================================================================
set_target_properties(hmm_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# ==============================================================================
# Include directories
# ==============================================================================
target_include_directories(hmm_kernels PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/cpu
    ${PROJECT_SOURCE_DIR}/include/cuda
)

# ==============================================================================
# Options de compilation CUDA
# ==============================================================================
target_compile_options(hmm_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler=-Wall
        -Xcompiler=-Wextra
    >
)

# Ajouter les flags définis au niveau root
target_compile_options(hmm_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS}>
)

# ==============================================================================
# Lier CUDA runtime
# ==============================================================================
target_link_libraries(hmm_kernels PUBLIC
    CUDA::cudart
)

# ==============================================================================
# Messages de debug
# ==============================================================================
message(STATUS "Building CUDA kernels library: hmm_kernels")
message(STATUS "  Sources: ${KERNEL_SOURCES}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")