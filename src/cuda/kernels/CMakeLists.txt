# ==============================================================================
# Library CUDA Kernels (primitives bas niveau)
# ==============================================================================

if(NOT BUILD_CUDA)
    message(STATUS "Skipping CUDA kernels - CUDA support disabled")
    return()
endif()

set(KERNEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/scan_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/potential_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/viterbi_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/baum_welch_kernels.cu
)

add_library(hmm_kernels STATIC ${KERNEL_SOURCES})




# ==============================================================================
# Configuration CUDA
# ==============================================================================
set_target_properties(hmm_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 14                    # ✅ Ajouter ça
    CUDA_STANDARD_REQUIRED ON           # ✅ Et ça
)

# ==============================================================================
# Include directories
# ==============================================================================
target_include_directories(hmm_kernels PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/cpu
    ${PROJECT_SOURCE_DIR}/include/cuda
    ${PROJECT_SOURCE_DIR}/include/cuda/kernels
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