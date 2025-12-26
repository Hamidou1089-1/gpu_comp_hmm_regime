# ==============================================================================
# Library CUDA (orchestration haut niveau)
# ==============================================================================

if(NOT BUILD_CUDA)
    message(STATUS "Skipping CUDA orchestration - CUDA support disabled")
    return()
endif()

# D'abord compiler les kernels
add_subdirectory(kernels)

# Sources de l'orchestration
set(CUDA_SOURCES
    algo_hmm_gpu.cu
    linear_algebra_gpu.cu
)

# Créer la library
add_library(hmm_gpu STATIC ${CUDA_SOURCES})

# ==============================================================================
# Configuration CUDA
# ==============================================================================
set_target_properties(hmm_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# ==============================================================================
# Include directories
# ==============================================================================
target_include_directories(hmm_gpu PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/cpu
    ${PROJECT_SOURCE_DIR}/include/cuda
)

# ==============================================================================
# Options de compilation CUDA
# ==============================================================================
target_compile_options(hmm_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler=-Wall
    >
)

# Ajouter les flags définis au niveau root
target_compile_options(hmm_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS}>
)

# ==============================================================================
# Dépendances
# ==============================================================================
target_link_libraries(hmm_gpu PUBLIC
    hmm_kernels  # Dépend des kernels
    hmm_cpu      # Pour structures partagées
    CUDA::cudart
    CUDA::cublas
)

# ==============================================================================
# Messages de debug
# ==============================================================================
message(STATUS "Building CUDA orchestration library: hmm_gpu")
message(STATUS "  Sources: ${CUDA_SOURCES}")
message(STATUS "  Dependencies: hmm_kernels, hmm_cpu")