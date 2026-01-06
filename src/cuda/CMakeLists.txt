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
    ${CMAKE_CURRENT_SOURCE_DIR}/algo_hmm_gpu.cu
)

# Créer la library
add_library(hmm_cuda STATIC ${CUDA_SOURCES})

# ==============================================================================
# Configuration CUDA
# ==============================================================================
set_target_properties(hmm_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# ==============================================================================
# Include directories
# ==============================================================================
target_include_directories(hmm_cuda PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/cpu
    ${PROJECT_SOURCE_DIR}/include/cuda
    ${PROJECT_SOURCE_DIR}/include/cuda/kernels
)

# ==============================================================================
# Options de compilation CUDA
# ==============================================================================
target_compile_options(hmm_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler=-Wall
    >
)

# Ajouter les flags définis au niveau root
target_compile_options(hmm_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS}>
)

# ==============================================================================
# Dépendances
# ==============================================================================
target_link_libraries(hmm_cuda PUBLIC
    hmm_kernels  # Dépend des kernels
    hmm_cpu      # Pour structures partagées
    CUDA::cudart
    CUDA::cublas
)

# ==============================================================================
# Messages de debug
# ==============================================================================
message(STATUS "Building CUDA orchestration library: hmm_cuda")
message(STATUS "  Sources: ${CUDA_SOURCES}")
message(STATUS "  Dependencies: hmm_kernels, hmm_cpu")