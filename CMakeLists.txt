cmake_minimum_required(VERSION 3.18)
project(GPU_HMM_Regime_Detection LANGUAGES CXX)

# ==============================================================================
# Options de compilation
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options du projet
option(BUILD_CUDA "Build with CUDA support" OFF)
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_PROFILING "Build with profiling support (perf/gprof)" OFF)
# ==============================================================================
# Configuration des flags de compilation
# ==============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Flags communs
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# ==============================================================================
# Détection et configuration CUDA
# ==============================================================================
if(BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86)  # Turing, Ampere, Ampere
    endif()
    
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math")
    
    message(STATUS "CUDA enabled - architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA disabled - CPU only build")
endif()

# ==============================================================================
# Répertoires d'inclusion
# ==============================================================================
include_directories(${PROJECT_SOURCE_DIR}/include/cpu)
include_directories(${PROJECT_SOURCE_DIR}/include/cuda)

# ==============================================================================
# Sous-répertoires
# ==============================================================================
add_subdirectory(src)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()




if(BUILD_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -fno-omit-frame-pointer")
    message(STATUS "Profiling enabled - perf/gprof compatible build")
endif()



# ==============================================================================
# Résumé de la configuration
# ==============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "GPU HMM Regime Detection - Configuration")
message(STATUS "========================================")
message(STATUS "Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler    : ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA support    : ${BUILD_CUDA}")
message(STATUS "Tests           : ${BUILD_TESTS}")
message(STATUS "Benchmarks      : ${BUILD_BENCHMARKS}")
message(STATUS "Profiling       : ${BUILD_PROFILING}")
message(STATUS "========================================")
message(STATUS "")