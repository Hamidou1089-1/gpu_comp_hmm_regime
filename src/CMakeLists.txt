# ==============================================================================
# Source files compilation
# ==============================================================================

# CPU implementation
add_subdirectory(cpu)


# ==============================================================================
# Main executable
# ==============================================================================
add_executable(hmm_main main.cpp)

# Link against the CPU library (always needed)
target_link_libraries(hmm_main PRIVATE hmm_cpu)

# Link against CUDA library if enabled
if(BUILD_CUDA)
    target_link_libraries(hmm_main PRIVATE hmm_cuda)
    target_compile_definitions(hmm_main PRIVATE USE_CUDA)
endif()

# Make sure the executable can find the headers
target_include_directories(hmm_main PRIVATE 
    ${PROJECT_SOURCE_DIR}/include
)

# Installation
install(TARGETS hmm_main DESTINATION bin)