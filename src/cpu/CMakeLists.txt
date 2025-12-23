# ==============================================================================
# CPU Implementation Library
# ==============================================================================

set(CPU_SOURCES
    low_level_linear_algebra.cpp
    algo_hmm.cpp
)

add_library(hmm_cpu STATIC ${CPU_SOURCES})

target_include_directories(hmm_cpu PUBLIC 
    ${PROJECT_SOURCE_DIR}/include/cpu
)

target_compile_options(hmm_cpu PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3 -march=native -ffast-math>
    $<$<CONFIG:Debug>:-g -O0>
)

# Link math library
target_link_libraries(hmm_cpu PUBLIC m)

# Installation
install(TARGETS hmm_cpu 
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)