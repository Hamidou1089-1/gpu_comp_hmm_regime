#!/bin/bash
# =============================================================================
# run_cpu_profiling.sh
# Script pour lancer le profiling CPU avec métriques perf
# =============================================================================

set -e

# Configuration
DATA_DIR="${1:-data/bench}"
OUTPUT_PREFIX="${2:-results/benchmarks/cpu_benchmark}"
MODE="${3:-all}"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         CPU PROFILING WITH PERF METRICS                      ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Data dir: ${DATA_DIR}"
echo "║  Output:   ${OUTPUT_PREFIX}"
echo "║  Mode:     ${MODE}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Créer répertoires
mkdir -p "$(dirname ${OUTPUT_PREFIX})"
mkdir -p results/perf

# Vérifier que l'exécutable existe
if [ ! -f "build/test/test_profile_cpu_robust" ]; then
    echo -e "${YELLOW}Building test_profile_cpu_robust...${NC}"
    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_TESTS_CPU=ON \
          -DBUILD_PROFILING=ON \
          ..
    make -j$(nproc) test_profile_cpu_robust
    cd ..
fi

# =============================================================================
# Phase 1: Timing Benchmark (sans perf)
# =============================================================================
echo -e "\n${BLUE}═══ Phase 1: Timing Benchmark ═══${NC}"
./build/test/test_profile_cpu_robust "${DATA_DIR}" "${OUTPUT_PREFIX}" "${MODE}"

# =============================================================================
# Phase 2: Perf Stats (métriques globales)
# =============================================================================
echo -e "\n${BLUE}═══ Phase 2: Collecting Perf Stats ═══${NC}"

PERF_OUTPUT="results/perf/cpu_perf_stats.txt"

# Vérifier si perf est disponible
if command -v perf &> /dev/null; then
    echo "Running perf stat..."
    
    # Métriques à collecter
    PERF_EVENTS="cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses,cycles,instructions,branches,branch-misses"
    
    perf stat -e ${PERF_EVENTS} \
        -o "${PERF_OUTPUT}" \
        ./build/test/test_profile_cpu_robust "${DATA_DIR}" "results/perf/cpu_with_perf" "${MODE}" 2>&1 || {
        echo -e "${YELLOW}Note: perf stat may require elevated privileges${NC}"
        echo -e "${YELLOW}Try: sudo perf stat -e ... or run with CAP_PERFMON capability${NC}"
    }
    
    if [ -f "${PERF_OUTPUT}" ]; then
        echo -e "${GREEN}✓ Perf stats saved: ${PERF_OUTPUT}${NC}"
        echo ""
        echo "=== Perf Summary ==="
        cat "${PERF_OUTPUT}"
    fi
else
    echo -e "${YELLOW}⚠ perf not found. Skipping cache metrics.${NC}"
    echo "Install with: sudo apt-get install linux-tools-common linux-tools-generic"
fi

# =============================================================================
# Phase 3: Perf Record (optionnel, pour flamegraph)
# =============================================================================
if [ "${MODE}" = "profile" ] || [ "${4}" = "--record" ]; then
    echo -e "\n${BLUE}═══ Phase 3: Perf Record (for flamegraph) ═══${NC}"
    
    PERF_DATA="results/perf/cpu_perf.data"
    
    perf record -g -o "${PERF_DATA}" \
        ./build/test/test_profile_cpu_robust "${DATA_DIR}" "results/perf/cpu_profile" "quick" || {
        echo -e "${YELLOW}perf record failed. May need elevated privileges.${NC}"
    }
    
    if [ -f "${PERF_DATA}" ]; then
        echo -e "${GREEN}✓ Perf data saved: ${PERF_DATA}${NC}"
        echo ""
        echo "To analyze:"
        echo "  perf report -i ${PERF_DATA}"
        echo ""
        echo "To generate flamegraph (requires FlameGraph tools):"
        echo "  perf script -i ${PERF_DATA} | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo -e "\n${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    PROFILING COMPLETE                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Results: ${OUTPUT_PREFIX}_results.csv"
echo "║  Perf:    ${PERF_OUTPUT}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
echo "Next steps:"
echo "  1. Analyze results:"
echo "     python analyze_benchmark_results.py --cpu-csv ${OUTPUT_PREFIX}_results.csv"
echo ""
echo "  2. Compare with GPU (on cluster):"
echo "     ./submit_profiling_job.sh timing all"
echo ""
echo "  3. Run hmmlearn baseline:"
echo "     python benchmark_hmmlearn.py ${DATA_DIR} results/benchmarks/hmmlearn_benchmark"