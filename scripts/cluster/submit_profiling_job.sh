#!/bin/bash
# ==============================================================================
# submit_profiling_job.sh
# Script de soumission pour profiling GPU robuste sur cluster ENSIMAG
# Compatible avec test_profile_gpu_robust et la nouvelle architecture
# ==============================================================================

set -e

# Configuration
REMOTE_USER="dialloh"
REMOTE_HOST="nash.ensimag.fr"
PROJECT_NAME="gpu_comp_hmm_regime"
PROJECT_DIR="$HOME/$PROJECT_NAME"
DATA_DIR="data/bench"
OUTPUT_DIR="results/benchmarks"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Arguments
MODE="${1:-timing}"      # timing | nsight | full
TEST_MODE="${2:-all}"    # all | scaling | convergence | quick

print_usage() {
    echo "Usage: $0 [MODE] [TEST_MODE]"
    echo ""
    echo "MODES:"
    echo "  timing    - Quick timing benchmark (default, ~30min)"
    echo "  nsight    - Detailed nsight-compute profiling (~4h)"
    echo "  full      - Both timing and nsight (~5h)"
    echo ""
    echo "TEST_MODES:"
    echo "  all         - Run all tests (default)"
    echo "  scaling     - Only scaling_T and scaling_N datasets"
    echo "  convergence - Only convergence tests"
    echo "  quick       - Small datasets only (T <= 10000)"
    echo ""
    echo "Examples:"
    echo "  $0                     # timing, all tests"
    echo "  $0 timing quick        # quick timing benchmark"
    echo "  $0 nsight scaling      # detailed profiling on scaling tests"
    echo "  $0 full all            # complete profiling suite"
}

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    print_usage
    exit 0
fi

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    GPU PROFILING JOB SUBMISSION - ENSIMAG CLUSTER            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Mode:      ${MODE}"
echo "║  Tests:     ${TEST_MODE}"
echo "║  Data dir:  ${DATA_DIR}"
echo "║  Output:    ${OUTPUT_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ==============================================================================
# 1. SYNC CODE TO CLUSTER
# ==============================================================================
echo -e "${BLUE}📤 Syncing code to cluster...${NC}"

rsync -avz --progress \
    --exclude 'build/' \
    --exclude '.vscode/' \
    --exclude '.venv/' \
    --exclude 'data/raw/' \
    --exclude '*.o' \
    --exclude '*.a' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'results/logs/*.out' \
    --exclude 'results/logs/*.err' \
    ./ ${REMOTE_USER}@${REMOTE_HOST}:~/${PROJECT_NAME}/

echo -e "${GREEN}✓ Code synced${NC}"

# ==============================================================================
# 2. CREATE JOB SCRIPT BASED ON MODE
# ==============================================================================

# Common header for all jobs
JOB_HEADER='#!/bin/bash
#SBATCH --job-name=hmm_JOB_TYPE_PLACEHOLDER
#SBATCH --output=results/logs/JOB_TYPE_PLACEHOLDER_%j.out
#SBATCH --error=results/logs/JOB_TYPE_PLACEHOLDER_%j.err
#SBATCH --time=TIME_LIMIT_PLACEHOLDER
#SBATCH --partition=rtx6000
#SBATCH --gres=shard:1           
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB

# Environment setup
module load cuda/12.0 2>/dev/null || true
module load cmake 2>/dev/null || true

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    HMM GPU PROFILING                                         ║"
echo "║    Job ID: $SLURM_JOB_ID"
echo "║    Node: $SLURM_NODELIST"
echo "║    Mode: MODE_PLACEHOLDER"
echo "║    Tests: TEST_MODE_PLACEHOLDER"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Date: $(date)"
echo ""

cd $HOME/PROJECT_NAME_PLACEHOLDER
mkdir -p results/logs results/benchmarks results/nsight

# GPU Info
echo "═══ GPU Information ═══"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
echo ""
'

# Build section (common)
BUILD_SECTION='
# ==============================================================================
# BUILD
# ==============================================================================
echo "═══ Building Project ═══"
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_CUDA=ON \
      -DBUILD_TESTS_CPU=ON \
      -DBUILD_TESTS_CUDA=ON \
      -DBUILD_PROFILING=ON \
      .. 2>&1 | tail -20
make -j4 test_profile_gpu_robust 2>&1 | tail -10
cd ..
echo "✓ Build complete"
echo ""
'

if [ "$MODE" == "nsight" ]; then
    # Nsight-compute profiling (detailed, slow)
    TIME_LIMIT="00:30:00"
    JOB_TYPE="nsight"
    
    RUN_SECTION='
# ==============================================================================
# NSIGHT COMPUTE PROFILING
# ==============================================================================
echo "═══ Running Nsight Compute (detailed profiling) ═══"
echo "This will take a while..."
echo ""

# Run with nsight-compute
ncu --set full \
    --target-processes all \
    --export results/nsight/profile_gpu_$(date +%Y%m%d_%H%M%S) \
    --force-overwrite \
    ./build/test/test_profile_gpu_robust DATA_DIR_PLACEHOLDER results/benchmarks/gpu_nsight TEST_MODE_PLACEHOLDER

echo ""
echo "═══ NSIGHT PROFILING COMPLETE ═══"
echo "Report saved in: results/nsight/"
ls -la results/nsight/*.ncu-rep 2>/dev/null || echo "(no reports found)"
'

elif [ "$MODE" == "full" ]; then
    # Full profiling: timing + nsight
    TIME_LIMIT="00:30:00"
    JOB_TYPE="full"
    
    RUN_SECTION='
# ==============================================================================
# PHASE 1: TIMING BENCHMARK
# ==============================================================================
echo "═══ Phase 1: Timing Benchmark ═══"
./build/test/test_profile_gpu_robust DATA_DIR_PLACEHOLDER results/benchmarks/gpu_benchmark TEST_MODE_PLACEHOLDER

echo ""
echo "═══ Phase 1 Results ═══"
cat results/benchmarks/gpu_benchmark_results.csv 2>/dev/null | head -20

# ==============================================================================
# PHASE 2: NSIGHT COMPUTE (on critical configs)
# ==============================================================================
echo ""
echo "═══ Phase 2: Nsight Compute (quick mode) ═══"
ncu --set full \
    --target-processes all \
    --export results/nsight/profile_gpu_$(date +%Y%m%d_%H%M%S) \
    --force-overwrite \
    ./build/test/test_profile_gpu_robust DATA_DIR_PLACEHOLDER results/benchmarks/gpu_nsight quick

echo ""
echo "═══ FULL PROFILING COMPLETE ═══"
echo "Timing results: results/benchmarks/gpu_benchmark_results.csv"
echo "Nsight reports: results/nsight/"
'

else
    # Default: timing only (fast)
    TIME_LIMIT="00:30:00"
    JOB_TYPE="timing"
    
    RUN_SECTION='
# ==============================================================================
# TIMING BENCHMARK
# ==============================================================================
echo "═══ Running Timing Benchmark ═══"
echo "Mode: TEST_MODE_PLACEHOLDER"
echo ""

./build/test/test_profile_gpu_robust DATA_DIR_PLACEHOLDER results/benchmarks/gpu_benchmark TEST_MODE_PLACEHOLDER

echo ""
echo "═══ BENCHMARK COMPLETE ═══"
echo ""
echo "Results:"
ls -la results/benchmarks/
echo ""
echo "CSV Preview:"
head -20 results/benchmarks/gpu_benchmark_results.csv 2>/dev/null || echo "(no results yet)"
'
fi

# Footer (common)
JOB_FOOTER='
echo ""
echo "═══ JOB FINISHED ═══"
echo "End time: $(date)"
echo "Duration: $SECONDS seconds"
'

# Assemble complete job script
JOB_SCRIPT="${JOB_HEADER}${BUILD_SECTION}${RUN_SECTION}${JOB_FOOTER}"

# Replace placeholders
JOB_SCRIPT="${JOB_SCRIPT//PROJECT_NAME_PLACEHOLDER/$PROJECT_NAME}"
JOB_SCRIPT="${JOB_SCRIPT//DATA_DIR_PLACEHOLDER/$DATA_DIR}"
JOB_SCRIPT="${JOB_SCRIPT//TEST_MODE_PLACEHOLDER/$TEST_MODE}"
JOB_SCRIPT="${JOB_SCRIPT//MODE_PLACEHOLDER/$MODE}"
JOB_SCRIPT="${JOB_SCRIPT//TIME_LIMIT_PLACEHOLDER/$TIME_LIMIT}"
JOB_SCRIPT="${JOB_SCRIPT//JOB_TYPE_PLACEHOLDER/$JOB_TYPE}"

# ==============================================================================
# 3. SUBMIT JOB
# ==============================================================================
echo -e "${BLUE}🚀 Submitting job to cluster...${NC}"

# Create job script on remote and submit
JOB_ID=$(ssh ${REMOTE_USER}@${REMOTE_HOST} "
    cd ${PROJECT_NAME}
    mkdir -p results/logs
    
    # Write job script
    cat > job_profiling.sh << 'EOFSCRIPT'
${JOB_SCRIPT}
EOFSCRIPT
    
    # Submit and get job ID
    sbatch --parsable job_profiling.sh
")

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    JOB SUBMITTED SUCCESSFULLY                                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Job ID:    ${JOB_ID}"
echo "║  Mode:      ${MODE}"
echo "║  Tests:     ${TEST_MODE}"
echo "║  Time limit: ${TIME_LIMIT}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
echo "Useful commands:"
echo "─────────────────────────────────────────────────────────────────"
echo "  Check status:"
echo "    ssh ${REMOTE_USER}@${REMOTE_HOST} 'squeue -u ${REMOTE_USER}'"
echo ""
echo "  Watch output live:"
echo "    ssh ${REMOTE_USER}@${REMOTE_HOST} 'tail -f ${PROJECT_NAME}/results/logs/${JOB_TYPE}_${JOB_ID}.out'"
echo ""
echo "  Cancel job:"
echo "    ssh ${REMOTE_USER}@${REMOTE_HOST} 'scancel ${JOB_ID}'"
echo ""
echo "  Download results when done:"
echo "    ./download_profiling_results.sh"
echo "    # or: make download-profiling"
echo ""