#!/bin/bash
# ==============================================================================
# download_profiling_results.sh
# Télécharge les résultats de profiling depuis le cluster ENSIMAG
# Compatible avec la nouvelle architecture de benchmark
# ==============================================================================

set -e

# Configuration
REMOTE_USER="dialloh"
REMOTE_HOST="nash.ensimag.fr"
PROJECT_NAME="gpu_comp_hmm_regime"
LOCAL_RESULTS_DIR="results"

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --logs-only     Download only log files"
    echo "  --results-only  Download only benchmark results (CSV/JSON)"
    echo "  --nsight-only   Download only nsight reports"
    echo "  --all           Download everything (default)"
    echo "  --clean-remote  Clean remote results after download"
    echo "  -h, --help      Show this help"
}

# Parse arguments
DOWNLOAD_LOGS=true
DOWNLOAD_RESULTS=true
DOWNLOAD_NSIGHT=true
CLEAN_REMOTE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --logs-only)
            DOWNLOAD_RESULTS=false
            DOWNLOAD_NSIGHT=false
            shift
            ;;
        --results-only)
            DOWNLOAD_LOGS=false
            DOWNLOAD_NSIGHT=false
            shift
            ;;
        --nsight-only)
            DOWNLOAD_LOGS=false
            DOWNLOAD_RESULTS=false
            shift
            ;;
        --all)
            shift
            ;;
        --clean-remote)
            CLEAN_REMOTE=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    DOWNLOAD PROFILING RESULTS FROM CLUSTER                   ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Remote: ${REMOTE_USER}@${REMOTE_HOST}"
echo "║  Project: ${PROJECT_NAME}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check SSH connectivity
echo -e "${BLUE}🔗 Checking connection...${NC}"
if ! ssh -q ${REMOTE_USER}@${REMOTE_HOST} "exit" 2>/dev/null; then
    echo -e "${RED}❌ Cannot connect to ${REMOTE_HOST}${NC}"
    echo "Please check:"
    echo "  - VPN connection (if required)"
    echo "  - SSH key configuration"
    exit 1
fi
echo -e "${GREEN}✓ Connected${NC}"

# Create local directories
mkdir -p ${LOCAL_RESULTS_DIR}/logs
mkdir -p ${LOCAL_RESULTS_DIR}/benchmarks
mkdir -p ${LOCAL_RESULTS_DIR}/nsight
mkdir -p ${LOCAL_RESULTS_DIR}/figures

# ==============================================================================
# 1. Download logs
# ==============================================================================
if [ "$DOWNLOAD_LOGS" = true ]; then
    echo ""
    echo -e "${BLUE}📥 Downloading logs...${NC}"
    rsync -avz --progress \
        --include='*.out' \
        --include='*.err' \
        --exclude='*' \
        ${REMOTE_USER}@${REMOTE_HOST}:~/${PROJECT_NAME}/results/logs/ \
        ${LOCAL_RESULTS_DIR}/logs/ 2>/dev/null || echo "  (no new logs)"
    
    # Count downloaded files
    LOG_COUNT=$(ls -1 ${LOCAL_RESULTS_DIR}/logs/*.out 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Downloaded ${LOG_COUNT} log files${NC}"
fi

# ==============================================================================
# 2. Download benchmark results
# ==============================================================================
if [ "$DOWNLOAD_RESULTS" = true ]; then
    echo ""
    echo -e "${BLUE}📥 Downloading benchmark results...${NC}"
    rsync -avz --progress \
        --include='*.csv' \
        --include='*.json' \
        --exclude='*' \
        ${REMOTE_USER}@${REMOTE_HOST}:~/${PROJECT_NAME}/results/benchmarks/ \
        ${LOCAL_RESULTS_DIR}/benchmarks/ 2>/dev/null || echo "  (no benchmark files yet)"
    
    # Show what we got
    if ls ${LOCAL_RESULTS_DIR}/benchmarks/*.csv 1>/dev/null 2>&1; then
        echo -e "${GREEN}✓ Downloaded benchmark files:${NC}"
        ls -la ${LOCAL_RESULTS_DIR}/benchmarks/*.csv 2>/dev/null
    else
        echo -e "${YELLOW}⚠ No benchmark CSV files found${NC}"
    fi
fi

# ==============================================================================
# 3. Download nsight reports
# ==============================================================================
if [ "$DOWNLOAD_NSIGHT" = true ]; then
    echo ""
    echo -e "${BLUE}📥 Downloading nsight reports...${NC}"
    rsync -avz --progress \
        --include='*.ncu-rep' \
        --include='*.nsys-rep' \
        --exclude='*' \
        ${REMOTE_USER}@${REMOTE_HOST}:~/${PROJECT_NAME}/results/nsight/ \
        ${LOCAL_RESULTS_DIR}/nsight/ 2>/dev/null || echo "  (no nsight reports yet)"
    
    # Show what we got
    if ls ${LOCAL_RESULTS_DIR}/nsight/*.ncu-rep 1>/dev/null 2>&1; then
        echo -e "${GREEN}✓ Downloaded nsight reports:${NC}"
        ls -la ${LOCAL_RESULTS_DIR}/nsight/*.ncu-rep 2>/dev/null
    else
        echo -e "${YELLOW}⚠ No nsight reports found${NC}"
    fi
fi

# ==============================================================================
# 4. Clean remote (optional)
# ==============================================================================
if [ "$CLEAN_REMOTE" = true ]; then
    echo ""
    echo -e "${YELLOW}🧹 Cleaning remote results...${NC}"
    ssh ${REMOTE_USER}@${REMOTE_HOST} "
        cd ${PROJECT_NAME}
        rm -f results/logs/*.out results/logs/*.err
        rm -f results/benchmarks/*.csv results/benchmarks/*.json
        rm -f results/nsight/*.ncu-rep results/nsight/*.nsys-rep
    "
    echo -e "${GREEN}✓ Remote results cleaned${NC}"
fi

# ==============================================================================
# 5. Summary
# ==============================================================================
echo ""
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    DOWNLOAD COMPLETE                                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
echo "Downloaded files summary:"
echo "─────────────────────────────────────────────────────────────────"

echo ""
echo "📁 Logs (${LOCAL_RESULTS_DIR}/logs/):"
ls -la ${LOCAL_RESULTS_DIR}/logs/*.out 2>/dev/null | tail -5 || echo "  (none)"

echo ""
echo "📁 Benchmarks (${LOCAL_RESULTS_DIR}/benchmarks/):"
ls -la ${LOCAL_RESULTS_DIR}/benchmarks/*.csv 2>/dev/null || echo "  (none)"

echo ""
echo "📁 Nsight reports (${LOCAL_RESULTS_DIR}/nsight/):"
ls -la ${LOCAL_RESULTS_DIR}/nsight/*.ncu-rep 2>/dev/null || echo "  (none)"

# ==============================================================================
# 6. Next steps
# ==============================================================================
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "─────────────────────────────────────────────────────────────────"

# Check if we have GPU results
if ls ${LOCAL_RESULTS_DIR}/benchmarks/gpu_*.csv 1>/dev/null 2>&1; then
    GPU_CSV=$(ls -t ${LOCAL_RESULTS_DIR}/benchmarks/gpu_*.csv | head -1)
    echo ""
    echo "  1. Run CPU profiling locally (if not done):"
    echo "     ./run_cpu_profiling.sh data/bench results/benchmarks/cpu_benchmark"
    echo ""
    echo "  2. Run hmmlearn baseline (if not done):"
    echo "     python benchmark_hmmlearn.py data/bench results/benchmarks/hmmlearn_benchmark"
    echo ""
    echo "  3. Analyze and generate plots:"
    echo "     python analyze_benchmark_results.py \\"
    echo "         --gpu-csv ${GPU_CSV} \\"
    echo "         --cpu-csv results/benchmarks/cpu_benchmark_results.csv \\"
    echo "         --hmmlearn-csv results/benchmarks/hmmlearn_benchmark_results.csv \\"
    echo "         --output-dir results/figures"
else
    echo ""
    echo "  No GPU benchmark results found yet."
    echo "  Check if job is still running:"
    echo "    ssh ${REMOTE_USER}@${REMOTE_HOST} 'squeue -u ${REMOTE_USER}'"
    echo ""
    echo "  View latest log:"
    echo "    cat ${LOCAL_RESULTS_DIR}/logs/\$(ls -t ${LOCAL_RESULTS_DIR}/logs/*.out 2>/dev/null | head -1)"
fi

# Check for nsight reports
if ls ${LOCAL_RESULTS_DIR}/nsight/*.ncu-rep 1>/dev/null 2>&1; then
    echo ""
    echo "  4. View nsight report:"
    echo "     ncu-ui ${LOCAL_RESULTS_DIR}/nsight/\$(ls -t ${LOCAL_RESULTS_DIR}/nsight/*.ncu-rep | head -1)"
fi

echo ""