#!/bin/bash
# ==============================================================================
# Submit GPU job to Ensicompute cluster
# Usage: bash submit_gpu_job.sh [test_name]
# ==============================================================================

set -e

REMOTE_USER="dialloh"
REMOTE_HOST="mash.ensimag.fr"
PROJECT_NAME="gpu_comp_hmm_regime"

TEST_NAME="${1:-all}"  # Par défaut, run tous les tests

echo "=========================================="
echo "Submitting GPU Job to Ensicompute"
echo "=========================================="
echo ""

# 1. Sync le code vers le cluster
echo "📤 Syncing code to cluster..."
rsync -avz --exclude 'build/' \
           --exclude '.vscode/' \
           --exclude '.venv/' \
           --exclude 'data/raw/' \
           --exclude '*.o' \
           --exclude '*.a' \
           ./ $REMOTE_USER@$REMOTE_HOST:~/$PROJECT_NAME/

echo "✓ Code synced"
echo ""

# 2. Créer le script de job
JOB_SCRIPT=$(cat <<'EOFSCRIPT'
#!/bin/bash
#SBATCH --job-name=hmm_gpu
#SBATCH --output=results/logs/gpu_%j.out
#SBATCH --error=results/logs/gpu_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=shard:1           # OBLIGATOIRE pour GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB                # Sinon réserve tout le serveur!

echo "=========================================="
echo "GPU Job - HMM Regime Detection"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="
echo ""

cd $HOME/PROJECT_NAME_PLACEHOLDER

# Créer les dossiers de résultats
mkdir -p results/logs
mkdir -p results/benchmarks

# Build avec CUDA si nécessaire
if [ ! -f "build/src/hmm_main" ]; then
    echo "Building with CUDA..."
    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_CUDA=ON \
          -DBUILD_TESTS_CPU=OFF \
          -DBUILD_TESTS_CUDA=ON \
          ..
    make -j4
    cd ..
    echo "✓ Build complete"
fi

cd build/tests

# Vérifier le GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Run les tests
if [ "TEST_NAME_PLACEHOLDER" == "all" ]; then
    echo "Running all GPU tests..."
    for test in test_*_cuda; do
        if [ -x "$test" ]; then
            echo ""
            echo "================================"
            echo "Running: $test"
            echo "================================"
            srun --gres=shard:1 --mem=16GB ./$test
        fi
    done
else
    echo "Running specific test: TEST_NAME_PLACEHOLDER"
    srun --gres=shard:1 --mem=2GB ./TEST_NAME_PLACEHOLDER
fi

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="
EOFSCRIPT
)

# Remplacer les placeholders
JOB_SCRIPT="${JOB_SCRIPT//PROJECT_NAME_PLACEHOLDER/$PROJECT_NAME}"
JOB_SCRIPT="${JOB_SCRIPT//TEST_NAME_PLACEHOLDER/$TEST_NAME}"

# 3. Soumettre le job via SSH
echo "🚀 Submitting job to Slurm..."
JOB_ID=$(ssh $REMOTE_USER@$REMOTE_HOST "cd $PROJECT_NAME && echo '$JOB_SCRIPT' | sbatch --parsable")

echo ""
echo "✅ Job submitted successfully!"
echo "   Job ID: $JOB_ID"
echo ""
echo "Commands to monitor:"
echo "  make cluster-status     # Check job status"
echo "  make cluster-logs       # Download logs"
echo "  ssh $REMOTE_USER@$REMOTE_HOST 'tail -f $PROJECT_NAME/results/logs/gpu_${JOB_ID}.out'"
echo ""