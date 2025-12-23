#!/bin/bash
# Download latest logs from cluster

REMOTE_USER="dialloh"
REMOTE_HOST="mash.ensimag.fr"
PROJECT_NAME="gpu_comp_hmm_regime"
LOCAL_DIR="./results_cluster"

echo "Downloading logs from cluster..."

mkdir -p $LOCAL_DIR/logs

# Télécharger tous les logs récents
rsync -avz --progress \
    $REMOTE_USER@$REMOTE_HOST:~/$PROJECT_NAME/results/logs/ \
    $LOCAL_DIR/logs/

echo ""
echo "✓ Logs downloaded to $LOCAL_DIR/logs/"
echo ""
echo "Latest logs:"
ls -lht $LOCAL_DIR/logs/ | head -5