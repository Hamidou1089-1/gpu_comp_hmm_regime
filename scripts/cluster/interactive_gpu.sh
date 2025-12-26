#!/bin/bash
# Launch interactive session with GPU for debugging

REMOTE_USER="dialloh"
REMOTE_HOST="mash.ensimag.fr"

echo "Launching interactive GPU session..."
echo "This will give you a shell with GPU access"
echo ""

ssh -t $REMOTE_USER@$REMOTE_HOST "srun --gres=shard:1 --cpus-per-task=8 --mem=16GB --pty bash"