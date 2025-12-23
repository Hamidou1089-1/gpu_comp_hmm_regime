#!/bin/bash
# Cancel jobs on cluster

REMOTE_USER="dialloh"
REMOTE_HOST="mash.ensimag.fr"

if [ -z "$1" ]; then
    echo "Usage: $0 [job_id|all]"
    echo ""
    echo "Examples:"
    echo "  $0 12345      # Cancel specific job"
    echo "  $0 all        # Cancel all your jobs"
    exit 1
fi

if [ "$1" == "all" ]; then
    echo "Cancelling all jobs for $REMOTE_USER..."
    ssh $REMOTE_USER@$REMOTE_HOST "scancel -u $REMOTE_USER"
else
    echo "Cancelling job $1..."
    ssh $REMOTE_USER@$REMOTE_HOST "scancel $1"
fi

echo "✓ Done"