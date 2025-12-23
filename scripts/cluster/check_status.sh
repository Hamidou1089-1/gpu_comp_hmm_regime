#!/bin/bash
# Check job status on ensicompute

REMOTE_USER="dialloh"
REMOTE_HOST="mash.ensimag.fr"

echo "Job status for $REMOTE_USER:"
echo ""

ssh $REMOTE_USER@$REMOTE_HOST "squeue -u $REMOTE_USER"

echo ""
echo "Legend:"
echo "  R  = Running"
echo "  PD = Pending (waiting for resources)"
echo "  CG = Completing"