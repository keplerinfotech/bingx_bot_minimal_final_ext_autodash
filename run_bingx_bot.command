#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
export PYTHONPATH=$(pwd)
python scripts/run_final_replay.py
if [ -f scripts/dashboard.html ]; then
    open scripts/dashboard.html
else
    echo "Dashboard file not found. Replay may have failed."
fi
exec $SHELL

