#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
export PYTHONPATH=$(pwd)
python scripts/run_final_replay.py --output-dir reports --lookback 12 --wick-ratio 0.25 --vol-burst-z 1.2 --demo --min-events 40 "$@"
if [ -f reports/dashboard.html ]; then
    open reports/dashboard.html
else
    echo "Dashboard file not found. Replay may have failed."
fi
exec $SHELL

