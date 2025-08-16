
#!/bin/bash
cd "$(dirname "$0")"
cd "$(pwd)"
python3 scripts/run_final_replay.py
open reports/dashboard.html
exec $SHELL
