#!/usr/bin/env bash
# run_sweep.sh - Phase 4: energy_sweep.py across power modes, one or more runs.
#
# Usage, from ~/Documents/Smolvlm:
#   bash run_sweep.sh                         # run 1, modes 0 1 2
#   bash run_sweep.sh 2 "2 1 0" 3 "1 2 0"     # runs 2 and 3, each in its own mode order
# Modes: 0 = 15W, 1 = 25W, 2 = MAXN SUPER.
# About 6 minutes per mode. Safe to re-run: finished run/mode/condition rows are skipped.
set -e
cd "$(dirname "$0")"
source ~/patchblock-venv/bin/activate

[ $# -eq 0 ] && set -- 1 "0 1 2"
if [ $(( $# % 2 )) -ne 0 ]; then
  echo "Arguments come in pairs: RUN \"MODE ORDER\". Example: bash run_sweep.sh 2 \"2 1 0\""
  exit 1
fi

sudo -v                                   # ask for the password once, up front
( while true; do sudo -n true; sleep 50; done ) 2>/dev/null &
KEEPALIVE=$!
trap 'kill $KEEPALIVE 2>/dev/null; sudo nvpmodel -m 1 >/dev/null 2>&1 || true' EXIT

while [ $# -gt 0 ]; do
  RUN=$1; ORDER=$2; shift 2
  for MODE in $ORDER; do
    echo "=================== run $RUN, nvpmodel -m $MODE ==================="
    sudo nvpmodel -m "$MODE"
    sudo nvpmodel -q
    sleep 30                              # let clocks, power and temperature settle
    python -u energy_sweep.py --resume --run "$RUN"
  done
done

python -u energy_sweep.py --summary-only
echo "Done. Board set back to 25W (mode 1)."
