#!/bin/bash
# run_all.sh — DEPRECATED single-machine pin → digest sequence.
#
# Use the coordinated `qsb_fleet.py search` command instead, which does
# pinning across all machines, then digest with full-fleet partitioning at
# the (seq, lt) the pin found.
#
# This script is kept for single-machine debugging only.
#
# Env vars:
#   QSB_TOTAL_GPUS, QSB_GLOBAL_OFFSET — fleet partitioning (or local default)
# Flags: easy, single_hash

set -e
cd "$(dirname "$0")"
mkdir -p results

TOTAL_GPUS="${QSB_TOTAL_GPUS:-1}"
GLOBAL_OFFSET="${QSB_GLOBAL_OFFSET:-0}"

EXTRA_FLAGS=""
for arg in "$@"; do
    case "$arg" in
        easy|single_hash) EXTRA_FLAGS="$EXTRA_FLAGS $arg" ;;
    esac
done

if command -v nvidia-smi >/dev/null; then
    LOCAL_GPUS=$(nvidia-smi -L | wc -l)
else
    LOCAL_GPUS=1
fi

echo "═════════════════════════════════════════════════════════════════════"
echo "  QSB single-machine search (DEPRECATED — use fleet.py search)"
echo "═════════════════════════════════════════════════════════════════════"
echo "  Local GPUs:    $LOCAL_GPUS"
echo "  Total fleet:   $TOTAL_GPUS"
echo "  Global offset: $GLOBAL_OFFSET"
echo "  Extra flags:   $EXTRA_FLAGS"
date

# Phase 1: pinning
echo ""
echo "──── Phase 1: pinning search ────"
./run_pin.sh $EXTRA_FLAGS

if [ ! -f pin_hit.json ]; then
    echo "No pin hit. Exiting."
    exit 0
fi

# Extract seq and lt from pin_hit.json
SEQ_HEX=$(python3 -c "import json; print(json.load(open('pin_hit.json'))['sequence_hex'])")
LT=$(python3 -c "import json; print(json.load(open('pin_hit.json'))['locktime'])")
echo "Pin hit: seq=$SEQ_HEX lt=$LT"

echo ""
echo "──── Phase 2: digest round 1 ────"
./run_digest.sh 1 "$SEQ_HEX" "$LT" $EXTRA_FLAGS

echo ""
echo "──── Phase 3: digest round 2 ────"
./run_digest.sh 2 "$SEQ_HEX" "$LT" $EXTRA_FLAGS

echo ""
echo "all phases done."
echo "Hits: pin_hit.json digest_r1_hit.json digest_r2_hit.json"
date
