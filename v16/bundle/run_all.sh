#!/bin/bash
# run_all.sh — Full QSB pipeline on multi-GPU
# Supports multi-machine parallelism for digest phases.
#
# Usage: bash run_all.sh [total_gpus] [global_offset]
#   total_gpus:    total GPUs across ALL machines (default: local count)
#   global_offset: this machine's starting GPU ID (default: 0)
#
# Example: 3 machines with 16 GPUs each:
#   Machine 1: bash run_all.sh 48 0
#   Machine 2: bash run_all.sh 48 16
#   Machine 3: bash run_all.sh 48 32

set -e
NUM_GPUS=$(nvidia-smi -L | wc -l)
TOTAL_GPUS=${1:-$NUM_GPUS}
GLOBAL_OFFSET=${2:-0}

echo "=== QSB Full Pipeline ==="
echo "  Local GPUs: $NUM_GPUS"
echo "  Total GPUs: $TOTAL_GPUS (offset $GLOBAL_OFFSET)"
echo ""

mkdir -p results logs

# ============================================================
# Phase 1: Pinning search
# ============================================================
echo "=== Phase 1: Pinning Search ==="
rm -f results/pinning_hit_*.txt
killall qsb_real 2>/dev/null || true
sleep 1

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    stdbuf -oL ./qsb_real pinning2.bin $gpu $TOTAL_GPUS $GLOBAL_OFFSET > logs/pin_gpu_${gpu}.log 2>&1 &
    sleep 1
done
echo "  $NUM_GPUS GPUs launched. Waiting for hit..."

while true; do
    sleep 10
    if compgen -G "results/pinning_hit_*.txt" > /dev/null 2>&1; then
        echo ""
        echo "  *** PINNING HIT! ***"
        killall qsb_real 2>/dev/null || true
        sleep 2
        cat results/pinning_hit_*.txt | head -4
        break
    fi
    tail -1 logs/pin_gpu_0.log 2>/dev/null || true
done

# Extract results
PINNING_FILE=$(ls results/pinning_hit_*.txt | head -1)
SEQUENCE=$(grep "^sequence=" "$PINNING_FILE" | head -1 | cut -d= -f2)
LOCKTIME=$(grep "^locktime=" "$PINNING_FILE" | head -1 | cut -d= -f2)
echo ""
echo "  seq=$SEQUENCE lt=$LOCKTIME"
echo ""

# ============================================================
# Phase 2: Digest Round 1
# ============================================================
echo "=== Phase 2: Digest Round 1 ==="
rm -f results/digest_hit_*.txt
killall qsb_digest 2>/dev/null || true
sleep 1

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    stdbuf -oL ./qsb_digest digest_r1.bin $gpu $SEQUENCE $LOCKTIME $TOTAL_GPUS $GLOBAL_OFFSET > logs/dig1_gpu_${gpu}.log 2>&1 &
    sleep 1
done
echo "  $NUM_GPUS GPUs launched (seq=$SEQUENCE lt=$LOCKTIME, global $GLOBAL_OFFSET..$((GLOBAL_OFFSET+NUM_GPUS-1)) of $TOTAL_GPUS). Waiting..."

while true; do
    sleep 15
    if compgen -G "results/digest_hit_*.txt" > /dev/null 2>&1; then
        echo ""
        echo "  *** DIGEST R1 HIT! ***"
        killall qsb_digest 2>/dev/null || true
        sleep 2
        cat results/digest_hit_*.txt
        break
    fi
    tail -1 logs/dig1_gpu_0.log 2>/dev/null || true
done

R1_FILE=$(ls results/digest_hit_*.txt | head -1)
R1_INDICES=$(grep "^indices=" "$R1_FILE" | head -1 | cut -d= -f2)
cp "$R1_FILE" results/round1_final.txt

# ============================================================
# Phase 3: Digest Round 2
# ============================================================
echo ""
echo "=== Phase 3: Digest Round 2 ==="
rm -f results/digest_hit_*.txt
killall qsb_digest 2>/dev/null || true
sleep 1

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    stdbuf -oL ./qsb_digest digest_r2.bin $gpu $SEQUENCE $LOCKTIME $TOTAL_GPUS $GLOBAL_OFFSET > logs/dig2_gpu_${gpu}.log 2>&1 &
    sleep 1
done
echo "  $NUM_GPUS GPUs launched. Waiting..."

while true; do
    sleep 15
    if compgen -G "results/digest_hit_*.txt" > /dev/null 2>&1; then
        echo ""
        echo "  *** DIGEST R2 HIT! ***"
        killall qsb_digest 2>/dev/null || true
        sleep 2
        cat results/digest_hit_*.txt
        break
    fi
    tail -1 logs/dig2_gpu_0.log 2>/dev/null || true
done

R2_FILE=$(ls results/digest_hit_*.txt | head -1)
R2_INDICES=$(grep "^indices=" "$R2_FILE" | head -1 | cut -d= -f2)
cp "$R2_FILE" results/round2_final.txt

# ============================================================
# Done!
# ============================================================
echo ""
echo "============================================"
echo "  ALL PHASES COMPLETE!"
echo "============================================"
echo "  Pinning: seq=$SEQUENCE lt=$LOCKTIME"
echo "  Round 1: $R1_INDICES"
echo "  Round 2: $R2_INDICES"
echo ""
echo "  On laptop run:"
echo "  python3 qsb_pipeline.py assemble \\"
echo "    --funding-txid 4fab76e9b0538a49a77443030f8e0243a5d2558155647a839acea0efaa4edc91 \\"
echo "    --funding-vout 0 --funding-value 10000 \\"
echo "    --version 2 \\"
echo "    --locktime $LOCKTIME --sequence $SEQUENCE \\"
echo "    --round1 $R1_INDICES --round2 $R2_INDICES"
