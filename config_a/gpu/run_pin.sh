#!/bin/bash
# run_pin.sh — pinning-only search on this machine.
#
# All local GPUs search in parallel. Each GPU iterates over (seq, lt) pairs
# using its own slot in the global fleet partitioning. Stops as soon as ANY
# GPU on this machine finds a hit.
#
# Env vars (set by the fleet launcher):
#   QSB_TOTAL_GPUS       total GPUs across the entire fleet
#   QSB_GLOBAL_OFFSET    this machine's GPU index offset
#
# Flags forwarded to the kernel: easy, single_hash
#
# Output:
#   results/pin_gpu*.log      per-GPU kernel logs
#   results/pinning_hit_*.txt raw hit data from the kernel (kernel-format)
#   pin_hit.json              normalized JSON of the first hit (for fleet pickup)
#   pin_status                "running" while searching, "found" when done with hit,
#                             "exhausted" if iteration completes without a hit

set -e
cd "$(dirname "$0")"
mkdir -p results

TOTAL_GPUS="${QSB_TOTAL_GPUS:-1}"
GLOBAL_OFFSET="${QSB_GLOBAL_OFFSET:-0}"

EXTRA_FLAGS=""
for arg in "$@"; do
    case "$arg" in
        easy|single_hash) EXTRA_FLAGS="$EXTRA_FLAGS $arg" ;;
        seq_start=*) EXTRA_FLAGS="$EXTRA_FLAGS $arg" ;;
    esac
done

if command -v nvidia-smi >/dev/null; then
    LOCAL_GPUS=$(nvidia-smi -L | wc -l)
else
    LOCAL_GPUS=1
fi

echo "running"  > pin_status

echo "═════════════════════════════════════════════════════════════════════"
echo "  QSB pinning search"
echo "═════════════════════════════════════════════════════════════════════"
echo "  Local GPUs:    $LOCAL_GPUS"
echo "  Total fleet:   $TOTAL_GPUS"
echo "  Global offset: $GLOBAL_OFFSET"
echo "  Extra flags:  $EXTRA_FLAGS"
date

# Launch one process per local GPU. Each gets a unique global GPU id (offset + g).
# qsb_real CLI:
#   ./qsb_real <pinning.bin> <gpu_index> [total_gpus] [global_offset] [easy] [single_hash]
# Kernel iterates over (seq, lt) internally; first to find a hit writes a file
# under results/ and other GPUs notice + exit.
declare -a pids=()
for g in $(seq 0 $((LOCAL_GPUS - 1))); do
    ./qsb_real pinning.bin "$g" "$TOTAL_GPUS" "$GLOBAL_OFFSET" $EXTRA_FLAGS \
        > "results/pin_gpu${g}.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || true; done

echo "pinning search done."

# Did any GPU on this machine find a hit?
hit_found=0
for f in results/pinning_hit_*.txt; do
    [ -e "$f" ] || continue
    seq=$(grep "^sequence=" "$f" | head -1 | cut -d= -f2)
    lt=$(grep "^locktime=" "$f" | head -1 | cut -d= -f2)
    hc=$(grep "^hash_choice=" "$f" | head -1 | cut -d= -f2)
    recid=$(grep "^recid=" "$f" | head -1 | cut -d= -f2)
    # Emit JSON. Sequence is reported as decimal by the kernel; convert to hex too.
    seq_hex=$(printf '0x%08x' "$seq")
    cat > pin_hit.json <<EOF
{
  "sequence": $seq,
  "sequence_hex": "$seq_hex",
  "locktime": $lt,
  "hash_choice": $hc,
  "recid": $recid,
  "machine_offset": $GLOBAL_OFFSET
}
EOF
    echo "PIN HIT (this machine): seq=$seq_hex lt=$lt hc=$hc recid=$recid"
    hit_found=1
    break
done

if [ $hit_found -eq 1 ]; then
    echo "found" > pin_status
else
    echo "exhausted" > pin_status
    echo "No pin hit on this machine's slice."
fi
date
