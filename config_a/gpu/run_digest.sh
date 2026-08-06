#!/bin/bash
# run_digest.sh — digest-only search for one round at a specific (seq, lt).
#
# Usage:
#   ./run_digest.sh <round_number> <sequence_hex_or_dec> <locktime> [easy] [single_hash]
#
# Where:
#   round_number  1 or 2 (which digest round)
#   sequence      the seq value the pin found, e.g. 0x15CD6F36
#   locktime      the lt value the pin found, e.g. 1044068535
#
# Env vars:
#   QSB_TOTAL_GPUS       total GPUs across the fleet (for partitioning C(n,t))
#   QSB_GLOBAL_OFFSET    this machine's GPU index offset
#
# All local GPUs work in parallel, each on its own slot in the global partition.
# First GPU to find a hit signals others to stop (kernel does this internally
# via the results/digest_rN_hit_*.txt file presence check).
#
# Output:
#   results/digest_rN_gpu*.log
#   results/digest_rN_hit_*.txt  (raw kernel hit data)
#   digest_rN_hit.json           (normalized first hit, for fleet pickup)
#   digest_rN_status             "running" / "found" / "exhausted"

set -e
cd "$(dirname "$0")"
mkdir -p results

if [ $# -lt 3 ]; then
    echo "usage: $0 <round 1|2> <sequence> <locktime> [easy] [single_hash]"
    exit 1
fi

ROUND="$1"; shift
SEQUENCE="$1"; shift
LOCKTIME="$1"; shift

TOTAL_GPUS="${QSB_TOTAL_GPUS:-1}"
GLOBAL_OFFSET="${QSB_GLOBAL_OFFSET:-0}"

EXTRA_FLAGS=""
for arg in "$@"; do
    case "$arg" in
        easy|single_hash|calibrate) EXTRA_FLAGS="$EXTRA_FLAGS $arg" ;;
    esac
done

if command -v nvidia-smi >/dev/null; then
    LOCAL_GPUS=$(nvidia-smi -L | wc -l)
else
    LOCAL_GPUS=1
fi

if [ ! -f "digest_r${ROUND}.bin" ]; then
    echo "ERROR: digest_r${ROUND}.bin not found"
    exit 1
fi

echo "running" > "digest_r${ROUND}_status"

echo "═════════════════════════════════════════════════════════════════════"
echo "  QSB digest round $ROUND search"
echo "═════════════════════════════════════════════════════════════════════"
echo "  Local GPUs:    $LOCAL_GPUS"
echo "  Total fleet:   $TOTAL_GPUS"
echo "  Global offset: $GLOBAL_OFFSET"
echo "  Sequence:      $SEQUENCE"
echo "  Locktime:      $LOCKTIME"
echo "  Extra flags:  $EXTRA_FLAGS"
date

# qsb_digest CLI:
#   ./qsb_digest <digest.bin> <gpu> <seq> <lt> [total_gpus] [global_offset] [easy] [single_hash] [--tiles=PATH]
#
# IMPORTANT: pass the MACHINE-level $GLOBAL_OFFSET (not GLOBAL_OFFSET+g).
# The kernel internally computes effective_id = global_offset + gpu_index, so
# adding `g` here would double-count.
#
# If a per-GPU tile file exists (digest_r${ROUND}_tiles_gpu_<global_id>.bin),
# pass it as --tiles=PATH for balanced LPT partitioning.
declare -a pids=()
for g in $(seq 0 $((LOCAL_GPUS - 1))); do
    global_id=$((GLOBAL_OFFSET + g))
    tile_file="digest_r${ROUND}_tiles_gpu_${global_id}.bin"
    extra_tile=""
    if [ -f "$tile_file" ]; then
        extra_tile="--tiles=$tile_file"
    fi
    ./qsb_digest "digest_r${ROUND}.bin" "$g" "$SEQUENCE" "$LOCKTIME" \
        "$TOTAL_GPUS" "$GLOBAL_OFFSET" $EXTRA_FLAGS $extra_tile \
        > "results/digest_r${ROUND}_gpu${g}.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || true; done
echo "digest round $ROUND done."

# Did any GPU on this machine find a hit?
hit_found=0
for f in results/digest_hit_*.txt; do
    [ -e "$f" ] || continue
    combo=$(grep "^indices=" "$f" | head -1 | cut -d= -f2)
    hc=$(grep "^hash_choice=" "$f" | head -1 | cut -d= -f2 || echo 0)
    recid=$(grep "^recid=" "$f" | head -1 | cut -d= -f2 || echo 0)
    sighash=$(grep "^sighash=" "$f" | head -1 | cut -d= -f2 || echo "")
    pubhash=$(grep "^pubhash=" "$f" | head -1 | cut -d= -f2 || echo "")
    combo_idx=$(grep "^combo_idx=" "$f" | head -1 | cut -d= -f2 || echo 0)
    cat > "digest_r${ROUND}_hit.json" <<EOF
{
  "round": $ROUND,
  "combo": "$combo",
  "hash_choice": $hc,
  "recid": $recid,
  "sequence": "$SEQUENCE",
  "locktime": $LOCKTIME,
  "machine_offset": $GLOBAL_OFFSET,
  "sighash": "$sighash",
  "pubhash": "$pubhash",
  "combo_idx": $combo_idx
}
EOF
    echo "DIGEST R${ROUND} HIT (this machine): combo=$combo hc=$hc recid=$recid sighash=${sighash:0:16}..."
    hit_found=1
    break
done

if [ $hit_found -eq 1 ]; then
    echo "found" > "digest_r${ROUND}_status"
else
    echo "exhausted" > "digest_r${ROUND}_status"
    echo "No digest r${ROUND} hit on this machine's slice."
fi
date
