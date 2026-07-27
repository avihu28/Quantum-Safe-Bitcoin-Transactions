# QSB v16 — gpu_scalar_mulmod bug FIXED + parallel split4

## What's new

**Bug fix**: The R2 false positives traced to a real bug in `gpu_scalar_mulmod` 
(modular multiplication). Original used `__int128` accumulators that could 
overflow to 129 bits when summing 4 products into one slot. Fixed by 
incrementally propagating carry after each product.

**Verified empirically** — for the false-positive `z=a8f2d43a...`, the buggy 
version produced wrong `u1`. The fix produces the correct `u1` matching CPU.

**New command**: `rent_r2_split4` — rents 4 machines in parallel for full 
coverage in ~30-45 min.

## Commands

| Command | Machines | Coverage | Time | Cost |
|---|---|---|---|---|
| `rent_r2_split4` | 4 (parallel, offsets 0/16/32/48) | FULL (firsts 0-121) | ~30-45 min | ~$15-25 |
| `rent_r2_only` | 1 (offset=0) | FULL | ~2.5h | ~$25 |
| `rent_r2_fast` | 1 (offset=16) | PARTIAL (~25%) | ~9-30 min | ~$5 |

**Recommended for first attempt: `rent_r2_split4`** — fastest path to either 
finding R2 or knowing definitively that R2 isn't in the search space.

## Layout

| Path | What it is |
|---|---|
| `qsb_orchestrator_v16.py` | Rents vast.ai hosts, uploads the bundle, drives the search |
| `bundle/` | The CUDA kernels and search inputs uploaded to each host — **this is the authoritative v16 source** |
| `make_bundle.sh` | Packs `bundle/` into `qsb_v16.zip` (generated, not tracked) |
| `pipeline/` | Pure-Python secp256k1, tx assembly, local verification |
| `results/` | Pre-computed pinning + R1 hit that `rent_r2_split4` pre-uploads |
| `verify_r2_hit.py` | CPU re-derivation of a GPU hit, as a final check |

The kernels in `bundle/` are **not** interchangeable with the ones in
`../config_a/gpu/` — those predate the `gpu_scalar_mulmod` fix described above.

## Prerequisites

```bash
pip install -r ../requirements.txt
export VAST_API_KEY=<your key>          # or write it to ~/.vast_api_key
ssh-keygen -t ed25519                   # orchestrator uses ~/.ssh/id_ed25519
```

`pipeline/qsb_state.json` is **not** in this repository — it contains every
HORS preimage and the ECDSA nonces, so publishing it would hand over the
spending witness. Regenerate it with `pipeline/qsb_pipeline.py`, or copy your
local one into `pipeline/` before running `verify_r2_hit.py` (which reads it
from `pipeline/qsb_state.json` by default, overridable as its 2nd argument).

## Workflow

```bash
cd v16
./make_bundle.sh                        # produces qsb_v16.zip from bundle/

# Stop any leftover machines from older versions
python3 qsb_orchestrator_v16.py stop

# Recommended: 4-machine parallel full search
python3 qsb_orchestrator_v16.py rent_r2_split4

# Monitor (will print which machine finds the hit)
python3 qsb_orchestrator_v16.py collect

# Verify the hit (final sanity check)
python3 verify_r2_hit.py results/<inst>/round2_final.txt

# Stop all 4
python3 qsb_orchestrator_v16.py stop
```

## How split4 covers the search space

Each machine has GPUs that handle firsts `{effective_id, effective_id + total_gpus, ...}`
where `effective_id = offset + local_gpu_id`.

With 4 machines × ~16 GPUs each, total_gpus=64:
- Machine A (offset=0):  firsts {0-15, 64-79}
- Machine B (offset=16): firsts {16-31, 80-95}
- Machine C (offset=32): firsts {32-47, 96-111}
- Machine D (offset=48): firsts {48-63, 112-121}

Together: ALL firsts 0-121 covered, with 4× parallelism over single-machine.

## What to expect

- All 4 machines start setup in parallel (~5 min compile + GTable load)
- After warmup, search rate ~5 GB combos/sec aggregated across all GPUs
- First hit (real or none) determined within 30-45 min of warmup
- If no hit found after all machines exhaust their slices: R2 isn't in search 
  space (problem with pinning, R1, or puzzle parameters)

## Bug investigation summary

- v11-v15: identified false-positive R2, attempted various workarounds
- r2_ec_test V1-V6: localized the bug to `gpu_scalar_mulmod` and confirmed fix
- v16: production fix applied, false positive eliminated

