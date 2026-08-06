# QSB Config A — full workflow

Workflow to produce a signed Quantum-Safe Bitcoin spending tx using the
paper's Config A (n=150, 8+1b R1, 7+2b R2, all-SHA-256) via a one-click
vast.ai GPU fleet.

## Summary

```
┌────────────┐     ┌──────────┐     ┌────────────┐     ┌────────────┐     ┌──────────┐
│  Setup     │ ──▶ │  Fund    │ ──▶ │  Export    │ ──▶ │  GPU fleet │ ──▶ │ Assemble │
│ (generate  │     │ (10k     │     │ (GPU input │     │ search     │     │ + submit │
│  state)    │     │  sats    │     │  files)    │     │ (~$100)    │     │ spending │
│            │     │  tx)     │     │            │     │            │     │  tx      │
└────────────┘     └──────────┘     └────────────┘     └────────────┘     └──────────┘
```

## Prerequisites

- Python 3.7+
- `pip install vastai` (for the fleet launcher)
- A Bitcoin input UTXO with ≥ 12,000 sats (10k for QSB + ~2k fee)
- A vast.ai account with funded balance (~$100)
- A mining pool that accepts non-standard transactions (e.g., MARA Slipstream)
  for broadcasting both the funding tx and the spending tx

## Layout

```
qsb_config_a/
├── pipeline/             # Python pipeline (setup, export, assemble)
│   ├── qsb_pipeline.py
│   ├── bitcoin_tx.py     # (contains the 3 bug fixes: op count, F&D, runtime)
│   ├── secp256k1.py
│   └── secp256k1_fast.py
├── gpu/                  # GPU kernels + per-machine orchestration
│   ├── qsb_digest_search.cu
│   ├── qsb_real_search.cu
│   ├── GPUHash.h, GPUMath.h
│   └── run_all.sh        # runs on each GPU machine
├── verify/               # Test & hit-verification harness
│   ├── verify_script_budget.py   # pre-flight op-budget check
│   ├── gpu_emulator.py           # byte-for-byte Python reference of GPU
│   ├── ref_sighash.py            # reference sighash impl
│   ├── test_gpu_cpu_equivalence.py   # 32-test equivalence suite
│   ├── test_end_to_end.py        # pipeline ↔ emulator integration test
│   ├── test_all.py               # ← THE master test runner
│   └── verify_hit.py             # validate a claimed GPU hit before assembly
├── fleet/                # vast.ai one-click launcher
│   └── qsb_fleet.py
├── FUNDING.md            # on-chain funding tx walkthrough
└── README.md             # this file
```

## Step-by-step

### 0 — Run the master test suite (MANDATORY)

Before spending a single sat of real money:

```bash
cd qsb_config_a
python3 verify/test_all.py
```

The suite runs:

- [A] Pre-flight: op budget (static + CHECKMULTISIG runtime), script size,
      FindAndDelete vs reference
- [B] 32-check equivalence: DER check, sighash, pubkey compression,
      puzzle_hash (all modes), ECDSA recovery, script structure
- [C] End-to-end: real `setup` → real `export` → CPU search → emulator
      byte-compare on actual Config A parameters
- [C2] verify_hit reject paths: wrong-indices, tampered-params,
      old-format-params — verifies the last-line defense catches bad hits
- [C3] GPU kernel static analysis: parses the `.cu` files with g++ using
      CUDA stubs, catches most errors nvcc would catch (the actual nvcc
      compile happens at fleet-launch time)
- [D] Config sanity: A/Ar/S fit, D flagged over-budget

**If any phase fails, STOP and do not proceed.**

### 1 — Setup

```bash
cd pipeline
python3 qsb_pipeline.py setup --config A
```

Produces:
- `qsb_state.json` — HORS secrets, commitments, dummy sigs, sig_nonces, full script
- `qsb_scriptpubkey.hex` — the locking script (~9.9 KB)

Setup hard-fails if the built script would be consensus-invalid (e.g., over
the 201-opcode limit at runtime). So if `setup` succeeds, your script passes
basic structural checks.

### 1.5 — Regtest validation (STRONGLY recommended)

Before spending real money, validate the pipeline against Bitcoin Core's
actual consensus engine in regtest. See [regtest/README.md](regtest/README.md)
for full details.

```bash
# Start a local regtest node
~/qsb_config_a/regtest/setup_regtest.sh start

# Phase 1: confirm the QSB scriptPubKey can be funded (free)
python3 ~/qsb_config_a/regtest/test_funding_tx.py

# (Later, after a small GPU search using the regtest funding txid:)
# Phase 3: confirm the assembled spending tx passes consensus
python3 ~/qsb_config_a/regtest/test_spending_tx.py --hits gpu_hits.json
```

This catches the consensus-failure class of bugs (the kind that broke Config D)
before any mainnet broadcast. **The test_all suite alone cannot catch these
bugs** because it doesn't run Bitcoin's actual script interpreter.

### 2 — Fund

See [FUNDING.md](FUNDING.md) for the full walkthrough. TL;DR:

```bash
python3 qsb_pipeline.py fund \
    --input-txid <your UTXO txid> \
    --input-vout <vout> \
    --input-value <sats> \
    --qsb-value 10000 \
    --change-address <20-byte pubkey hash hex>
# Sign the produced qsb_funding_unsigned.hex with your wallet
# Broadcast the signed tx via a mining pool that accepts non-standard tx
# Wait for confirmation, record: funding_txid, funding_vout (usually 0)
```

### 3 — Export GPU input files

After the funding tx confirms:

```bash
python3 qsb_pipeline.py export \
    --funding-txid <funding txid> \
    --funding-vout 0 \
    --funding-value 10000 \
    --dest-address $(printf '00%.0s' {1..20}) \
    --locktime 0 --sequence 4294967294 --version 1
```

This produces:
- `pinning.bin` + `gpu_pinning_params.json`
- `digest_r1.bin` + `gpu_digest_r1_params.json`
- `digest_r2.bin` + `gpu_digest_r2_params.json`

### 4 — Launch the fleet

Set up vast.ai credentials once:

```bash
pip install vastai
export VAST_API_KEY=<your key from https://cloud.vast.ai/manage-keys/>
```

**Before paying for 10 GPUs, do a cheap smoke test first:**

```bash
python3 ../fleet/qsb_fleet.py smoke --gpu RTX_3060 --max-dph 0.20
```

This rents 1 cheap GPU (~$0.05–$0.20), validates that:

- The bundle uploads correctly
- Both kernels compile with nvcc on a real CUDA host
- The kernels run and produce sensible output
- The instance is then destroyed (stops billing)

If the smoke test passes, launch the real fleet:

```bash
python3 ../fleet/qsb_fleet.py launch \
    --count 10 \
    --gpu RTX_4090 \
    --max-dph 0.50 \
    --gpu-dir ../gpu
```

This:

1. Searches vast.ai for offers matching your criteria
2. Prompts for confirmation (`-y` to skip)
3. Packages up all the input files + kernel sources into a tarball
4. Rents N instances, uploads the tarball
5. On each instance: compiles the kernels with nvcc, starts the search
   with a disjoint slice of the (sequence, locktime) space
6. Writes `qsb_fleet.json` locally so subsequent commands know your fleet

No cross-machine sync is required — each machine runs fully independently.
Different locktime ranges mean different searches mean first one to find a
hit wins.

### 5 — Monitor

```bash
python3 ../fleet/qsb_fleet.py status    # every ~30s, see progress
python3 ../fleet/qsb_fleet.py hits --out ./hits/   # pull hits if any
```

Total honest work expected: ~2^47 EC recoveries. At RTX 4090 speeds
(~10⁸ recoveries/sec/GPU), that's ~40 minutes per GPU. A 10-machine fleet
finishes in ~4 minutes of compute, plus boot/compile overhead.

### 6 — When a hit appears

```bash
python3 ../fleet/qsb_fleet.py hits --out ./hits/
ls hits/
```

Each instance's `hits/instance_<id>/` directory contains:
- `qsb_hits.jsonl` — one JSON line per hit
- `qsb_search.log`, `pin_gpu*.log`, `digest_r*_gpu*.log`

The JSONL is parseable by the pipeline. Pick the first pinning hit (any one
will do) and its corresponding R1, R2 digest subsets.

### 7 — **VERIFY** the hit on CPU

Before trusting any GPU result, cross-check on the CPU:

```bash
python3 ../verify/verify_hit.py pin \
    --locktime <from hit JSON> \
    --sequence 0xfffffffe \
    --funding-txid <funding txid>

python3 ../verify/verify_hit.py digest \
    --round 1 \
    --indices <comma-sep subset from hit JSON> \
    --locktime <pin locktime> --sequence 0xfffffffe \
    --funding-txid <funding txid>

# same for round 2
```

Each command either prints "✅ HIT CONFIRMED" or "❌ HIT DOES NOT REPRODUCE".
If any check fails, the GPU kernel and CPU pipeline have diverged — DO NOT
assemble the tx. Stop, investigate, and re-test before proceeding.

### 8 — Assemble + broadcast

If all three `verify_hit` checks pass:

```bash
python3 qsb_pipeline.py assemble \
    --locktime <pin locktime> --sequence 4294967294 \
    --round1 <R1 indices> --round2 <R2 indices> \
    --funding-txid <funding txid> --funding-vout 0 \
    --funding-value 10000
```

This produces `qsb_raw_tx.hex` — the final signed spending transaction.

Broadcast via the same mining pool you used for the funding tx.

### 9 — Tear down the fleet

Don't forget:

```bash
python3 ../fleet/qsb_fleet.py destroy
```

Otherwise you'll keep burning vast.ai credits.

## Bugs caught during development

The test harness caught several serious bugs in the original v16 code. All
fixed in this version; the test suite enforces they stay fixed:

| Bug | Impact | Where |
|-----|--------|-------|
| `count_opcodes` missed CHECKMULTISIG's runtime `+N` addition | Config D hit 221 ops, failed consensus | bitcoin_tx.py |
| `count_opcodes` used `>= 0x60` instead of `> 0x60` | Miscounted OP_16 pushes as opcodes | bitcoin_tx.py |
| `find_and_delete` scanned bytes, not opcodes | Could remove a pattern inside a data push | bitcoin_tx.py |
| `puzzle_hash('ripemd160')` did HASH160 instead of RIPEMD160 | Pipeline searched an impossible condition for Config Ar | qsb_pipeline.py |
| `cmd_export` assumed base_sc starts with HORS at offset 0 | Actually starts with pinning — offsets shifted by ~50 bytes, every GPU "hit" would be a false positive | qsb_pipeline.py |
| `cmd_export` wrote pinning.bin in wrong format for kernel's loader | Kernel would read garbage seq/lt offsets | qsb_pipeline.py |
| Kernel tried both SHA-256 hash iterations even when script only did one | Would report false hits for Config A | qsb_digest_search.cu, qsb_real_search.cu |

## What makes this safer than the v16 pipeline

1. **The test gate.** `python3 verify/test_all.py` must pass before any
   GPU spend. It catches every bug class above plus any future regression.

2. **The hit verifier.** `verify_hit.py` is the last-line defense between
   the GPU and the blockchain. Every claimed hit is re-run on CPU and
   byte-compared before being trusted.

3. **Hard failures at setup.** `cmd_setup` raises if the built script
   would be consensus-invalid. No silent acceptance of over-budget scripts.

4. **Single-hash Config A.** By moving from sha256_double to sha256-single,
   we eliminate an entire class of "hash_choice confusion" bugs from both
   the pipeline and the kernel, and simplify the GPU code.

## Troubleshooting

### "test_all.py failed"
Do not spend GPU money. Look at which phase failed:
- `preflight` — your config exceeds Bitcoin's op budget. Rerun
  `verify/verify_script_budget.py --config A --verbose` to see the breakdown.
- `equivalence` — a CPU/GPU contract broke. Check recent edits to
  `pipeline/bitcoin_tx.py` or `verify/gpu_emulator.py`.
- `e2e` — the pipeline's exported params don't round-trip through the emulator.
  Most likely cause: layout drift in `cmd_export`. Check the `pre_hors_section`,
  `hors_section`, `tail_section` split.
- `config_sanity` — one of the named configs changed sizes. Audit the config
  table in `qsb_pipeline.py`.

### "nvcc: command not found" on the GPU instance
The bootstrap script assumes the vast.ai image has CUDA installed. If it
doesn't, the instance will fail to compile the kernels. Check the image
requirements in `fleet/qsb_fleet.py` and pick an image with `nvcc` (most
recent PyTorch / CUDA images ship it).

### "HIT found" in a GPU log but `verify_hit.py` says invalid
**This is expected occasionally** — the kernel's "easy mode" accepts any
hash whose first nibble is 3, which includes many non-DER sequences. The
`verify_hit.py` step is the authoritative check. Only hits that pass
`verify_hit.py` should be used for assembly.

If *every* hit fails verification, something is wrong with the pipeline /
kernel contract. Do NOT run assembly. Investigate:
1. Check `qsb_state.json` matches the one used to produce the GPU inputs.
2. Rerun `test_all.py`.
3. Check that the funding-tx parameters given to `export` match the actual
   funding tx.

### "My fleet ran for 24 hours with no pin hit"
Pinning is the variable-cost phase — the locktime space has ~2^32 positions
but only ~2^23 give a valid DER, so expected work is ~2^(32-23) = ~500 GPU-
seconds per GPU before finding one. If you've searched way longer than that:
- Double-check `--lt-range` on `launch` — did you shard too narrowly?
- Check the kernel logs — is it actually processing locktimes, or stuck?
- Check your funding tx was broadcast and confirmed.

### "My funding tx isn't getting mined"
QSB funding outputs are large P2SH with unusual scripts, which standard
miners won't relay. Use a mining pool that accepts non-standard tx:
Slipstream (mara.com), F2Pool's direct-submit, or private arrangements.
See `FUNDING.md` for the full walkthrough.

### "I want to verify the GPU hit manually"
```
cd workdir/  # where qsb_state.json lives
python3 verify/verify_hit.py pin \
    --locktime <lt>         \
    --sequence <seq>        \
    --funding-txid <txid>   \
    --funding-vout 0
# then for each digest round:
python3 verify/verify_hit.py digest \
    --round 1               \
    --subset <i,j,k,...>    \
    --locktime <lt>         \
    --sequence <seq>        \
    --funding-txid <txid>   \
    --funding-vout 0
```
If all three verifications pass, you can safely run `assemble`.

### "I want to abort a running fleet"
```
python3 fleet/qsb_fleet.py destroy
```
This terminates all instances immediately. Vast.ai billing stops at destroy
(not stop). Use `stop` if you want to resume later; use `destroy` when done.

## Emergency stop

If anything looks wrong — wrong hit reports, unexpected costs, strange logs —
the safest action is:

```
python3 fleet/qsb_fleet.py destroy
```

Your funding UTXO is still yours (the assemble step hasn't run yet). You
can redo setup/export/fleet with corrected code without losing the funding.
The only thing spent so far is GPU time; the Bitcoin is still in the P2SH
UTXO waiting to be spent by a valid tx.
