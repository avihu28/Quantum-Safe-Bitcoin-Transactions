# Quantum Safe Bitcoin (QSB)

A quantum-safe Bitcoin transaction scheme using only existing consensus rules.

📄 **Paper**: [Quantum-Safe Bitcoin Transactions Without Softforks](https://github.com/avihu28/Quantum-Safe-Bitcoin-Transactions/blob/main/paper/QSB.pdf)

## Overview

Quantum Safe Bitcoin (QSB) enables signing Bitcoin transactions in a way that remains secure even against an adversary with a large-scale quantum computer running Shor's algorithm. The scheme requires **no changes to the Bitcoin protocol** — it operates entirely within the existing legacy script constraints (201 opcodes, 10,000 bytes).

### The Problem

Standard Bitcoin transactions rely on ECDSA signatures over the secp256k1 curve. Shor's algorithm can efficiently compute discrete logarithms, allowing a quantum adversary to forge ECDSA signatures — breaking the fundamental security assumption that protects Bitcoin transactions.

### Our Approach

QSB builds on [Binohash](https://robinlinus.com/binohash.pdf) (Linus, 2026), which uses a HORS-like one-time signature scheme embedded in Bitcoin Script. Binohash achieves transaction integrity through a proof-of-work puzzle based on signature sizes (`OP_SIZE`). However, this puzzle relies on the assumption that the smallest known ECDSA `r`-value cannot be improved — a quantum adversary running Shor's algorithm could compute the discrete logarithm of `r = 1`, breaking the puzzle entirely.

QSB replaces this with a **hash-to-signature puzzle**: the script hashes a transaction-bound public key via `OP_RIPEMD160` and interprets the 20-byte output as a DER-encoded ECDSA signature. A random 20-byte string satisfies the DER structural constraints with probability ~2^-46 — providing the proof-of-work target. Since this puzzle depends only on the pre-image resistance of RIPEMD-160 (not on any elliptic curve assumption), it is **fully resistant to Shor's algorithm**.

### Key Properties

- **Quantum safe**: Security relies on hash pre-image resistance, not ECDSA. ~118-bit second pre-image resistance under Shor; ~59-bit under Grover.
- **No protocol changes**: Uses only existing Bitcoin consensus rules (legacy P2SH transactions).
- **Practical cost**: ~$75–$150 in cloud GPU compute for the off-chain search, with embarrassingly parallel scaling.
- **Non-standard transaction**: Requires submission via a miner-direct service (e.g., Slipstream) since the transaction exceeds standard relay policy limits.

### How It Works

```
  TRANSACTION PINNING              DIGEST ROUND (×2)

  Hardcoded signature              Choose t of n dummy sigs
  commits to SIGHASH_ALL           (= the digest)
           |                                |
           v                                v
  Changing the tx changes          Verify Lamport signature
  the derived key                  (HORS preimages)
           |                                |
           v                                v
  +----------------------+        Subset determines sighash
  | Hash the key — must  |        (via FindAndDelete)
  | produce a valid sig  |                  |
  | (~2^46 work)         |                  v
  +----------------------+        +----------------------------+
                                  | Derive key, hash it —      |
  Any tx modification             | must produce valid sig     |
  requires new puzzle solve       | (hash-to-sig puzzle)       |
                                  +----------------------------+
```

The spending process has three phases:

1. **Transaction pinning**: Search over `(sequence, locktime)` pairs until the recovered public key's RIPEMD-160 hash is a valid DER signature. This pins the transaction to a specific set of parameters (~2^46 work).

2. **Digest rounds (×2)**: For the pinned transaction, search over subsets of dummy signatures. Each subset produces a different `scriptCode` (via `FindAndDelete`), yielding a different sighash and thus a different recovered public key. Find a subset whose recovered key hashes to a valid DER signature (~2^46 candidates per round).

3. **Assembly**: Recover all public keys, extract HORS preimages, and construct the final spending transaction with the full witness.

The indices of the selected dummy signatures in each round form a **digest** — a compact, collision-resistant identifier of the transaction, analogous to a hash-based signature.

### Why Is This Quantum Safe?

The security of standard Bitcoin transactions rests on ECDSA — broken by Shor's algorithm. QSB replaces every security-critical component with hash-based alternatives:

- **Transaction pinning** depends on RIPEMD-160 pre-image resistance, not ECDSA hardness. A quantum adversary gains no advantage from Shor's algorithm; only Grover's quadratic speedup applies.
- **The Lamport signature** (HORS) uses hash commitments — the spender reveals preimages of committed hashes, which a quantum computer cannot forge.
- **ECDSA is used only as a vehicle**, not as a security assumption. The scheme exploits the fact that Bitcoin Script can verify ECDSA signatures (`OP_CHECKSIG`), but the *hardness* comes from hashing, not from the elliptic curve.

The result: ~118-bit pre-image security under Shor (roughly halved under Grover), compared to 0-bit security for standard ECDSA transactions.

### The Hash-to-Signature Puzzle

The core innovation is the hash-to-signature puzzle. A DER-encoded ECDSA signature has rigid structural constraints — specific tag bytes (`0x30`, `0x02`), internally consistent length fields, and positive integer values. A random 20-byte string satisfies all of these with probability ~2^-46.

The puzzle works as follows: the locking script contains a hardcoded ECDSA signature `sig_nonce` with a known `(r, s)`. When the spender provides a public key `key_nonce`, the script:

1. Verifies `(sig_nonce, key_nonce)` via `OP_CHECKSIGVERIFY` — this binds `key_nonce` to the current transaction's sighash.
2. Computes `OP_RIPEMD160(key_nonce)` — producing a 20-byte hash.
3. Interprets this hash as a signature `sig_puzzle` and verifies it via another `OP_CHECKSIGVERIFY`.

Step 3 succeeds only if the hash happens to be valid DER — a ~2^-46 event. Since `key_nonce` is determined by the transaction (via step 1), modifying any part of the transaction changes `key_nonce`, changes the hash, and almost certainly breaks step 3. This is the proof-of-work: finding a transaction whose derived key hashes to valid DER.

### Constraints and Tradeoffs

The scheme operates under Bitcoin's tightest constraints:

- **201 non-push opcodes** — every opcode counts, limiting the number of signature selections per round.
- **10,000 byte script size** — must fit ~150 dummy signatures, ~150 hash commitments, and all verification logic across two rounds.
- **Bare script output** — the script exceeds P2SH's 520-byte redeem script limit, so it must be placed directly in the scriptPubKey.
- **Non-standard transaction** — exceeds default relay policy, requiring miner-direct submission (e.g., via Slipstream).

These constraints force careful parameter tuning. The "bonus key" optimization adds cheap subset selections (3 opcodes each vs. 9 for full selections) to match the combinatorial search space to the fixed ~2^46 puzzle target — eliminating grinding overhead while staying within the opcode budget.

## Repository Structure

```
├── paper/
│   ├── QSB.tex              # LaTeX source
│   └── QSB.pdf              # Compiled paper
├── gpu/                     # CUDA GPU search code
│   ├── qsb_allgpu.cu       # Pinning search (SHA-256d + EC recovery)
│   ├── qsb_digest_gpu.cu   # Digest search (subset enumeration + EC)
│   ├── qsb_search.cu       # Production search (reads binary params)
│   ├── qsb_params.h        # Binary param file reader
│   ├── GPUMath.h            # secp256k1 field arithmetic (CudaBrainSecp)
│   ├── GPUHash.h            # SHA-256 / RIPEMD-160 on GPU
│   ├── Makefile
│   ├── launch_multi_gpu.sh  # Multi-GPU launcher
│   └── run_pinning.sh       # Per-machine pinning search
├── pipeline/                # Python pipeline — AUTHORITATIVE (consensus-corrected)
│   ├── qsb_pipeline.py     # Full pipeline: setup → export → search → assemble
│   ├── bitcoin_tx.py        # Transaction construction, sighash, FindAndDelete
│   ├── secp256k1.py         # EC math, ECDSA sign/recover, DER encode/parse
│   ├── secp256k1_fast.py    # Fast EC math using coincurve
│   ├── benchmark.py         # Benchmarking and graduated tests
│   ├── qsb_run.py          # vast.ai fleet orchestration (multi-machine)
│   ├── run_qsb.sh          # All-in-one run script for vast.ai
│   ├── test_consensus_cpu.py     # Pure-Python consensus gate (real ECDSA)
│   └── test_bitcoinconsensus.py  # Gold-standard gate via libbitcoinconsensus
├── script/                  # Generated Scripts — STALE (pre-dates the OP_ROLL fix)
├── v16/                     # Current orchestrator — start here to run a search
│   ├── qsb_orchestrator_v16.py  # Rents hosts, uploads bundle, drives the search
│   ├── bundle/              # v16 CUDA kernels + search inputs (authoritative)
│   ├── make_bundle.sh       # Packs bundle/ into the uploaded qsb_v16.zip
│   ├── pipeline/            # STALE COPY — still carries the off-by-one (see below)
│   ├── results/             # Hits found against the pre-fix script — STALE
│   └── verify_r2_hit.py     # CPU re-derivation of a GPU hit
├── config_a/                # Config A tree: runbooks, regtest, verify/ (pipeline corrected)
├── verifier/                # Rust consensus verifier (libbitcoinconsensus)
├── transactions/            # Funding/spending txs (hex) — STALE (built pre-fix)
├── requirements.txt
└── README.md
```

### Which GPU sources are current?

`v16/bundle/` holds the kernels actually used by the current orchestrator. They
carry the `gpu_scalar_mulmod` carry-propagation fix that eliminated the R2 false
positives — see [`v16/README.md`](v16/README.md). The copies under `gpu/` and
`config_a/gpu/` predate that fix and are kept for reference.

### Which Python pipeline is current?

There are three copies of the Python transaction-building code. They are **not**
identical, and only the first is safe to build a real lock with:

| Tree | Status |
|------|--------|
| `pipeline/` | **Authoritative.** Model-derived `OP_ROLL` positions, `hash_mode`-aware puzzle, SIGHASH_SINGLE bug value `2**248`. |
| `config_a/pipeline/` | Corrected the same way; additionally retains the legacy hand-formula builder, used only for `hash_mode='sha256_double'`. |
| `v16/pipeline/` | **Stale — still carries the off-by-one.** Untouched by the position fix. Do not generate a lock from it. |

The fix (PR #4, merged in `2c91720`) replaced hand-derived stack-position
formulas with a live stack model. The old formulas omitted the `OP_0`
CHECKMULTISIG dummy, used a fixed commitment gap that must shrink each
iteration, and ignored cross-round drift — together producing an **unspendable**
lock. The same commit corrected the SIGHASH_SINGLE bug value from `1` to
`2**248` (Bitcoin Core returns `uint256::ONE`, whose little-endian bytes are
read big-endian by secp256k1's `msg32`).

`hash_mode='sha256_double'` (Config D) is **not** consensus-corrected in any
tree. `pipeline/` now raises rather than silently emitting a mismatched script.

### Stale generated artifacts

Everything below was generated **before** the position fix landed, so it encodes
the old `OP_ROLL` depths and must be regenerated before use:

- `script/script_*.txt` — generated locking scripts
- `transactions/*.hex` — funding/spending txs built from those scripts
- `pipeline/*.bin`, `pipeline/gpu_*_params.json` — GPU search inputs
- `pipeline/qsb_scriptpubkey.hex`, `pipeline/qsb_funding_tx.hex`
- `v16/results/*.txt` — hits found against the pre-fix script

Regenerate via `pipeline/qsb_pipeline.py setup` / `export`, and gate the result
on `verifier/` (libbitcoinconsensus) before committing funds.

### Files not in this repository

`qsb_state.json`, `qsb_solution.json`, and the assembled spending transaction
are deliberately excluded: they contain the HORS preimages and ECDSA nonces that
constitute the spending witness. Regenerate them with `pipeline/qsb_pipeline.py`.

## Status

This is a work in progress. The current implementation covers:

- ✅ **Paper**: Full technical description of the QSB scheme
- ✅ **Script generation**: Complete Bitcoin Script for all configurations
- ✅ **GPU pinning search**: Implemented and tested on cloud GPUs (238M/s on RTX PRO 6000). Successfully found a real DER hit at `sequence=151205, locktime=656535577` after ~6 hours on 8 GPUs.
- ⬜ **GPU digest search**: Implemented but not yet tested end-to-end with real transaction parameters
- ⬜ **Transaction assembly**: Pipeline code exists but awaits a completed digest search
- ⬜ **On-chain broadcast**: Not yet attempted

See [`gpu/README.md`](gpu/README.md) for build instructions and usage.

## Configuration

| Config | n | t1 | t2 | Opcodes | Digest | Pre-image | Cost |
|--------|---|----|----|---------|--------|-----------|------|
| Baseline | 150 | 8 | 8 | 197 | 84.5b | 2^138 | ~$75-150 |
| **Config A** | **150** | **8+1b** | **7+2b** | **201** | **80.4b** | **2^118** | **~$75-150** |

## Measured Performance

| GPU | Pinning rate | Digest rate |
|-----|-------------|-------------|
| RTX 4070 SUPER | 88 M/s | 82 M/s |
| RTX PRO 6000 (Blackwell) | 238 M/s | ~160 M/s (est.) |

## Cost Breakdown (Config A)

| Phase | Candidates | Est. cost |
|-------|-----------|-----------| 
| Pinning | ~2^46.4 | $25–$50 |
| Digest round 1 | C(150,9) ≈ 2^46.2 | $25–$50 |
| Digest round 2 | C(150,9) ≈ 2^46.2 | $25–$50 |
| **Total** | | **$75–$150** |

The computation is embarrassingly parallel — wall-clock time scales inversely with the number of GPUs.

## Key Technical Details

- **DER probability**: 2^-46.4 at consensus level (sighash byte unconstrained; `SCRIPT_VERIFY_STRICTENC` is policy-only)
- **Search space**: sequence (32 bits) × locktime (32 bits) = 2^64 candidates for pinning
- **Midstate trick**: SHA-256 precomputation over ~5KB fixed scriptCode prefix reduces per-candidate cost to 2-3 SHA-256 blocks
- **EC recovery**: CudaBrainSecp precomputed GTable (16 chunks × 65536 points) for fast scalar multiplication on GPU

## References

- [Binohash: Signing Bitcoin Transactions via Proof of Work](https://robinlinus.com/binohash.pdf) — Robin Linus, 2026
- [Signing a Bitcoin Transaction with Lamport Signatures (no OP_CAT)](https://groups.google.com/g/bitcoindev/c/mR53go5gHIk) — Ethan Heilman, 2024

## License

MIT — see [LICENSE](LICENSE).

Note: `gpu/GPUMath.h` and `gpu/GPUHash.h` are from [CudaBrainSecp](https://github.com/nicecash/CudaBrainSecp) (Jean Luc PONS / VanitySearch) and are licensed under GPL-3.0. All other files in this repository are MIT licensed.
