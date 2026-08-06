# Regtest validation harness

Run our pipeline through a real Bitcoin Core consensus engine *before* spending
any GPU money or broadcasting to mainnet. Same code Bitcoin uses to validate
mainnet blocks; same yes/no answer.

## Setup (one-time)

Install Bitcoin Core (`bitcoind` + `bitcoin-cli`) on your machine. On macOS:

```bash
brew install bitcoin
```

On Linux:

```bash
sudo apt install bitcoind   # or download from bitcoin.org
```

Verify it's on PATH:

```bash
bitcoind --version
```

## What this validates

| Phase | Cost | Time | Validates |
|-------|------|------|-----------|
| **1.** Funding tx (Tx1) | $0 | seconds | The QSB scriptPubKey is well-formed and accepted by Bitcoin Core when locked as an output. |
| **3.** Spending tx (Tx2) | requires GPU output | seconds + GPU time | The full assembled Tx2 passes Bitcoin Core's consensus engine — script execution, all signature verifications, byte layout, sigop count, F&D semantics, sighash, everything. |

(Phase 2, a structural test without real DER hits, is skipped because brute-forcing a
strict-DER hit on CPU is infeasible. Phase 1 + Phase 3 together give equivalent coverage.)

## Running

All commands assume you're running from a single working directory (e.g.,
`~/qsb_run/`) that will hold the pipeline state, the funding info, and any GPU
output files. The regtest helper scripts live in this `regtest/` directory of
the repo and are referenced by absolute path.

### One-time per session: start the regtest node

```bash
# Start a fresh regtest node + wallet with 50 BTC of mature coins
~/qsb_config_a/regtest/setup_regtest.sh start
```

The node listens on RPC port 18443. Data is in `/tmp/qsb_regtest`. You can
override either via `QSB_REGTEST_RPC_PORT` and `QSB_REGTEST_DIR`.

### Phase 1 — funding tx

```bash
# (Once) generate the QSB state
mkdir ~/qsb_run && cd ~/qsb_run
python3 ~/qsb_config_a/pipeline/qsb_pipeline.py setup --config A
ls qsb_state.json   # should exist

# Run the funding-tx test
python3 ~/qsb_config_a/regtest/test_funding_tx.py
```

Expected output:

```
  ✅ PHASE 1 PASSED
  - Bitcoin Core accepted the QSB scriptPubKey output
  - tx confirmed in regtest block
```

This writes `regtest_funding.json` to the cwd. Use that file's `funding_txid`
as the input to your GPU search.

### Phase 3 — spending tx

You need real GPU search results before this step. Format them as
`gpu_hits.json`:

```json
{
  "pin_locktime": 12345,
  "pin_sequence": "0xfffffffe",
  "round1_indices": [0, 5, 12, 23, 41, 67, 89, 121, 143],
  "round2_indices": [1, 8, 17, 39, 52, 78, 91, 102, 119]
}
```

Then run (still from `~/qsb_run`, with both `qsb_state.json` and
`regtest_funding.json` present):

```bash
python3 ~/qsb_config_a/regtest/test_spending_tx.py --hits gpu_hits.json
```

Or with CLI flags:

```bash
python3 ~/qsb_config_a/regtest/test_spending_tx.py \
    --pin-locktime 12345 \
    --pin-sequence 0xfffffffe \
    --r1-indices 0,5,12,23,41,67,89,121,143 \
    --r2-indices 1,8,17,39,52,78,91,102,119
```

Expected output on success:

```
  ✅ PHASE 3 PASSED — SPENDING TX IS CONSENSUS-VALID

  Bitcoin Core's full consensus engine has validated:
    - Script structure (op count at runtime, byte layout, F&D)
    - All ECDSA signatures (pinning, sig_puzzle, both digest rounds)
    - Sighash computation
    - Transaction structure (size, sigops, etc.)
```

If Phase 3 fails, the rejection reason from `testmempoolaccept` tells you
exactly which consensus rule was violated. Fix in the pipeline, re-test,
proceed to mainnet.

### Tear down

```bash
~/qsb_config_a/regtest/setup_regtest.sh stop
# Removes /tmp/qsb_regtest entirely. Wallet + chain state gone.
```

## End-to-end workflow

```
1. python3 verify/test_all.py                       # CPU test suite (free)
2. ~/qsb_config_a/regtest/setup_regtest.sh start    # start regtest node
3. python3 ~/qsb_config_a/regtest/test_funding_tx.py # Phase 1 (no GPU)
4. (run GPU search using regtest funding txid; ~$10-20 with small fleet)
5. python3 ~/qsb_config_a/regtest/test_spending_tx.py --hits ...  # Phase 3
6. ~/qsb_config_a/regtest/setup_regtest.sh stop     # clean up
7. (run GPU search using mainnet funding txid; ~$100 with full fleet)
8. broadcast Tx1 + Tx2 via Slipstream
```

Step 4 is necessary because the GPU search results are tied to a specific
funding txid. The regtest run produces a different txid than mainnet.

If you want to skip step 4 (one GPU search instead of two), accept that you
won't have full Bitcoin-Core consensus validation of the actual mainnet Tx2
before broadcast. Phase 1 + extensive CPU tests still catch most issues — but
the most consequential bug class (runtime script-execution failures) is exactly
what Phase 3 catches and the CPU tests cannot.

## Common rejection reasons (for Phase 3)

| reject-reason | meaning |
|---|---|
| `mandatory-script-verify-flag-failed (Operation not valid with the current stack size)` | Script tried to operate on more stack items than exist. Likely a witness construction bug. |
| `mandatory-script-verify-flag-failed (Script failed an OP_VERIFY operation)` | An OP_*_VERIFY popped 0/false. Usually means a sig didn't verify. |
| `mandatory-script-verify-flag-failed (Operation reserved or invalid)` | Disabled opcode encountered. Pipeline bug — shouldn't happen with our builder. |
| `mandatory-script-verify-flag-failed (Operation not valid with the current op count)` | The MAX_OPS_PER_SCRIPT bug that bit Config D. Should never happen with Config A (we have a preflight verifier). |
| `bad-txns-vin-empty` / `bad-txns-vout-empty` | Tx structure invalid. Pipeline bug. |
| `bad-txns-too-many-sigops` | Tx exceeds MAX_BLOCK_SIGOPS_COST/5. Should never happen with Config A. |
| `non-mandatory-script-verify-flag-failed` | Standardness-only failure. Should NOT appear in regtest with `acceptnonstdtxn=1`. If it does, the regtest config is wrong. |

## Caveats

- Regtest validates **consensus**, not **mempool relay policy**. A consensus-valid tx
  can still be rejected by some nodes' standardness rules. Slipstream and similar
  direct-to-miner endpoints bypass standardness checks, which is exactly what we want.
- Soft forks: regtest pre-activates all soft forks. This matches current mainnet,
  so consensus rules are equivalent.
- The regtest data dir is `/tmp/qsb_regtest` by default. Override with
  `QSB_REGTEST_DIR=/some/other/path`.
