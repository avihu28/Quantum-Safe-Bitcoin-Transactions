# QSB Consensus Verifier

Runs Bitcoin Core's **actual libbitcoinconsensus** (via the well-maintained
rust-bitcoinconsensus crate) against your QSB tx to prove it's consensus-valid.

This is the same library a Bitcoin node uses for script verification. It runs
**consensus rules only** (no standardness checks), which is exactly what you
need — your tx is non-standard by design but must be consensus-valid.

## One-time setup

If you don't have Rust installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Accept defaults, then:
source "$HOME/.cargo/env"
```

## Build & run

```bash
# 1. Unzip this folder somewhere
cd qsb_verify

# 2. Make sure your funding tx hex is cached
curl -s https://mempool.space/api/tx/4fab76e9b0538a49a77443030f8e0243a5d2558155647a839acea0efaa4edc91/hex \
  | tr -d '\n' > /tmp/funding_tx.hex

# 3. Build (first build clones Bitcoin Core source, compiles libbitcoinconsensus.a,
#    and links it statically. ~3-8 minutes on first build, instant thereafter.)
cargo build --release

# 4. Run
./target/release/qsb_verify \
    ~/Downloads/qsb_v16/pipeline/qsb_raw_tx.hex \
    /tmp/funding_tx.hex \
    0 0
```

## Expected output

**If consensus-valid:**
```
Funding vout 0: value = 10000 sats, scriptPubKey = <N> bytes
Spending tx size: 1237 bytes

=== Consensus verification (VERIFY_ALL flags) ===
✅ SCRIPT SUCCEEDED — transaction is CONSENSUS-VALID
   (it may still be non-standard for mempool policy,
    but a miner like Slipstream can include it in a block)
```

**If script fails:**
```
❌ SCRIPT FAILED: SomeError
Trying with minimal flags (just P2SH)...
  With P2SH-only: <result>
  With NO flags: <result>
```
The specific error tells us exactly which consensus rule was violated.

## Why this is better than btcdeb

- btcdeb has interactive-mode quirks that make automation painful
- rust-bitcoinconsensus is literally **the same C++ code as Bitcoin Core**,
  just wrapped in Rust for easier embedding
- Runs headless, returns a clean success/error result
- No syncing needed, no policy checks, just pure consensus

## Troubleshooting

- **Build fails "no C compiler"**: `xcode-select --install`
- **Build fails on clone**: needs network access to github.com
- **`error[E0658]`** (nightly features): `rustup default stable`
