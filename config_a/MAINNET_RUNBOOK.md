# QSB Config A — Mainnet Runbook (FINAL — 50K QSB / 18 sat-vB on TX2)

## Numbers locked in

```
TX1 (funding):
  input:    143,534 sats (4fab76e9...:1, P2WPKH, your address)
  output 0: 50,000 sats (QSB bare script)  
  output 1: 63,402 sats (P2WPKH change → bc1qse6vtq...)
  fee:      30,132 sats (3 sat/vB on 10,044 vB)

TX2 (spending):
  input 0:  63,402 sats (TX1 vout 1, your change UTXO — UNSIGNED at assembly)
  input 1:  50,000 sats (TX1 vout 0, the QSB output)
  output:   89,660 sats → bc1qse6vtq... (your address)
  fee:      23,742 sats (18 sat/vB on 1,319 vB — generous for Slipstream)

Net redemption: 89,660 sats (~$90 at $100k/BTC)
Total fees:     53,874 sats (~$54)
GPU search cost: ~$300-500
```

## Pre-flight checklist

- [x] `~/qsb_run/qsb_state.json` exists (Apr 28, Config A) ✓
- [x] `~/qsb_run/qsb_scriptpubkey.hex` exists (9923 bytes) ✓
- [ ] You have the privkey for `bc1qse6vtqgaemyqs2cn73tkrzehl0hwz0ggfs8kzj` 
      loaded in some signing wallet (Sparrow / Electrum / bitcoind / Ledger)
- [ ] Slipstream API endpoint accessible

---

## STEP 1 — Refresh local files

```bash
cd ~/qsb_run
rm -rf ~/Downloads/qsb_config_a
tar -xzf ~/Downloads/qsb_config_a.tar.gz -C ~/Downloads/
```

⚠ **Do NOT run `setup` again.** Use your existing `qsb_state.json`.

---

## STEP 2 — Build unsigned funding tx (TX1)

```bash
cd ~/qsb_run

python3 ~/Downloads/qsb_config_a/pipeline/qsb_pipeline.py fund \
    --input-txid 4fab76e9b0538a49a77443030f8e0243a5d2558155647a839acea0efaa4edc91 \
    --input-vout 1 \
    --input-value 143534 \
    --qsb-value 50000 \
    --change-address bc1qse6vtqgaemyqs2cn73tkrzehl0hwz0ggfs8kzj \
    --fee 30132
```

This writes `qsb_funding_unsigned.hex` (10,016 bytes).

---

## STEP 3 — Compute the funding txid (still unsigned)

```bash
python3 ~/Downloads/qsb_config_a/pipeline/compute_funding_txid.py \
    qsb_funding_unsigned.hex
```

You should get something looking like:
```
Computed funding txid: <64 hex chars>
```

**Save this as the variable for following steps:**

```bash
export FUNDING_TXID=<paste from output>
```

---

## STEP 4 — Export GPU search params

```bash
python3 ~/Downloads/qsb_config_a/pipeline/qsb_pipeline.py export \
    --funding-txid $FUNDING_TXID \
    --funding-vout 0 \
    --funding-value 50000 \
    --extra-input-txid $FUNDING_TXID \
    --extra-input-vout 1 \
    --extra-input-value 63402 \
    --extra-input-sequence 0xfffffffd \
    --output-value 89660 \
    --output-address bc1qse6vtqgaemyqs2cn73tkrzehl0hwz0ggfs8kzj \
    --version 2 \
    --sequence 0xfffffffe \
    --locktime 0
```

Should print "Pinning ... R1 ... R2" lines all green.

Files produced:
- `gpu_pinning_params.json` + `pinning.bin`
- `gpu_digest_r1_params.json` + `digest_r1.bin`
- `gpu_digest_r2_params.json` + `digest_r2.bin`

---

## STEPS 5 & 6 in PARALLEL

### STEP 5 — Launch GPU fleet (start it before broadcasting funding)

In one terminal:

```bash
# Provision 16 machines × 4 GPUs = 64 GPUs (or whatever your budget allows)
python3 ~/Downloads/qsb_config_a/fleet/qsb_fleet.py provision \
    --num-machines 16 --gpus-per-machine 4 \
    --gpu-type RTX_4090 --max-hourly 5.0

# Wait for instances to come up (5-10 min). Check:
python3 ~/Downloads/qsb_config_a/fleet/qsb_fleet.py status

# Once all running:
python3 ~/Downloads/qsb_config_a/fleet/qsb_fleet.py start

# Upload params + start search
python3 ~/Downloads/qsb_config_a/fleet/qsb_fleet.py upload-search-params
python3 ~/Downloads/qsb_config_a/fleet/qsb_fleet.py search
```

While it runs, monitor:

```bash
watch -n 60 'python3 ~/Downloads/qsb_config_a/fleet/qsb_fleet.py check-results'
```

### STEP 6 — Sign and broadcast funding tx

In another terminal, sign `qsb_funding_unsigned.hex` with your wallet.

#### Option A: bitcoin-cli (with your privkey loaded)

```bash
bitcoin-cli signrawtransactionwithwallet $(cat qsb_funding_unsigned.hex) \
    '[{"txid":"4fab76e9b0538a49a77443030f8e0243a5d2558155647a839acea0efaa4edc91", "vout":1, "scriptPubKey":"00148674c5811dcec8082b13f457618b37fbeee13d08", "amount":0.00143534}]'
```

The output's `hex` field is your signed tx.

#### Option B: Sparrow / Electrum

1. Open wallet
2. File → "Load Transaction" → paste the contents of `qsb_funding_unsigned.hex`
3. Wallet should detect the input as yours; click Sign
4. Export the signed hex

#### Verify the signed txid matches

```bash
SIGNED_TX=<paste your signed hex>
echo $SIGNED_TX | python3 -c "
import hashlib, sys, struct
raw = bytes.fromhex(sys.stdin.read().strip())
# strip witness (segwit marker 0x0001 at offset 4 means there's witness data)
if raw[4:6] == b'\\x00\\x01':
    # parse non-segwit form by removing marker + witnesses
    # Easier: just print txid via slow pure-python parsing OR trust your
    # wallet's UI. Most wallets show the txid before broadcast.
    print('(witness present — please verify txid in your wallet UI)')
else:
    txid = hashlib.sha256(hashlib.sha256(raw).digest()).digest()[::-1]
    print(f'txid: {txid.hex()}')
"
```

It MUST equal `$FUNDING_TXID`. If not, STOP and re-check.

#### Broadcast via Slipstream

```bash
curl -X POST https://slipstream.mara.com/api/transactions \
    -H 'Content-Type: application/json' \
    -d "{\"tx_hex\":\"$SIGNED_TX\"}"
```

Or use the web UI at https://slipstream.mara.com if API is locked.

After submission, watch:

```bash
watch -n 30 "curl -s https://mempool.space/api/tx/$FUNDING_TXID/status | python3 -m json.tool"
```

When `"confirmed": true`, TX1 is mined. (This may take 10-60 min depending 
on MARA's block schedule.)

---

## STEP 7 — wait for GPU search hits

The fleet runs pin → R1 → R2 sequentially. Expected timeline:
- Pin: ~30 min
- R1: 1-2 hours
- R2: 1-2 hours
- **Total: 2.5-4.5 hours after kickoff**

You'll see hits like:
```
[GPU 23] PIN HIT: lt=887083300 seq=0x8000dfd2
[GPU 41] R1  HIT: indices=[14,32,41,45,63,82,86,95,108]
[GPU 7]  R2  HIT: indices=[3,26,32,40,60,64,109,110,129]
```

For each, run verifier:

```bash
LT=<from PIN HIT>
SEQ=<from PIN HIT, can be 0xHEX or decimal>

python3 ~/Downloads/qsb_config_a/verify/verify_hit.py pin \
    --locktime $LT --sequence $SEQ \
    --funding-txid $FUNDING_TXID --funding-vout 0 \
    --funding-value 50000 --version 2

python3 ~/Downloads/qsb_config_a/verify/verify_digest_against_kernel.py \
    --round 1 --indices "14,32,41,45,63,82,86,95,108" \
    --locktime $LT --sequence $SEQ

python3 ~/Downloads/qsb_config_a/verify/verify_digest_against_kernel.py \
    --round 2 --indices "3,26,32,40,60,64,109,110,129" \
    --locktime $LT --sequence $SEQ
```

All three should report ✓ matches the kernel.

---

## STEP 8 — assemble the spending tx

```bash
python3 ~/Downloads/qsb_config_a/pipeline/qsb_pipeline.py assemble \
    --locktime $LT \
    --sequence $SEQ \
    --version 2 \
    --round1 "14,32,41,45,63,82,86,95,108" \
    --round2 "3,26,32,40,60,64,109,110,129" \
    --funding-txid $FUNDING_TXID \
    --funding-vout 0 \
    --funding-value 50000 \
    --extra-input-txid $FUNDING_TXID \
    --extra-input-vout 1 \
    --extra-input-value 63402 \
    --extra-input-sequence 0xfffffffd \
    --output-value 89660 \
    --output-address bc1qse6vtqgaemyqs2cn73tkrzehl0hwz0ggfs8kzj
```

Replace the indices with what the verifier confirmed.

Output: `qsb_raw_tx.hex` — tx with input[0] UNSIGNED, input[1] (QSB) signed
via the puzzle solution.

---

## STEP 9 — sign input[0] with your wallet

### bitcoin-cli

```bash
bitcoin-cli signrawtransactionwithwallet $(cat qsb_raw_tx.hex) \
    "[{\"txid\":\"$FUNDING_TXID\",\"vout\":1,\"scriptPubKey\":\"00148674c5811dcec8082b13f457618b37fbeee13d08\",\"amount\":0.00063402}]"
```

The output's `hex` is your fully-signed spending tx. Verify `complete: true`.

### Sparrow / Electrum

Load `qsb_raw_tx.hex`. The wallet should:
- Detect input[0] (the change UTXO) as yours and prompt for signing
- Leave input[1] (QSB) untouched (its scriptSig is huge and pre-set)

⚠ Do NOT click "Re-sign all inputs" or similar — only sign input[0].

---

## STEP 10 — broadcast TX2 via Slipstream

```bash
SIGNED_TX2=<paste your fully-signed spending tx hex>

curl -X POST https://slipstream.mara.com/api/transactions \
    -H 'Content-Type: application/json' \
    -d "{\"tx_hex\":\"$SIGNED_TX2\"}"
```

Monitor confirmation:

```bash
SPENDING_TXID=<txid printed by your wallet or compute via python>
watch -n 30 "curl -s https://mempool.space/api/tx/$SPENDING_TXID/status | python3 -m json.tool"
```

When confirmed, check your address:
- https://mempool.space/address/bc1qse6vtqgaemyqs2cn73tkrzehl0hwz0ggfs8kzj
- You should see a new 89,660 sat UTXO 🎉

---

## TROUBLESHOOTING

### "GPU search has been running 6+ hours, no hit yet"

Check `qsb_fleet.py check-results` — there's a small probability (~17% per
fully-exhausted pin) that no hit exists in this pin's search space. If the
fleet has fully exhausted R1 with no hits, you'll need a new pin. Run:

```bash
python3 ~/Downloads/qsb_config_a/fleet/qsb_fleet.py kill-kernels
python3 ~/Downloads/qsb_config_a/fleet/qsb_fleet.py search --start-round 0
```

This restarts pin search from scratch with a fresh seed.

### "Slipstream rejected TX1"

Probably the fee is too low for current Slipstream pricing. Bump TX1's fee
by repeating Steps 2-6 with a higher `--fee`. Note: this CHANGES the funding
txid, so you'd also need to redo Step 4 (re-export GPU params) and restart
the GPU search.

### "TX2 mined but with wrong outputs / fee"

This shouldn't be possible — the puzzle witness commits to the exact
output and fee via the sighash. If somehow the tx mined differently, the
script wouldn't have verified. You're safe.
