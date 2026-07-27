# Funding transaction walkthrough

This document walks through creating and broadcasting the on-chain transaction
that locks your 10,000 sats into the QSB output. Read carefully — a mistake
here means the funds land in a script that cannot be spent by anything (or
that can be spent by the wrong party).

## Prerequisites

1. You have run `qsb_pipeline.py setup --config A`. This produced:
   - `qsb_state.json` — your HORS secrets, dummy signatures, and the full script
   - `qsb_scriptpubkey.hex` — hex of the locking script (~9.9 KB)

2. You have an existing P2PKH or P2WPKH UTXO with at least **12,000 sats** to
   fund the tx (10,000 for the QSB output + ~2,000 fee).

3. **You have run `python3 verify/test_all.py` and all 4 phases passed.**
   If any test failed, do NOT proceed — a bug could cause you to fund an
   unspendable script.

## Why a bare script output?

Most Bitcoin outputs today are P2SH or P2WSH wrappers around a redeem script.
QSB uses a **bare** output (scriptPubKey = the full QSB script, no hash wrapper)
because:

- The locking script already exceeds 520 bytes, which is the P2SH redeem
  script size limit. So P2SH is not an option.
- P2WSH allows larger scripts but introduces witness-stack semantics that don't
  match QSB's existing witness layout.
- Bare script outputs are non-standard (most nodes won't relay them) but are
  still valid in a block, which is all we need.

**Consequence: you cannot broadcast this tx to the regular mempool via a
standard node.** You must submit directly to a miner (see "Broadcasting" below).

## Step 1 — Build the unsigned funding tx

```bash
cd path/to/pipeline
python3 qsb_pipeline.py fund \
    --input-txid   <txid of your funding UTXO> \
    --input-vout   <output index> \
    --input-value  <value in sats> \
    --qsb-value    10000 \
    --change-address <your change pubkey hash, 20 bytes hex>
```

This writes `qsb_funding_unsigned.hex`. Inspect it:

```bash
python3 -c "
import sys
hex_tx = open('qsb_funding_unsigned.hex').read().strip()
print(f'Size: {len(hex_tx) // 2} bytes')
print(f'First 100 bytes: {hex_tx[:200]}')
print(f'Last 100 bytes:  {hex_tx[-200:]}')
"
```

### What to check

1. **Output count** should be 2 (QSB output + change) or 1 (if change < dust).
2. **QSB output's scriptPubKey** should match `qsb_scriptpubkey.hex` exactly.
3. **Value at QSB output** should be exactly 10,000 sats (`0x2710` = `10270000000000` little-endian).
4. **Change value** = input_value − 10000 − fee. Verify the math.

Don't skip this. You can decode the tx with `bitcoin-cli decoderawtransaction`
or any online decoder, but check byte-by-byte that the QSB output's script is
identical to `qsb_scriptpubkey.hex`.

## Step 2 — Sign the funding tx

If your input UTXO is in a Bitcoin Core wallet:

```bash
bitcoin-cli signrawtransactionwithwallet $(cat qsb_funding_unsigned.hex)
```

Save the returned `hex` — this is the signed tx.

If it's in a hardware wallet or Electrum, import the unsigned hex and sign
using that wallet's normal flow.

### Important: verify the signed tx before broadcasting

The SIGNED tx should:
- Have the **same outputs** as the unsigned tx (scripts and values unchanged)
- Have a **valid signature** on input 0

You can check with:

```bash
bitcoin-cli decoderawtransaction <signed_hex>
```

and visually compare the `vout` array against the unsigned version.

## Step 3 — Broadcast

Because this tx is non-standard, you have three options:

### Option A — Direct miner submission (recommended)

Use [MARA Slipstream](https://slipstream.mara.com/), F2Pool's direct-to-miner
endpoint, or another mining pool that accepts non-standard tx:

```bash
curl -X POST https://slipstream.mara.com/inbox -d 'signed_hex=<signed_hex>'
```

(Check the current endpoint on their site; these URLs do change.)

### Option B — Your own mining setup

If you mine yourself, include the tx in a block you mine.

### Option C — Regular node + prayer

`bitcoin-cli sendrawtransaction <signed_hex> true` with `maxfeerate=0`.
Most nodes will reject this as non-standard. Even if one node accepts it,
they won't relay it. You'd need to reach a miner anyway.

## Step 4 — Confirm + record

Once the tx is mined, grab:

- `funding_txid` = the txid (you'll use this in subsequent pipeline commands)
- `funding_vout` = the output index (usually 0; the QSB output)
- `funding_value` = 10000 (the sats that went into the QSB output)

Now you're ready to run the GPU search.

## Debugging if something goes wrong

### "My funding tx never confirmed"

The tx may have been silently dropped. Options:
- Wait 24 hours, then consider it lost/unused — your input UTXO is still
  spendable; just make a new funding attempt with a different pipeline setup.
- Check the mining pool's status page.
- If you have the txid, query it on a block explorer — if it shows as
  "unconfirmed", it's in a mempool somewhere; if not found at all, it was
  dropped.

### "I confirmed the wrong tx"

If you accidentally broadcast a funding tx that locked funds into a bug-ridden
script (e.g., you didn't run `test_all.py` first):

**Do not assume the funds are lost** — run `test_all.py`; if the script is
actually correct (e.g., you had a false alarm), you can still proceed.

If the script is genuinely broken, the 10,000 sats in the QSB output are
**permanently stuck**. This is why you must run `test_all.py` before funding.
The change output (if any) is still spendable normally.
