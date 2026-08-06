#!/bin/bash
# One-click Config A setup.
# Run this ONCE at the start. Produces:
#   - qsb_state.json            (HORS secrets, dummy sigs, full script)
#   - qsb_scriptpubkey.hex      (the locking script to fund)
#   - Prints the funding instruction.
set -e

cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════════════════════════"
echo "  QSB Config A — One-click setup"
echo "═══════════════════════════════════════════════════════════════════"
echo

# Preflight: build the locking script and verify its op budget BEFORE doing anything else.
python3 ../verify/verify_script_budget.py --config A || { echo "Preflight FAILED."; exit 1; }
echo

# Setup. The --seed is optional — remove it for fresh randomness.
python3 qsb_pipeline.py setup --config A

SCRIPT_SIZE=$(wc -c < qsb_scriptpubkey.hex)
echo
echo "═══════════════════════════════════════════════════════════════════"
echo "  Setup complete."
echo "═══════════════════════════════════════════════════════════════════"
echo
echo "  Locking script size (hex): $SCRIPT_SIZE chars"
echo "  Locking script size (bin): $((SCRIPT_SIZE / 2)) bytes"
echo
echo "  Next step: fund this script with ~10,000 sats."
echo "  The scriptPubKey hex is in qsb_scriptpubkey.hex."
echo
echo "  If you have an existing UTXO and bitcoin-cli:"
echo "    python3 qsb_pipeline.py fund \\"
echo "        --input-txid <txid> \\"
echo "        --input-vout <n> \\"
echo "        --input-value <sats> \\"
echo "        --qsb-value 10000 \\"
echo "        --change-address <20-byte hex pubkeyhash>"
echo
