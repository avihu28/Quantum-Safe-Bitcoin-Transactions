#!/usr/bin/env python3
"""
Verify the assembled QSB transaction LOCALLY using python-bitcoinlib's VerifyScript.

Usage:
  python3 verify_tx_locally.py [path_to_raw_tx_hex] [path_to_qsb_state.json]
"""
import sys
import json
import traceback
from pathlib import Path

TX_HEX_PATH = sys.argv[1] if len(sys.argv) > 1 else "qsb_raw_tx.hex"
STATE_PATH = sys.argv[2] if len(sys.argv) > 2 else "qsb_state.json"

print("=== QSB Transaction Local Verification ===\n")

# Robust import
try:
    from bitcoin.core import CTransaction
    from bitcoin.core.script import CScript
    from bitcoin.core.scripteval import VerifyScript
except ImportError as e:
    print(f"python-bitcoinlib import failed: {e}")
    print()
    print("Try:")
    print("  python3 -m pip install python-bitcoinlib")
    sys.exit(1)

# Try to import flag constants; different versions expose different sets.
SCRIPT_FLAGS = set()
try:
    from bitcoin.core.scripteval import SCRIPT_VERIFY_P2SH
    SCRIPT_FLAGS.add(SCRIPT_VERIFY_P2SH)
    print("  using SCRIPT_VERIFY_P2SH")
except ImportError:
    print("  no SCRIPT_VERIFY_P2SH (OK, using defaults)")

# Load tx
with open(TX_HEX_PATH) as f:
    tx_hex = f.read().strip()
tx_bytes = bytes.fromhex(tx_hex)
tx = CTransaction.deserialize(tx_bytes)

print(f"\nTransaction loaded:")
print(f"  Version:  {tx.nVersion}")
print(f"  Locktime: {tx.nLockTime}")
print(f"  Inputs:   {len(tx.vin)}")
print(f"  Outputs:  {len(tx.vout)}")
print(f"  Size:     {len(tx_bytes)} bytes")
print()

# Locking script: try qsb_solution.json first, fall back to qsb_state.json
locking_script_hex = None
sol_path = Path("qsb_solution.json")
if sol_path.exists():
    try:
        sol = json.load(open(sol_path))
        locking_script_hex = sol.get('locking_script_hex') or sol.get('full_script_hex')
    except Exception:
        pass

if not locking_script_hex:
    state_path = Path(STATE_PATH)
    if state_path.exists():
        try:
            state = json.load(open(state_path))
            locking_script_hex = state.get('full_script_hex')
        except Exception:
            pass

if not locking_script_hex:
    print("ERROR: Could not find locking script (tried qsb_solution.json and qsb_state.json)")
    sys.exit(1)

locking_script = CScript(bytes.fromhex(locking_script_hex))
print(f"Locking script loaded: {len(locking_script)} bytes")
print()

# ScriptSig from input 0
script_sig = tx.vin[0].scriptSig
print(f"ScriptSig: {len(script_sig)} bytes")
print()

# Verify
print("Running VerifyScript...")
try:
    VerifyScript(script_sig, locking_script, tx, 0, flags=SCRIPT_FLAGS)
    print()
    print("✅ ✅ ✅  SCRIPT VALIDATION PASSED  ✅ ✅ ✅")
    print()
    print("Transaction should be accepted by any Bitcoin node.")
except Exception as e:
    print()
    print("⚠️  VerifyScript raised:")
    print(f"   {type(e).__name__}: {e}")
    print()
    traceback.print_exc()
    print()
    print("NOTE: python-bitcoinlib's script interpreter may not implement")
    print("every quirk of Bitcoin Core (especially FindAndDelete semantics).")
    print("An error here is NOT definitive proof the tx is invalid.")
    print()
    print("If you want a definitive answer, submit to Slipstream — it runs")
    print("the real consensus code and will give a clear error if rejected:")
    print()
    print("  curl -X POST https://slipstream.mara.com/api/transactions \\")
    print("    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"tx_hex\":\"'$(cat {TX_HEX_PATH})'\"}}'")
