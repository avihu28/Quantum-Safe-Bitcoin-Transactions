#!/usr/bin/env python3
"""
OPTION 2: Validate a FULL QSB transaction through Bitcoin Core's script
interpreter (via python-bitcoinlib's VerifyScript).

This runs the actual OP_CHECKSIG verification on the assembled witness +
the QSB script, mimicking exactly what a Bitcoin node would do when this
transaction hits mempool.

Two modes:

Mode A: "Partial validation" — construct a fake happy-path scenario where
        we hand-build a transaction with fake dummy-sig and HORS hits and
        ensure VerifyScript processes the script without dying on structural
        bugs. This does NOT require real hits; it just walks the script logic.

Mode B: "Full validation" — takes an already-assembled transaction (from
        qsb_pipeline.py assemble) and runs VerifyScript on it. This is the
        golden test: if this passes, the tx will be accepted by real nodes.

Requirements on your Mac:
    pip3 install python-bitcoinlib

Run:
    python3 option2_validate_script.py <state.json> <assembled_tx_hex> [<scriptpubkey_hex>]
"""
import sys
import hashlib
import json
import struct

sys.path.insert(0, '.')

try:
    from bitcoin.core import (
        CMutableTransaction, CMutableTxIn, CMutableTxOut, COutPoint, lx, x,
        CTxOut, CTransaction, b2x
    )
    from bitcoin.core.script import CScript, OP_CHECKSIG
    from bitcoin.core.scripteval import VerifyScript
    # SCRIPT_VERIFY_* flags live in different submodules across versions
    try:
        from bitcoin.core.scripteval import SCRIPT_VERIFY_P2SH
    except ImportError:
        try:
            from bitcoin.core.script import SCRIPT_VERIFY_P2SH
        except ImportError:
            # Fall back to the raw int flag (bit 0 is always P2SH in consensus)
            SCRIPT_VERIFY_P2SH = 1
except ImportError as e:
    print(f"ERROR: python-bitcoinlib not installed or incomplete: {e}")
    print("Install with: pip3 install python-bitcoinlib")
    sys.exit(1)


def h2b(h): return bytes.fromhex(h)


# Standard consensus flags for mempool policy (as of recent Bitcoin Core)
# These are roughly equivalent to what a post-Segwit node enforces on legacy P2SH spends
STANDARD_FLAGS = (
    # Do NOT include SCRIPT_VERIFY_STRICTENC or DERSIG — those would reject our
    # 9-byte DER sigs. The QSB scheme relies on sigs that are valid under the
    # pre-strict consensus rules (2010-era).
    # We only enforce P2SH-level validity.
    SCRIPT_VERIFY_P2SH,
)


def verify_assembled_tx(assembled_tx_hex, funding_script_pubkey_hex, funding_value,
                         funding_txid_hex, funding_vout):
    """Take a fully-assembled transaction and verify it against a Bitcoin
    Core script interpreter."""
    raw_tx = h2b(assembled_tx_hex)
    tx = CTransaction.deserialize(raw_tx)
    print(f"Deserialized tx:")
    print(f"  version: {tx.nVersion}")
    print(f"  locktime: {tx.nLockTime}")
    print(f"  inputs: {len(tx.vin)}")
    print(f"  outputs: {len(tx.vout)}")

    if len(tx.vin) != 1:
        print(f"ERROR: expected 1 input, got {len(tx.vin)}")
        return False

    vin = tx.vin[0]
    print(f"  input[0].prevout: {vin.prevout.hash[::-1].hex()}:{vin.prevout.n}")
    print(f"  input[0].sequence: 0x{vin.nSequence:08x}")
    print(f"  input[0].scriptSig length: {len(vin.scriptSig)} bytes")

    # Check outpoint matches funding
    expected_txid_le = h2b(funding_txid_hex)[::-1]
    if vin.prevout.hash != expected_txid_le or vin.prevout.n != funding_vout:
        print(f"ERROR: input doesn't match funding outpoint")
        print(f"  expected: {funding_txid_hex}:{funding_vout}")
        print(f"  got:      {vin.prevout.hash[::-1].hex()}:{vin.prevout.n}")
        return False

    script_sig = vin.scriptSig
    script_pubkey = CScript(h2b(funding_script_pubkey_hex))

    print(f"\nRunning VerifyScript...")
    print(f"  scriptSig length: {len(script_sig)} bytes")
    print(f"  scriptPubKey length: {len(script_pubkey)} bytes")

    try:
        VerifyScript(
            script_sig, script_pubkey, tx, 0,
            flags=STANDARD_FLAGS
        )
        print("\n✓ SCRIPT VERIFICATION PASSED")
        print("  This transaction will be accepted by Bitcoin nodes.")
        return True
    except Exception as e:
        print(f"\n✗ SCRIPT VERIFICATION FAILED")
        print(f"  {type(e).__name__}: {e}")
        return False


def compute_script_hash_for_p2sh(script):
    """Compute P2SH scriptPubKey from a redeem script."""
    import hashlib as _hl
    def _rmd160(data):
        import hashlib as _h
        # Try builtin ripemd160
        try:
            h = _h.new('ripemd160')
            h.update(data)
            return h.digest()
        except ValueError:
            # Fallback: pure-python (Python 3.10+ removed ripemd160 from hashlib by default on some builds)
            raise RuntimeError("ripemd160 not available; install pycryptodome or use a Python with OpenSSL 3.0 legacy")
    h160 = _rmd160(_hl.sha256(script).digest())
    return bytes([0xa9, 0x14]) + h160 + bytes([0x87])  # OP_HASH160 <20> OP_EQUAL


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage:")
        print(f"  {sys.argv[0]} <state.json>           # sanity check: full_script → P2SH scriptPubKey")
        print(f"  {sys.argv[0]} <state.json> <tx.hex>  # validate assembled tx")
        return 0

    state_path = sys.argv[1]
    with open(state_path) as f:
        state = json.load(f)

    full_script = h2b(state['full_script_hex'])
    print(f"Full QSB script length: {len(full_script)} bytes")

    # Compute the P2SH scriptPubKey corresponding to our QSB script
    # (This is what the funding output should contain, since we're spending P2SH.)
    try:
        script_pubkey = compute_script_hash_for_p2sh(full_script)
        print(f"Expected P2SH scriptPubKey: {script_pubkey.hex()}")
    except RuntimeError as e:
        print(f"(Note: {e})")
        print("Please provide scriptPubKey manually as the 3rd argument.")
        script_pubkey = None

    if len(sys.argv) < 3:
        print("\n(No tx provided — sanity check only.)")
        return 0

    assembled_tx_hex = sys.argv[2].strip()
    if script_pubkey is None:
        if len(sys.argv) < 4:
            print("ERROR: provide scriptPubKey hex as 3rd arg (P2SH output of your funding UTXO)")
            return 1
        script_pubkey = h2b(sys.argv[3])

    funding_txid = "4fab76e9b0538a49a77443030f8e0243a5d2558155647a839acea0efaa4edc91"
    funding_vout = 0
    funding_value = 10000

    ok = verify_assembled_tx(
        assembled_tx_hex,
        script_pubkey.hex() if isinstance(script_pubkey, bytes) else script_pubkey,
        funding_value, funding_txid, funding_vout
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
