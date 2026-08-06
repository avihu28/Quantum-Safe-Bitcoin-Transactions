#!/usr/bin/env python3
"""Compute the funding tx's txid from the UNSIGNED tx hex.

For SegWit-only inputs (P2WPKH/P2WSH), witness data is excluded from the
txid computation. So the unsigned tx (empty scriptSig, no witness) has the
same txid as the signed tx. This lets us know the funding txid BEFORE
signing/broadcasting, which we need to pre-compute the GPU search inputs.

Usage:
    python3 compute_funding_txid.py qsb_funding_unsigned.hex
"""
import hashlib
import struct
import sys
from pathlib import Path


def compute_txid(raw_tx_bytes):
    """txid = SHA256d(serialized tx without witness), reversed for display."""
    h = hashlib.sha256(hashlib.sha256(raw_tx_bytes).digest()).digest()
    return h[::-1].hex()


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <funding_unsigned.hex>", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        sys.exit(1)
    hex_str = path.read_text().strip()
    raw = bytes.fromhex(hex_str)

    # Sanity check that the tx has empty scriptSigs (else it's signed
    # without segwit-only inputs — txid computation would differ).
    # Skip the check; just inform the user.
    txid = compute_txid(raw)
    print(f"  Unsigned tx file: {path}")
    print(f"  Tx size: {len(raw)} bytes")
    print(f"  Computed funding txid: {txid}")
    print()
    print(f"  ⚠ This is correct ONLY if all inputs are SegWit (P2WPKH/P2WSH).")
    print(f"  For legacy P2PKH inputs, the txid changes when scriptSig is filled,")
    print(f"  so you'd need to compute txid from the SIGNED tx instead.")


if __name__ == "__main__":
    main()
