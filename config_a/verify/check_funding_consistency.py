#!/usr/bin/env python3
"""Check that the funding info baked into gpu_pinning_params.json matches what
verify_hit would use (from --funding-txid / regtest_funding.json).

A mismatch means the GPU is searching for hits against a DIFFERENT preimage
than the CPU is verifying. The kernel's hits will look "bogus" to verify_hit
even when the kernel math is correct.
"""
import argparse
import json
import struct
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="gpu_pinning_params.json")
    ap.add_argument("--funding-txid", default=None,
                    help="The funding txid you intend to spend (display order).")
    ap.add_argument("--funding-vout", type=int, default=None)
    ap.add_argument("--funding", default="regtest_funding.json",
                    help="If --funding-txid is omitted, read from this JSON.")
    args = ap.parse_args()

    with open(args.params) as f:
        p = json.load(f)

    # Try multiple field names (format has evolved)
    prefix_hex = p.get("pin_prefix") or p.get("tx_prefix")
    if not prefix_hex:
        print("✗ Cannot find pin_prefix or tx_prefix in params file")
        return 1
    prefix = bytes.fromhex(prefix_hex)

    # Parse first 41 bytes
    if len(prefix) < 41:
        print(f"✗ pin_prefix too short ({len(prefix)} bytes); cannot decode")
        return 1
    version = struct.unpack("<I", prefix[0:4])[0]
    in_count = prefix[4]  # varint, assume <0xfd
    prev_hash_internal = prefix[5:37]
    prev_hash_display = prev_hash_internal[::-1].hex()
    prev_index = struct.unpack("<I", prefix[37:41])[0]

    print("=" * 60)
    print("Funding info IN gpu_pinning_params.json (the kernel uses this):")
    print("=" * 60)
    print(f"  version       = {version}")
    print(f"  input_count   = {in_count}")
    print(f"  funding_txid  = {prev_hash_display}")
    print(f"  funding_vout  = {prev_index}")

    # Resolve the user's intended funding info
    user_txid = args.funding_txid
    user_vout = args.funding_vout
    if not user_txid:
        # Try regtest_funding.json
        rf_path = Path(args.funding)
        if rf_path.exists():
            with open(rf_path) as f:
                rf = json.load(f)
            user_txid = rf.get("funding_txid") or rf.get("txid")
            if user_vout is None:
                user_vout = rf.get("vout", 0)
            print(f"\n(loaded user funding info from {rf_path})")

    if not user_txid:
        print("\n⚠ No --funding-txid supplied and no regtest_funding.json found.")
        print("  Cannot compare. Pass --funding-txid <hex> or run setup_regtest.")
        return 0

    if user_vout is None:
        user_vout = 0

    print()
    print("=" * 60)
    print("Funding info you (CPU/verify_hit) are using:")
    print("=" * 60)
    print(f"  funding_txid  = {user_txid}")
    print(f"  funding_vout  = {user_vout}")

    print()
    print("=" * 60)
    txid_match = (user_txid.lower() == prev_hash_display.lower())
    vout_match = (user_vout == prev_index)
    if txid_match and vout_match:
        print("✓ MATCH — kernel and verify_hit are using the same funding info.")
        print("  Any sighash mismatch is from another source (version, etc.)")
    else:
        print("✗ MISMATCH — kernel and verify_hit see DIFFERENT preimages!")
        if not txid_match:
            print(f"  funding_txid: kernel={prev_hash_display}")
            print(f"               you   ={user_txid}")
        if not vout_match:
            print(f"  funding_vout: kernel={prev_index}, you={user_vout}")
        print()
        print("  This is why your pin hits 'don't reproduce on CPU'. The GPU is")
        print("  finding valid hits against the funding info baked into")
        print("  gpu_pinning_params.json, but verify_hit is checking against a")
        print("  DIFFERENT funding transaction. Both math checks are correct;")
        print("  they're just checking different transactions.")
        print()
        print("  Fix: either")
        print("    (a) Re-run pipeline `export` with the CURRENT funding info, or")
        print("    (b) Pass the correct (kernel-baked-in) funding info to")
        print("        verify_hit / debug-pin.")
    return 0 if (txid_match and vout_match) else 1


if __name__ == "__main__":
    sys.exit(main())
