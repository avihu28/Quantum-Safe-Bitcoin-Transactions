#!/usr/bin/env python3
"""Compute sighash z three ways and compare:
  1. From gpu_pinning_params.json's pin_prefix (== what the kernel computes)
  2. From verify_hit reconstruction (what verify_hit / launcher gate uses)
  3. From debug_pin_intermediate.py's reconstruction

Prints all three z values. If (1) differs from (2)/(3), the bug is in the
preimage reconstruction logic — and we can find which byte differs.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from secp256k1 import N, P
from bitcoin_tx import Transaction, TxIn, find_and_delete


def h2b(s): return bytes.fromhex(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", type=lambda x: int(x, 0))
    ap.add_argument("lt", type=int)
    ap.add_argument("--params", default="gpu_pinning_params.json")
    ap.add_argument("--state", default="qsb_state.json")
    ap.add_argument("--funding", default="regtest_funding.json")
    ap.add_argument("--funding-txid", default=None)
    ap.add_argument("--funding-vout", type=int, default=None)
    ap.add_argument("--version", type=int, default=None)
    args = ap.parse_args()

    print(f"\nComputing sighash for seq=0x{args.seq:08x} lt={args.lt}")
    print("=" * 60)

    # ── Method 1: from pin_prefix directly (what the kernel does) ──
    with open(args.params) as f:
        p = json.load(f)
    pin_prefix = bytes.fromhex(p.get("pin_prefix") or p["tx_prefix"])
    combined_suffix_hex = p.get("combined_suffix")
    seq_offset = p.get("seq_offset")
    lt_offset = p.get("lt_offset")
    total_preimage_len = p["total_preimage_len"]

    if combined_suffix_hex is None:
        print("⚠ params file has old format (no combined_suffix). "
              "Method 1 unavailable.")
        z1 = None
    else:
        combined_suffix = bytearray(bytes.fromhex(combined_suffix_hex))
        # Patch seq + lt
        combined_suffix[seq_offset:seq_offset+4] = struct.pack("<I", args.seq)
        combined_suffix[lt_offset:lt_offset+4] = struct.pack("<I", args.lt)

        # The midstate covers midstate_blocks*64 bytes of pin_prefix; the rest
        # of pin_prefix is the FIRST bytes of combined_suffix already. So we
        # need to be careful about the layout.
        midstate_blocks = p.get("midstate_blocks")
        if midstate_blocks is not None:
            # Reconstruct full preimage = pin_prefix[0:midstate_blocks*64] +
            #                              combined_suffix
            full_preimage = pin_prefix[:midstate_blocks*64] + bytes(combined_suffix)
        else:
            # No midstate split — combined_suffix appended to pin_prefix
            full_preimage = pin_prefix + bytes(combined_suffix)

        if len(full_preimage) != total_preimage_len:
            print(f"⚠ Length mismatch: built {len(full_preimage)}, "
                  f"expected {total_preimage_len}")
        z1_be = hashlib.sha256(hashlib.sha256(full_preimage).digest()).digest()
        z1 = z1_be.hex()
        print(f"\n[1] z from pin_prefix + combined_suffix (== kernel):")
        print(f"    {z1}")

    # ── Method 2: full reconstruction (what verify_hit does) ──
    with open(args.state) as f:
        state = json.load(f)
    full_script = h2b(state["full_script_hex"])
    pin_sig = h2b(state["pin_sig"])

    user_txid = args.funding_txid
    user_vout = args.funding_vout
    user_version = args.version
    if not user_txid or user_vout is None or user_version is None:
        rf_path = Path(args.funding)
        if rf_path.exists():
            with open(rf_path) as f:
                rf = json.load(f)
            user_txid = user_txid or rf.get("funding_txid") or rf.get("txid")
            if user_vout is None:
                user_vout = rf.get("vout", 0)
            if user_version is None:
                # default to version=1 (verify_hit's default)
                user_version = 1

    pin_script_code = find_and_delete(full_script, pin_sig)
    funding_txid_internal = h2b(user_txid)[::-1]
    tx = Transaction(version=user_version, locktime=args.lt)
    tx.add_input(TxIn(funding_txid_internal, user_vout, b"", args.seq))
    z2_int = tx.sighash(0, pin_script_code, sighash_type=0x01)
    z2 = f"{z2_int:064x}"
    print(f"\n[2] z from verify_hit-style reconstruction:")
    print(f"    {z2}")

    # ── Compare and diagnose ──
    print("\n" + "=" * 60)
    if z1 is not None and z1 == z2:
        print("✓ Methods 1 and 2 agree.")
        print("  If GPU sighash differs from these, the kernel itself is buggy.")
    elif z1 is not None and z1 != z2:
        print("✗ Methods 1 and 2 DISAGREE.")
        print("  The kernel uses (1); verify_hit uses (2). They reconstruct")
        print("  different preimages. Need to find where they diverge.")
        print()
        print("  Likely culprits:")
        print("    - Different version (kernel may have been built with a")
        print("      different --version than verify_hit defaults to)")
        print("    - Different sighash_type")
        print("    - Different scriptCode (full_script_hex changed since export?)")
        print("    - Different pin_sig (pin_sig changed since export?)")
        print()
        diagnose_preimage_diff(p, state, args.seq, args.lt,
                              user_txid, user_vout, user_version)


def diagnose_preimage_diff(params, state, seq, lt, user_txid, user_vout, user_version):
    """Reconstruct what verify_hit would build and diff against pin_prefix bytes."""
    full_script = bytes.fromhex(state["full_script_hex"])
    pin_sig = bytes.fromhex(state["pin_sig"])
    pin_script_code = find_and_delete(full_script, pin_sig)

    # Build verify_hit's preimage manually (without going through Transaction)
    from bitcoin_tx import serialize_varint
    funding_txid_internal = bytes.fromhex(user_txid)[::-1]
    pre = b""
    pre += struct.pack("<I", user_version)
    pre += serialize_varint(1)
    pre += funding_txid_internal
    pre += struct.pack("<I", user_vout)
    pre += serialize_varint(len(pin_script_code))
    pre += pin_script_code
    pre += struct.pack("<I", seq)
    pre += serialize_varint(0)  # 0 outputs
    pre += struct.pack("<I", lt)
    pre += struct.pack("<I", 0x01)  # SIGHASH_ALL

    print(f"  verify_hit preimage length: {len(pre)}")

    # Compare to params' expected preimage
    pin_prefix = bytes.fromhex(params.get("pin_prefix") or params["tx_prefix"])
    combined_suffix_hex = params.get("combined_suffix")
    if combined_suffix_hex is None:
        print(f"  (cannot compare bytes — old format)")
        return
    combined_suffix = bytearray(bytes.fromhex(combined_suffix_hex))
    seq_offset = params["seq_offset"]
    lt_offset = params["lt_offset"]
    combined_suffix[seq_offset:seq_offset+4] = struct.pack("<I", seq)
    combined_suffix[lt_offset:lt_offset+4] = struct.pack("<I", lt)

    midstate_blocks = params.get("midstate_blocks", 0)
    kernel_preimage = pin_prefix[:midstate_blocks*64] + bytes(combined_suffix)
    print(f"  kernel preimage length:     {len(kernel_preimage)}")

    if len(pre) != len(kernel_preimage):
        print(f"  ✗ Lengths differ! ({len(pre)} vs {len(kernel_preimage)})")
        print(f"    A length difference means a varint/field is different.")

    # Find first byte that differs
    n = min(len(pre), len(kernel_preimage))
    first_diff = -1
    for i in range(n):
        if pre[i] != kernel_preimage[i]:
            first_diff = i
            break

    if first_diff < 0:
        if len(pre) == len(kernel_preimage):
            print("  ✓ Preimages are byte-identical (within compared range).")
            print("    But sighashes still differ?? Check SHA256 implementation.")
        else:
            print(f"  Preimages match for first {n} bytes; the rest differs in length.")
    else:
        print(f"\n  ✗ First byte differs at offset {first_diff}:")
        print(f"    verify_hit byte: 0x{pre[first_diff]:02x}")
        print(f"    kernel byte    : 0x{kernel_preimage[first_diff]:02x}")
        # Show context
        ctx_start = max(0, first_diff - 8)
        ctx_end = min(n, first_diff + 16)
        print(f"\n  Context (offsets {ctx_start}..{ctx_end-1}):")
        print(f"    verify_hit: {pre[ctx_start:ctx_end].hex()}")
        print(f"    kernel    : {kernel_preimage[ctx_start:ctx_end].hex()}")

        # Annotate which field this is
        if first_diff < 4:
            print(f"\n  → Difference is in VERSION (bytes 0-3)")
            print(f"    verify_hit version: {struct.unpack('<I', pre[0:4])[0]}")
            print(f"    kernel version:     {struct.unpack('<I', kernel_preimage[0:4])[0]}")
        elif first_diff == 4:
            print(f"\n  → Difference is in INPUT_COUNT varint (offset 4)")
        elif 5 <= first_diff < 37:
            print(f"\n  → Difference is in FUNDING_TXID (bytes 5-36)")
        elif 37 <= first_diff < 41:
            print(f"\n  → Difference is in FUNDING_VOUT (bytes 37-40)")
        elif first_diff < 41 + 5:
            print(f"\n  → Difference is in scriptCode varint length (offset ~41)")
        else:
            print(f"\n  → Difference is in scriptCode bytes (after varint length)")
            print(f"    This means full_script_hex or pin_sig has changed since export.")


if __name__ == "__main__":
    main()
