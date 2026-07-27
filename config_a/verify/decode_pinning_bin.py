#!/usr/bin/env python3
"""Decode pinning.bin and gpu_pinning_params.json, dump all fields, and check
that the SHA-256 layout invariants hold:

  midstate_blocks * 64 + suffix_len == total_preimage_len
  pin_prefix_len + (suffix_len - prefix_remainder_len) == total_preimage_len
"""
import argparse
import json
import struct
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="gpu_pinning_params.json")
    ap.add_argument("--pinning-bin", default="pinning.bin")
    args = ap.parse_args()

    print("=" * 60)
    print("Decoding gpu_pinning_params.json")
    print("=" * 60)
    with open(args.params) as f:
        p = json.load(f)
    for k in ['type', 'pin_prefix_len', 'tx_prefix_len', 'combined_suffix_len',
              'total_preimage_len', 'midstate_blocks', 'prefix_remainder_len',
              'seq_offset', 'lt_offset']:
        if k in p:
            print(f"  {k:<22} = {p[k]}")

    print()
    print("=" * 60)
    print("Decoding pinning.bin (the kernel reads this)")
    print("=" * 60)
    if not Path(args.pinning_bin).exists():
        print(f"  ✗ {args.pinning_bin} not found")
        return 1
    with open(args.pinning_bin, "rb") as f:
        pb = f.read()

    print(f"  file size = {len(pb)} bytes")
    if len(pb) < 32:
        return 1
    midstate = [struct.unpack(">I", pb[i*4:(i+1)*4])[0] for i in range(8)]
    midstate_hex = "".join(f"{w:08x}" for w in midstate)
    print(f"  midstate (BE)        = {midstate_hex}")
    off = 32
    suffix_len = struct.unpack("<I", pb[off:off+4])[0]
    off += 4
    print(f"  suffix_len           = {suffix_len}")
    suffix = pb[off:off+suffix_len]
    off += suffix_len
    print(f"  suffix (first 16)    = {suffix[:16].hex()}")
    print(f"  suffix (last  16)    = {suffix[-16:].hex()}")
    total = struct.unpack("<I", pb[off:off+4])[0]
    off += 4
    print(f"  total_preimage_len   = {total}")
    seq_off = struct.unpack("<I", pb[off:off+4])[0]
    off += 4
    print(f"  seq_offset           = {seq_off}")
    lt_off = struct.unpack("<I", pb[off:off+4])[0]
    off += 4
    print(f"  lt_offset            = {lt_off}")

    print()
    print("=" * 60)
    print("Layout invariant checks")
    print("=" * 60)

    pin_prefix_len = p.get('pin_prefix_len', p.get('tx_prefix_len'))
    combined_suffix_len = p.get('combined_suffix_len', suffix_len)
    midstate_blocks = p.get('midstate_blocks')
    prefix_remainder_len = p.get('prefix_remainder_len')

    # Invariant 1: pinning.bin's suffix_len should equal params' combined_suffix_len
    print(f"  pinning.bin suffix_len    : {suffix_len}")
    print(f"  params combined_suffix_len: {combined_suffix_len}")
    if suffix_len == combined_suffix_len:
        print(f"  ✓ pinning.bin and params agree on suffix length")
    else:
        print(f"  ✗ MISMATCH between pinning.bin and params!")

    # Invariant 2: midstate covers (midstate_blocks * 64) bytes; rest is suffix
    if midstate_blocks is not None:
        midstate_bytes = midstate_blocks * 64
        print(f"\n  midstate covers       : {midstate_bytes} bytes")
        print(f"  suffix bytes          : {suffix_len}")
        print(f"  sum                   : {midstate_bytes + suffix_len}")
        print(f"  total_preimage_len    : {total}")
        if midstate_bytes + suffix_len == total:
            print(f"  ✓ midstate + suffix = total ✓")
        else:
            print(f"  ✗ MISMATCH! midstate + suffix ≠ total_preimage_len")
            print(f"    Difference: {midstate_bytes + suffix_len - total}")
            print(f"    Likely cause: export computed total wrong, OR midstate covers")
            print(f"    a different number of bytes than midstate_blocks*64.")

    # Invariant 3: pin_prefix_len = midstate_blocks*64 + prefix_remainder_len
    if pin_prefix_len is not None and midstate_blocks is not None:
        expected_pin_prefix_len = midstate_blocks * 64 + (prefix_remainder_len or 0)
        print(f"\n  pin_prefix_len from JSON  : {pin_prefix_len}")
        print(f"  midstate_blocks*64 + remainder_len: {expected_pin_prefix_len}")
        if pin_prefix_len == expected_pin_prefix_len:
            print(f"  ✓ pin_prefix layout consistent")
        else:
            print(f"  ✗ MISMATCH!")

    # Invariant 4: combined_suffix should equal prefix_remainder + 13
    if prefix_remainder_len is not None:
        print(f"\n  prefix_remainder_len    : {prefix_remainder_len}")
        print(f"  combined_suffix_len     : {combined_suffix_len}")
        print(f"  expected (remainder+13) : {prefix_remainder_len + 13}")
        if combined_suffix_len == prefix_remainder_len + 13:
            print(f"  ✓ combined_suffix length matches remainder + (seq+outcount+lt+sighash)")
        else:
            print(f"  ✗ MISMATCH! combined_suffix has {combined_suffix_len - prefix_remainder_len - 13} extra bytes")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
