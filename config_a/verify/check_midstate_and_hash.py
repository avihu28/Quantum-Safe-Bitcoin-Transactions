#!/usr/bin/env python3
"""Compute midstate, first SHA-256, and sighash from gpu_pinning_params.json.
Compare to GPU debug output to localize where SHA-256 starts diverging.

GPU reports these via `qsb_real ... debug <seq> <lt>`:
  DBG: midstate
  DBG: first_sha256
  DBG: sighash_z

Run this and compare line-by-line. The first divergence localizes the bug.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path


def sha256_midstate(data: bytes, num_blocks: int) -> list[int]:
    """Compute SHA-256 internal state (h[0..7]) after processing num_blocks 64-byte chunks.

    Uses pure-Python SHA-256 implementation since hashlib doesn't expose midstate.
    """
    K = [
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    ]
    state = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

    def rotr(x, n): return ((x >> n) | (x << (32 - n))) & 0xffffffff

    for blk_idx in range(num_blocks):
        block = data[blk_idx*64 : (blk_idx+1)*64]
        W = [int.from_bytes(block[i*4:(i+1)*4], 'big') for i in range(16)]
        for i in range(16, 64):
            s0 = rotr(W[i-15], 7) ^ rotr(W[i-15], 18) ^ (W[i-15] >> 3)
            s1 = rotr(W[i-2], 17) ^ rotr(W[i-2], 19) ^ (W[i-2] >> 10)
            W.append((W[i-16] + s0 + W[i-7] + s1) & 0xffffffff)
        a, b, c, d, e, f, g, h = state
        for i in range(64):
            S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
            ch = (e & f) ^ ((~e) & g)
            t1 = (h + S1 + ch + K[i] + W[i]) & 0xffffffff
            S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            t2 = (S0 + maj) & 0xffffffff
            h = g; g = f; f = e; e = (d + t1) & 0xffffffff
            d = c; c = b; b = a; a = (t1 + t2) & 0xffffffff
        state = [(state[i] + v) & 0xffffffff for i, v in enumerate([a,b,c,d,e,f,g,h])]
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", type=lambda x: int(x, 0))
    ap.add_argument("lt", type=int)
    ap.add_argument("--params", default="gpu_pinning_params.json")
    ap.add_argument("--pinning-bin", default="pinning.bin",
                    help="optional: also extract midstate from pinning.bin and compare")
    args = ap.parse_args()

    with open(args.params) as f:
        p = json.load(f)
    pin_prefix = bytes.fromhex(p.get("pin_prefix") or p["tx_prefix"])
    combined_suffix_hex = p.get("combined_suffix")
    if combined_suffix_hex is None:
        print("✗ params file lacks combined_suffix (old format). Cannot proceed.")
        return 1
    combined_suffix = bytearray(bytes.fromhex(combined_suffix_hex))
    seq_offset = p["seq_offset"]
    lt_offset = p["lt_offset"]
    midstate_blocks = p["midstate_blocks"]
    total_preimage_len = p["total_preimage_len"]

    # Patch seq + lt
    combined_suffix[seq_offset:seq_offset+4] = struct.pack("<I", args.seq)
    combined_suffix[lt_offset:lt_offset+4] = struct.pack("<I", args.lt)

    # Compute midstate: SHA-256 state after hashing pin_prefix[0:midstate_blocks*64]
    pp_full_blocks = pin_prefix[:midstate_blocks*64]
    if len(pp_full_blocks) != midstate_blocks*64:
        print(f"✗ pin_prefix doesn't have {midstate_blocks} full blocks worth of data!")
        print(f"  pin_prefix is {len(pin_prefix)} bytes, expected at least {midstate_blocks*64}")
        return 1
    midstate = sha256_midstate(pp_full_blocks, midstate_blocks)

    print("=" * 60)
    print(f"For seq=0x{args.seq:08x} lt={args.lt}")
    print("=" * 60)

    # Print midstate as concatenated 32-bit BE hex (matches GPU output format)
    midstate_hex = "".join(f"{w:08x}" for w in midstate)
    print(f"\nDBG: midstate (computed by CPU from pin_prefix) =")
    print(f"  {midstate_hex}")
    print(f"\n  → Compare to GPU debug output: 'DBG: midstate = ...'")
    print(f"  → If these differ, pinning.bin has WRONG midstate (export bug).")

    # Check pinning.bin if available
    if Path(args.pinning_bin).exists():
        with open(args.pinning_bin, "rb") as f:
            pb = f.read()
        # Layout: midstate(8*4 BE) + suffix_len(4 LE) + ...
        if len(pb) < 32:
            print(f"\n✗ pinning.bin is too short ({len(pb)} bytes)")
        else:
            stored_midstate = [struct.unpack(">I", pb[i*4:(i+1)*4])[0] for i in range(8)]
            stored_hex = "".join(f"{w:08x}" for w in stored_midstate)
            print(f"\nDBG: midstate (stored in pinning.bin) =")
            print(f"  {stored_hex}")
            if stored_hex == midstate_hex:
                print(f"  ✓ pinning.bin midstate matches what CPU recomputes from pin_prefix")
            else:
                print(f"  ✗ MISMATCH! pinning.bin has wrong midstate.")
                print(f"     This is an export bug: gpu_pinning_params.json and pinning.bin")
                print(f"     are inconsistent. Solution: re-run `qsb_pipeline.py export`.")

    # Compute full first SHA-256 (over the entire preimage)
    full_preimage = pin_prefix[:midstate_blocks*64] + bytes(combined_suffix)
    if len(full_preimage) != total_preimage_len:
        print(f"\n⚠ length mismatch: built {len(full_preimage)}, "
              f"expected {total_preimage_len}")
    first_sha = hashlib.sha256(full_preimage).digest()
    print(f"\nDBG: first_sha256 =")
    print(f"  {first_sha.hex()}")
    print(f"\n  → Compare to GPU debug output: 'DBG: first_sha256 = ...'")

    # Sighash z = SHA-256(SHA-256(preimage))
    z = hashlib.sha256(first_sha).digest()
    print(f"\nDBG: sighash_z =")
    print(f"  {z.hex()}")
    print(f"\n  → Compare to GPU debug output: 'DBG: sighash_z = ...'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
