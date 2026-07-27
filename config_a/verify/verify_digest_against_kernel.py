#!/usr/bin/env python3
"""Verify a digest hit against the emulator path.

The GPU kernel reports its hit as STORAGE indices (i.e., row offsets into
the dummy push array as packed in digest_rN.bin), which is the REVERSE of
state index ordering used by verify_hit / cmd_assemble.

This tool tries both interpretations and tells you which one matches.

Usage:
    python3 verify_digest_against_kernel.py \\
        --round 1 \\
        --indices 41,54,63,67,86,104,108,117,135 \\
        --sequence 0x8000dfd2 \\
        --locktime 887083300 \\
        --gpu-sighash <kernel-reported sighash hex> \\
        [--gpu-pubhash <kernel-reported pubhash hex>]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from gpu_emulator import emulate_digest_round


def try_emulator(params, indices, sequence, locktime, label):
    out = emulate_digest_round(params, indices, sighash_type=0x01,
                                sequence=sequence, locktime=locktime)
    z_hex = out["sighash"].hex()
    return out, z_hex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--indices", required=True)
    ap.add_argument("--sequence", type=lambda x: int(x, 0), required=True)
    ap.add_argument("--locktime", type=int, required=True)
    ap.add_argument("--gpu-sighash", default=None)
    ap.add_argument("--gpu-pubhash", default=None)
    ap.add_argument("--params", default=None)
    args = ap.parse_args()

    if args.params:
        params_path = Path(args.params)
    else:
        params_path = Path(f"gpu_digest_r{args.round}_params.json")
    if not params_path.exists():
        print(f"ERROR: {params_path} not found")
        return 1
    with open(params_path) as f:
        params = json.load(f)
    n = params["n"]

    raw_indices = sorted(int(x) for x in args.indices.split(","))
    storage_to_state = sorted([n - 1 - i for i in raw_indices])

    print("=" * 60)
    print(f"DIGEST R{args.round} — emulator vs kernel verification")
    print("=" * 60)
    print(f"  n:                    {n}")
    print(f"  Input indices:        {raw_indices}")
    print(f"  Reversed (n-1-i):     {storage_to_state}")
    print(f"  Sequence:             0x{args.sequence:08x}")
    print(f"  Locktime:             {args.locktime}")
    print()

    # Try both interpretations
    print("─── Interpretation A: indices as state/pool indices ───")
    out_a, z_a = try_emulator(params, raw_indices, args.sequence, args.locktime, "as-is")
    print(f"  Emulator sighash:  {z_a}")
    if args.gpu_sighash:
        if z_a.lower() == args.gpu_sighash.lower():
            print(f"  GPU sighash:       {args.gpu_sighash}  ✓ MATCH")
        else:
            print(f"  GPU sighash:       {args.gpu_sighash}  ✗ no match")

    print()
    print("─── Interpretation B: indices as STORAGE indices (kernel's native) ───")
    out_b, z_b = try_emulator(params, storage_to_state, args.sequence, args.locktime, "reversed")
    print(f"  Emulator sighash:  {z_b}")
    if args.gpu_sighash:
        if z_b.lower() == args.gpu_sighash.lower():
            print(f"  GPU sighash:       {args.gpu_sighash}  ✓ MATCH")
        else:
            print(f"  GPU sighash:       {args.gpu_sighash}  ✗ no match")

    print()
    if not args.gpu_sighash:
        print("  (no --gpu-sighash supplied; cannot determine which interpretation is correct)")
        return 0

    matched = None
    if z_a.lower() == args.gpu_sighash.lower():
        matched = ("A", out_a, raw_indices)
    elif z_b.lower() == args.gpu_sighash.lower():
        matched = ("B", out_b, storage_to_state)

    if matched is None:
        print(f"  ✗ NEITHER interpretation matches the kernel's sighash.")
        print(f"     This suggests a deeper bug — kernel computed something different")
        print(f"     than either interpretation predicts. Investigate kernel pipeline.")
        return 1

    label, out, state_indices = matched
    print(f"  ✓ Interpretation {label} matched the GPU sighash")
    print(f"  Correct STATE indices for assembly: {state_indices}")
    print()

    hit = out["hit"]
    if hit is None:
        print(f"  ⚠ Sighash matches but emulator found NO valid DER candidate.")
        print(f"     Either GPU misclassified, or emulator has a bug in DER check.")
        print(f"     Candidates considered:")
        for c in out["candidates"]:
            print(f"       recid={c['recid']} valid={c['is_valid_der']}  "
                  f"hash={c['puzzle_hash'].hex()[:32]}...")
        return 1

    print(f"  ✓ Emulator found valid DER hit:")
    print(f"      recid:       {hit['recid']}")
    print(f"      hash_choice: {hit['hash_choice']}")
    print(f"      pubkey:      {hit['pubkey'].hex()}")
    print(f"      puzzle_hash: {hit['puzzle_hash'].hex()}")
    if args.gpu_pubhash:
        if hit["puzzle_hash"].hex().lower() == args.gpu_pubhash.lower():
            print(f"      GPU pubhash: {args.gpu_pubhash}  ✓ MATCH")
        else:
            print(f"      GPU pubhash: {args.gpu_pubhash}  ✗ MISMATCH")

    print()
    print(f"  ✅ HIT VERIFIED")
    print(f"     Use STATE indices for cmd_assemble: {state_indices}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
