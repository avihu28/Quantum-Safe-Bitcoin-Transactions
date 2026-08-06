#!/usr/bin/env python3
"""debug_digest_intermediate.py — CPU-side counterpart of the GPU digest
debug kernel. Dumps every intermediate value in the same DBG: format so
fleet's debug-digest command can diff them.

Usage:
    python3 debug_digest_intermediate.py <round> <subset_csv>

Example:
    python3 debug_digest_intermediate.py 1 0,1,2,3,4,5,6,7,8

Reads gpu_digest_r<round>_params.json and qsb_state.json from cwd.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from secp256k1 import N, P, B, G, modinv, point_add, point_mul


def h2b(s): return bytes.fromhex(s)


def dump_u64x4_be(name: str, val: int) -> None:
    print(f"DBG: {name} = {val:064x}")


def dump_bytes(name: str, b: bytes) -> None:
    print(f"DBG: {name} = {b.hex()}")


def is_valid_der(d: bytes) -> bool:
    l = len(d)
    if l < 9 or d[0] != 0x30: return False
    tl = d[1]
    if tl + 3 != l: return False
    idx = 2
    for _ in range(2):
        if idx >= l - 1 or d[idx] != 0x02: return False
        idx += 1
        il = d[idx]; idx += 1
        if il == 0 or idx + il > l - 1: return False
        if il > 1 and d[idx] == 0 and not (d[idx + 1] & 0x80): return False
        if d[idx] & 0x80: return False
        idx += il
    return idx == l - 1


def der_r_on_curve(d: bytes) -> bool:
    if len(d) < 4 or d[0] != 0x30 or d[2] != 0x02: return False
    rl = d[3]
    if rl == 0 or 4 + rl > len(d): return False
    r_bytes = d[4:4 + rl]
    if len(r_bytes) > 1 and r_bytes[0] == 0: r_bytes = r_bytes[1:]
    if len(r_bytes) > 32: return False
    x_bytes = b"\x00" * (32 - len(r_bytes)) + r_bytes
    x = int.from_bytes(x_bytes, "big")
    y_sq = (pow(x, 3, P) + B) % P
    y = pow(y_sq, (P + 1) // 4, P)
    return pow(y, 2, P) == y_sq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("round", type=int)
    ap.add_argument("subset", help="comma-separated indices")
    ap.add_argument("--work-dir", default=".")
    args = ap.parse_args()

    work = Path(args.work_dir).resolve()
    ri = args.round
    skip = sorted(int(x) for x in args.subset.split(","))

    params_file = work / f"gpu_digest_r{ri}_params.json"
    if not params_file.exists():
        print(f"ERROR: {params_file} not found")
        return 1
    with open(params_file) as f:
        p = json.load(f)
    with open(work / "qsb_state.json") as f:
        state = json.load(f)

    n_pool = p["n"]
    t_sel = p["t"]
    if len(skip) != t_sel:
        print(f"ERROR: expected {t_sel} indices, got {len(skip)}")
        return 1

    print(f"DBG: n_pool = {n_pool}")
    print(f"DBG: t_sel = {t_sel}")
    print(f"DBG: total_preimage_len = {p['total_preimage_len']}")
    print(f"DBG: prefix_remainder_len = {p.get('prefix_remainder_len', 0)}")
    tail = h2b(p["tail_section"])
    tx_suffix = h2b(p["tx_suffix"])
    print(f"DBG: tail_len = {len(tail)}")
    print(f"DBG: tx_suffix_len = {len(tx_suffix)}")
    print(f"DBG: skip_indices = {','.join(str(x) for x in skip)}")

    # Reconstruct prefix_remainder + filtered dummies + tail + tx_suffix EXACTLY
    # like the GPU kernel does (iterating i = 0..n_pool-1, skipping when i ∈ skip).
    # The "dummy_sigs" in the binary file are written in REVERSED pool order, so
    # d_dummy_sigs[i*10..] corresponds to push of pool index (n-1-i). We mirror
    # that exactly.

    prefix_remainder = h2b(p.get("prefix_remainder", ""))

    # dummy_pushes_script_order in JSON — same as binary file order.
    dummy_pushes = [h2b(p_) for p_ in p["dummy_sig_pushes"]]
    if len(dummy_pushes) != n_pool:
        print(f"ERROR: dummy_sig_pushes has {len(dummy_pushes)} entries, expected {n_pool}")
        return 1
    skip_set = set(skip)

    suffix = bytearray(prefix_remainder)
    for i in range(n_pool):
        if i in skip_set:
            continue
        suffix += dummy_pushes[i]
    suffix += tail
    suffix += tx_suffix

    print(f"DBG: suffix_pos = {len(suffix)}")
    print(f"DBG: suffix_first32 = {bytes(suffix[:32]).hex()}")
    print(f"DBG: suffix_last32 = {bytes(suffix[-32:]).hex()}")

    # Compute SHA-256 starting from midstate.
    # Need the midstate. In emulate_digest_round it computes from scratch by
    # building tx_prefix + scriptcode + tx_suffix and SHA-256ing the whole
    # thing. We do the same here using all the params.

    # Reconstruct fixed_prefix (= what the midstate covers).
    # fixed_prefix = d_tx_prefix + pre_hors + hors_section
    tx_prefix = h2b(p["tx_prefix"])
    pre_hors = h2b(p.get("pre_hors_section", ""))
    hors = h2b(p["hors_section"])
    fixed_prefix = tx_prefix + pre_hors + hors

    midstate_blocks = p["midstate_blocks"]
    midstate_bytes_count = midstate_blocks * 64

    # Midstate covers exactly midstate_bytes_count bytes of fixed_prefix.
    # The rest of fixed_prefix is prefix_remainder.
    expected_remainder = fixed_prefix[midstate_bytes_count:]
    if expected_remainder != prefix_remainder:
        print(f"DBG: WARNING: prefix_remainder mismatch")
        print(f"  expected: {expected_remainder.hex()}")
        print(f"  got     : {prefix_remainder.hex()}")

    # Compute midstate (from scratch, hashing fixed_prefix[:midstate_bytes_count])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_midstate_and_hash import sha256_midstate
    midstate = sha256_midstate(fixed_prefix[:midstate_bytes_count], midstate_blocks)
    print(f"DBG: midstate = {''.join(f'{w:08x}' for w in midstate)}")

    # Full preimage = midstate_bytes + suffix
    full_preimage = fixed_prefix[:midstate_bytes_count] + bytes(suffix)
    if len(full_preimage) != p["total_preimage_len"]:
        print(f"DBG: WARNING: preimage length mismatch: built={len(full_preimage)}, "
              f"expected={p['total_preimage_len']}")

    first_sha = hashlib.sha256(full_preimage).digest()
    print(f"DBG: first_sha256 = {first_sha.hex()}")

    z = hashlib.sha256(first_sha).digest()
    print(f"DBG: sighash_z = {z.hex()}")
    z_int = int.from_bytes(z, "big")
    dump_u64x4_be("z_scalar", z_int)

    sig_r = p["sig_r"]
    sig_s = p["sig_s"]
    r_inv = modinv(sig_r, N)
    neg_r_inv = (-r_inv) % N
    dump_u64x4_be("neg_r_inv", neg_r_inv)
    u1 = (neg_r_inv * z_int) % N
    dump_u64x4_be("u1", u1)

    u1G = point_mul(u1, G)
    dump_u64x4_be("u1G_x_affine", u1G[0])
    dump_u64x4_be("u1G_y_affine", u1G[1])

    # u2R for recid=0
    x = sig_r
    y_sq = (pow(x, 3, P) + B) % P
    y_root = pow(y_sq, (P + 1) // 4, P)
    if pow(y_root, 2, P) != y_sq:
        print("DBG: ERROR: sig_r not a valid x-coord")
        return 1
    y_even = y_root if y_root % 2 == 0 else P - y_root
    R0 = (x, y_even)
    u2 = (sig_s * r_inv) % N
    u2R0 = point_mul(u2, R0)
    dump_u64x4_be("u2R_x", u2R0[0])
    dump_u64x4_be("u2R_y", u2R0[1])

    two_u2R = point_add(u2R0, u2R0)
    neg_2u2R = (two_u2R[0], (P - two_u2R[1]) % P)
    dump_u64x4_be("neg_2u2R_x", neg_2u2R[0])
    dump_u64x4_be("neg_2u2R_y", neg_2u2R[1])

    Q1 = point_add(u1G, u2R0)
    dump_u64x4_be("Q1_aff_x", Q1[0])
    dump_u64x4_be("Q1_aff_y", Q1[1])
    Q2 = point_add(Q1, neg_2u2R)
    dump_u64x4_be("Q2_aff_x", Q2[0])
    dump_u64x4_be("Q2_aff_y", Q2[1])

    for ri_, Q in [(0, Q1), (1, Q2)]:
        prefix = 0x02 if Q[1] % 2 == 0 else 0x03
        pk = bytes([prefix]) + Q[0].to_bytes(32, "big")
        print(f"DBG: recid{ri_}_pubkey = {pk.hex()}")
        h = hashlib.sha256(pk).digest()
        print(f"DBG: recid{ri_}_sha_pk = {h.hex()}")
        vd = is_valid_der(h)
        rc = der_r_on_curve(h)
        ve = (len(h) >= 9 and (h[0] >> 4) == 3)
        print(f"DBG: recid{ri_}_valid_der = {1 if vd else 0}")
        print(f"DBG: recid{ri_}_r_on_curve = {1 if rc else 0}")
        print(f"DBG: recid{ri_}_valid_strict = {1 if vd and rc else 0}")
        print(f"DBG: recid{ri_}_valid_easy = {1 if ve else 0}")

    print("DBG: END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
