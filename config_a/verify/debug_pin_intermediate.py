#!/usr/bin/env python3
"""debug_pin_intermediate.py — CPU-side counterpart of the GPU debug kernel.

Runs the full pin pipeline for ONE specific (seq, lt) and prints every
intermediate value in the same "DBG: name = hex" format as the GPU debug
kernel. Used with `diff` to localize where GPU and CPU computations
diverge.

Usage:
    python3 debug_pin_intermediate.py <seq_hex> <locktime> --funding-txid <hex>

Read state from cwd (qsb_state.json + gpu_pinning_params.json) by default.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from secp256k1 import N, P, B, G, modinv, point_add, point_mul
from bitcoin_tx import find_and_delete, Transaction, TxIn


def h2b(s): return bytes.fromhex(s)


def dump_u64x4_be(name: str, val: int) -> None:
    """Print 256-bit value as 64 hex chars (BE)."""
    print(f"DBG: {name} = {val:064x}")


def dump_bytes_be(name: str, b: bytes) -> None:
    print(f"DBG: {name} = {b.hex()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", type=lambda x: int(x, 0),
                    help="sequence value, hex (0x...) or decimal")
    ap.add_argument("lt", type=int, help="locktime")
    ap.add_argument("--work-dir", default=".", help="dir with state + params")
    ap.add_argument("--funding-txid", required=True, help="funding tx id (hex)")
    ap.add_argument("--funding-vout", type=int, default=0)
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args()

    work = Path(args.work_dir).resolve()
    with open(work / "qsb_state.json") as f:
        state = json.load(f)
    with open(work / "gpu_pinning_params.json") as f:
        params = json.load(f)

    print(f"DBG: seq = {args.seq:08x}")
    print(f"DBG: lt = {args.lt}")

    full_script = h2b(state["full_script_hex"])
    pin_sig = h2b(state["pin_sig"])
    pin_r = state["pin_r"]
    pin_s = state["pin_s"]
    hash_mode = state.get("hash_mode", "sha256")

    # Build the tx with the claimed (seq, lt) and compute sighash
    pin_script_code = find_and_delete(full_script, pin_sig)
    funding_txid = h2b(args.funding_txid)[::-1]
    tx = Transaction(version=args.version, locktime=args.lt)
    tx.add_input(TxIn(funding_txid, args.funding_vout, b"", args.seq))
    z_int = tx.sighash(0, pin_script_code, sighash_type=0x01)
    z_be = z_int.to_bytes(32, "big")
    dump_bytes_be("sighash_z", z_be)
    dump_u64x4_be("z_scalar", z_int)

    # Recovery: reconstruct what the kernel computes
    # u1 = (-z * r^-1) mod n
    r_inv = modinv(pin_r, N)
    neg_r_inv = (-r_inv) % N
    dump_u64x4_be("neg_r_inv", neg_r_inv)
    u1 = (neg_r_inv * z_int) % N
    dump_u64x4_be("u1", u1)

    # u1 * G
    u1G = point_mul(u1, G)
    dump_u64x4_be("u1G_x_affine", u1G[0])
    dump_u64x4_be("u1G_y_affine", u1G[1])

    # u2 * R for recid=0
    # R has x=pin_r, y=even (recid=0)
    x = pin_r
    y_sq = (pow(x, 3, P) + B) % P
    y_root = pow(y_sq, (P + 1) // 4, P)
    if pow(y_root, 2, P) != y_sq:
        print("DBG: ERROR: pin_r is not a valid x-coord on secp256k1")
        return 1
    y_even = y_root if y_root % 2 == 0 else P - y_root
    R0 = (x, y_even)
    u2 = (pin_s * r_inv) % N
    u2R0 = point_mul(u2, R0)
    dump_u64x4_be("u2R_x", u2R0[0])
    dump_u64x4_be("u2R_y", u2R0[1])

    # neg_2u2R = -2 * u2R
    two_u2R = point_add(u2R0, u2R0)
    neg_2u2R = (two_u2R[0], (P - two_u2R[1]) % P)
    dump_u64x4_be("neg_2u2R_x", neg_2u2R[0])
    dump_u64x4_be("neg_2u2R_y", neg_2u2R[1])

    # Q1 (recid=0) = u1G + u2R   (affine)
    Q1 = point_add(u1G, u2R0)
    dump_u64x4_be("Q1_aff_x", Q1[0])
    dump_u64x4_be("Q1_aff_y", Q1[1])

    # Q2 (recid=1) = u1G + u2R' where R' has -y, equivalently u1G - u2R
    Q2 = point_add(Q1, neg_2u2R)
    dump_u64x4_be("Q2_aff_x", Q2[0])
    dump_u64x4_be("Q2_aff_y", Q2[1])

    # Compressed pubkeys + SHA256 + DER checks
    for ri, Q in [(0, Q1), (1, Q2)]:
        prefix = 0x02 if Q[1] % 2 == 0 else 0x03
        pk = bytes([prefix]) + Q[0].to_bytes(32, "big")
        print(f"DBG: recid{ri}_pubkey = {pk.hex()}")
        h = hashlib.sha256(pk).digest()
        print(f"DBG: recid{ri}_sha_pk = {h.hex()}")
        valid_der = is_valid_der(h)
        r_on_curve = der_r_on_curve(h)
        print(f"DBG: recid{ri}_valid_der = {1 if valid_der else 0}")
        print(f"DBG: recid{ri}_r_on_curve = {1 if r_on_curve else 0}")
        print(f"DBG: recid{ri}_valid_strict = {1 if valid_der and r_on_curve else 0}")
        print(f"DBG: recid{ri}_valid_easy = {1 if (len(h) >= 9 and (h[0] >> 4) == 3) else 0}")

    print("DBG: END")
    return 0


def is_valid_der(d: bytes) -> bool:
    """Mirror of gpu_is_valid_der."""
    l = len(d)
    if l < 9 or d[0] != 0x30:
        return False
    tl = d[1]
    if tl + 3 != l:
        return False
    idx = 2
    for _ in range(2):
        if idx >= l - 1 or d[idx] != 0x02:
            return False
        idx += 1
        il = d[idx]
        idx += 1
        if il == 0 or idx + il > l - 1:
            return False
        if il > 1 and d[idx] == 0 and not (d[idx + 1] & 0x80):
            return False
        if d[idx] & 0x80:
            return False
        idx += il
    return idx == l - 1


def der_r_on_curve(d: bytes) -> bool:
    """Mirror of gpu_der_r_on_curve. Extract r from DER, check on curve."""
    if len(d) < 4 or d[0] != 0x30 or d[2] != 0x02:
        return False
    rl = d[3]
    if rl == 0 or 4 + rl > len(d):
        return False
    r_bytes = d[4:4 + rl]
    # Strip leading zero
    if len(r_bytes) > 1 and r_bytes[0] == 0:
        r_bytes = r_bytes[1:]
    # Pad to 32 bytes for x-coord
    if len(r_bytes) > 32:
        return False
    x_bytes = b"\x00" * (32 - len(r_bytes)) + r_bytes
    x = int.from_bytes(x_bytes, "big")
    # Check y² = x³ + 7 has a root mod p
    y_sq = (pow(x, 3, P) + B) % P
    y = pow(y_sq, (P + 1) // 4, P)
    return pow(y, 2, P) == y_sq


if __name__ == "__main__":
    sys.exit(main())
