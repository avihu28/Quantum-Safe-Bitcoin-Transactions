"""
GPU kernel emulator — Python reference implementation of the CUDA kernels.

This emulator computes EXACTLY what the GPU computes for a single candidate.
Used to:
  1. Unit-test GPU logic without a GPU.
  2. Verify every claimed GPU hit before tx assembly.
  3. Detect silent model divergence between CPU and GPU code.

For each phase (pinning, digest round), the emulator:
  - Reads the exported GPU input files (pinning.bin, digest_rN.bin, JSON params)
  - Runs the same SHA256(d) → EC recovery → puzzle_hash → DER check pipeline
  - Returns (sighash, pubkey, puzzle_hash, is_valid_der) byte-for-byte.
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pipeline'))

from bitcoin_tx import push_data, find_and_delete  # type: ignore
from secp256k1 import N as CURVE_N, P as CURVE_P, G, point_mul, point_add  # type: ignore
from secp256k1 import modinv, ecdsa_recover  # type: ignore


# --- DER validity check (matches kernel gpu_is_valid_der) ---

def is_valid_der(d):
    """Strict DER check. Returns True if `d` is a valid DER signature with
    a sighash byte at the end. Mirrors gpu_is_valid_der in the CUDA kernel."""
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


def der_r_on_curve(d):
    """The kernel additionally checks that `r` is a valid x-coord on secp256k1."""
    rl = d[3]
    r_start = 4
    if rl > 0 and d[r_start] == 0:
        r_start += 1
        rl -= 1
    if rl > 32 or rl <= 0:
        return False
    r_bytes = d[r_start:r_start + rl]
    r = int.from_bytes(r_bytes.rjust(32, b'\x00'), 'big')
    y_sq = (pow(r, 3, CURVE_P) + 7) % CURVE_P
    return pow(y_sq, (CURVE_P - 1) // 2, CURVE_P) == 1


def compress_pubkey(point):
    x, y = point
    prefix = 0x02 if (y & 1) == 0 else 0x03
    return bytes([prefix]) + x.to_bytes(32, 'big')


# --- puzzle_hash matches the script's OP_SHA256 / OP_RIPEMD160 path ---

def puzzle_hash(pubkey_bytes, hash_mode, hash_choice=None):
    """Compute sig_puzzle given pubkey and hash_mode.

    hash_mode='sha256':        h = SHA256(pk)
    hash_mode='ripemd160':     h = RIPEMD160(pk)    (NOT HASH160!)
    hash_mode='sha256_double': h = SHA256(pk) or SHA256(SHA256(pk)) per hash_choice bit
    """
    if hash_mode == 'sha256':
        h = hashlib.sha256(pubkey_bytes).digest()
        return h, is_valid_der(h), 0
    if hash_mode == 'ripemd160':
        h = hashlib.new('ripemd160', pubkey_bytes).digest()
        return h, is_valid_der(h), 0
    if hash_mode == 'sha256_double':
        h1 = hashlib.sha256(pubkey_bytes).digest()
        if hash_choice == 0:
            return h1, is_valid_der(h1), 0
        if hash_choice == 1:
            h2 = hashlib.sha256(h1).digest()
            return h2, is_valid_der(h2), 1
        if is_valid_der(h1):
            return h1, True, 0
        h2 = hashlib.sha256(h1).digest()
        if is_valid_der(h2):
            return h2, True, 1
        return h1, False, 0
    raise ValueError(f"Unknown hash_mode: {hash_mode}")


# --- DIGEST-round emulator ---

def emulate_digest_round(params, subset_indices, sighash_type=0x01,
                          sequence=None, locktime=None):
    """Emulate one GPU thread processing one subset in a digest round.

    params: dict loaded from gpu_digest_rN_params.json (corrected format)
    subset_indices: sorted t-tuple of pool indices (0..n-1)
    sequence: optional 32-bit sequence to patch into tx_suffix. If None,
        uses whatever was baked into params at export time. The GPU kernel
        ALWAYS patches runtime CLI seq into tx_suffix at offset 0, so
        verify_hit must also pass the actual pin's seq here — otherwise the
        emulator computes a sighash for a different transaction than the
        kernel actually hashed.
    locktime: optional 32-bit locktime, same logic as sequence (offset 5,
        after seq(4) + varint(0)(1)).

    Returns:
      {
        'sighash': 32-byte z
        'scriptcode': reconstructed scriptcode bytes
        'preimage': full sighash preimage
        'candidates': list of recovery/hash trials
        'hit': first valid DER candidate, or None
      }
    """
    n = params['n']
    t = params['t']
    hash_mode = params.get('hash_mode', 'sha256')

    assert len(subset_indices) == t
    assert sorted(subset_indices) == list(subset_indices), \
        "Subset indices must be sorted ascending"

    # Reconstruct scriptcode using the SAME layout the exporter used:
    #   pre_hors_section + hors_section + filtered_dummies + tail_section
    pre_hors = bytes.fromhex(params.get('pre_hors_section', ''))
    hors = bytes.fromhex(params['hors_section'])
    tail = bytes.fromhex(params['tail_section'])
    dummy_pushes_script_order = [bytes.fromhex(p) for p in params['dummy_sig_pushes']]

    selected_set = set(subset_indices)
    scriptcode = bytearray(pre_hors)
    scriptcode += hors
    # Script emits dummy pushes in REVERSED pool order (pool index n-1 first,
    # pool index 0 last). dummy_pushes_script_order[k] = push of pool index (n-1-k).
    for pool_idx in range(n - 1, -1, -1):
        if pool_idx in selected_set:
            continue
        script_order_pos = n - 1 - pool_idx
        scriptcode += dummy_pushes_script_order[script_order_pos]
    scriptcode += tail

    expected_len = params['scriptcode_len']
    assert len(scriptcode) == expected_len, (
        f"scriptcode reconstruction wrong length: {len(scriptcode)} != {expected_len}")

    # Build full preimage: tx_prefix + scriptcode + tx_suffix
    tx_prefix = bytes.fromhex(params['tx_prefix'])
    tx_suffix = bytearray(bytes.fromhex(params['tx_suffix']))
    # tx_suffix layout depends on tx structure baked at export time. For the
    # NEW 2-in/1-out structure: [seq(4)] [varint(1)(1)] [output(value 8 + 
    # script_len varint + script)] [locktime(4)] [sighash_type(4)]. For OLD
    # 0-output structure: [seq(4)] [varint(0)(1)] [locktime(4)] [sighash(4)].
    #
    # The seq is always at offset 0. The locktime offset is found by walking
    # backwards: it's at len(tx_suffix) - 8 (locktime + sighash_type at the end).
    seq_offset = 0
    lt_offset = len(tx_suffix) - 8
    if sequence is not None:
        tx_suffix[seq_offset:seq_offset+4] = struct.pack('<I', sequence)
    if locktime is not None:
        tx_suffix[lt_offset:lt_offset+4] = struct.pack('<I', locktime)
    preimage = tx_prefix + bytes(scriptcode) + bytes(tx_suffix)
    assert len(preimage) == params['total_preimage_len'], (
        f"preimage length mismatch: {len(preimage)} vs {params['total_preimage_len']}")

    # SHA-256d → z
    z_bytes = hashlib.sha256(hashlib.sha256(preimage).digest()).digest()
    z_int = int.from_bytes(z_bytes, 'big')

    # EC recovery with both flags + both r / r+N
    r_val = params['sig_r']
    s_val = params['sig_s']
    candidates = []
    r_tries = [r_val]
    if r_val + CURVE_N < CURVE_P:
        r_tries.append(r_val + CURVE_N)

    for r_try in r_tries:
        for recid in [0, 1]:
            pt = ecdsa_recover(r_try, s_val, z_int, recid)
            if pt is None:
                continue
            pk = compress_pubkey(pt)
            if hash_mode == 'sha256_double':
                for hc in [0, 1]:
                    h, valid, hc_out = puzzle_hash(pk, hash_mode, hash_choice=hc)
                    candidates.append({
                        'recid': recid, 'r_try': r_try, 'hash_choice': hc_out,
                        'pubkey': pk, 'puzzle_hash': h, 'is_valid_der': valid,
                    })
                    if valid:
                        break
            else:
                h, valid, hc_out = puzzle_hash(pk, hash_mode)
                candidates.append({
                    'recid': recid, 'r_try': r_try, 'hash_choice': hc_out,
                    'pubkey': pk, 'puzzle_hash': h, 'is_valid_der': valid,
                })

    hit = next((c for c in candidates if c['is_valid_der']), None)
    return {
        'sighash': z_bytes,
        'scriptcode': bytes(scriptcode),
        'preimage': preimage,
        'candidates': candidates,
        'hit': hit,
    }


# --- PINNING emulator ---

def emulate_pinning(params, locktime, sighash_type=0x01, sequence=None):
    """Emulate one GPU thread trying a specific locktime in the pinning search.

    params: dict loaded from gpu_pinning_params.json
    locktime: 32-bit locktime being tried
    sequence: optional override for the sequence value (for multi-seq searches).
              If None, uses the sequence already baked into the params.
    """
    if 'combined_suffix' in params:
        # New format: pin_prefix + combined_suffix with lt/seq splices
        pin_prefix = bytes.fromhex(params['pin_prefix'])
        suffix = bytearray(bytes.fromhex(params['combined_suffix']))
        seq_off = params['seq_offset']
        lt_off = params['lt_offset']
        if sequence is not None:
            struct.pack_into('<I', suffix, seq_off, sequence)
        struct.pack_into('<I', suffix, lt_off, locktime)
        # sighash_type is already embedded at the end of combined_suffix
        preimage = pin_prefix + bytes(suffix)
    else:
        # Old format: tx_prefix ends right before locktime; we append locktime + sighash
        tx_prefix = bytes.fromhex(params['tx_prefix'])
        preimage = tx_prefix + struct.pack('<I', locktime) + struct.pack('<I', sighash_type)

    assert len(preimage) == params['total_preimage_len'], (
        f"preimage length mismatch: {len(preimage)} vs {params['total_preimage_len']}")

    z_bytes = hashlib.sha256(hashlib.sha256(preimage).digest()).digest()
    z_int = int.from_bytes(z_bytes, 'big')

    pin_r = params['pin_r']
    pin_s = params['pin_s']
    hash_mode = params.get('hash_mode', 'sha256')
    candidates = []
    r_tries = [pin_r]
    if pin_r + CURVE_N < CURVE_P:
        r_tries.append(pin_r + CURVE_N)

    for r_try in r_tries:
        for recid in [0, 1]:
            pt = ecdsa_recover(r_try, pin_s, z_int, recid)
            if pt is None:
                continue
            pk = compress_pubkey(pt)
            if hash_mode == 'sha256_double':
                for hc in [0, 1]:
                    h, valid, hc_out = puzzle_hash(pk, hash_mode, hash_choice=hc)
                    candidates.append({
                        'recid': recid, 'r_try': r_try, 'hash_choice': hc_out,
                        'pubkey': pk, 'puzzle_hash': h, 'is_valid_der': valid,
                    })
                    if valid:
                        break
            else:
                h, valid, hc_out = puzzle_hash(pk, hash_mode)
                candidates.append({
                    'recid': recid, 'r_try': r_try, 'hash_choice': hc_out,
                    'pubkey': pk, 'puzzle_hash': h, 'is_valid_der': valid,
                })

    hit = next((c for c in candidates if c['is_valid_der']), None)
    return {'sighash': z_bytes, 'preimage': preimage,
            'candidates': candidates, 'hit': hit}


if __name__ == '__main__':
    print("gpu_emulator loaded successfully.")
