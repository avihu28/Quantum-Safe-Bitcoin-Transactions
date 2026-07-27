#!/usr/bin/env python3
"""Verify a v14 R2 hit by comparing GPU's reported intermediates against CPU computation.

Usage:
  python3 verify_r2_hit.py <round2_final.txt> <qsb_state.json>

Or if pipeline files are in the standard place:
  python3 verify_r2_hit.py results/<inst_id>/round2_final.txt
"""
import sys, json, hashlib
from pathlib import Path

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

hit_path = sys.argv[1]
state_path = sys.argv[2] if len(sys.argv) > 2 else 'pipeline/qsb_state.json'

# Add pipeline to path so we can import bitcoin_tx, secp256k1
sys.path.insert(0, str(Path(state_path).parent))
from bitcoin_tx import find_and_delete, Transaction, TxIn, serialize_varint
from secp256k1 import (point_mul, point_add, compress_pubkey,
                       ecdsa_recover, is_valid_der_sig, modinv, N, P, G)

# Parse hit file
hit = {}
with open(hit_path) as f:
    for line in f:
        line = line.strip()
        if '=' in line:
            k, v = line.split('=', 1)
            hit[k.strip()] = v.strip()

with open(state_path) as f:
    state = json.load(f)

print("=== GPU's reported values ===")
indices = [int(x) for x in hit['indices'].split(',')]
recid = int(hit['recid'])
hc = int(hit['hash_choice'])
print(f"indices    = {indices}")
print(f"recid      = {recid}")
print(f"hc         = {hc}")
print(f"sighash    = {hit.get('sighash', '(missing)')}")
print(f"key_nonce  = {hit.get('key_nonce', '(missing)')}")
print(f"pubhash    = {hit.get('pubhash', '(missing)')}")
print(f"qx         = {hit.get('qx', '(missing)')}")
print(f"qy         = {hit.get('qy', '(missing)')}")

# Convert to original indices
N_POOL = 130
orig = sorted([N_POOL - 1 - k for k in indices])
print(f"\nConverted (original) indices: {orig}")

# Build sighash on CPU
def h2b(h): return bytes.fromhex(h)
full_script = h2b(state['full_script_hex'])
funding_txid = h2b('4fab76e9b0538a49a77443030f8e0243a5d2558155647a839acea0efaa4edc91')[::-1]

# Need pinning's seq/lt — assume from pinning file or hardcoded
pin_path = Path(hit_path).parent / 'pinning_hit.txt'
seq, lt = 2147507660, 570428883  # default = our verified pinning
if pin_path.exists():
    for line in pin_path.read_text().splitlines():
        if '=' in line:
            k, v = line.strip().split('=', 1)
            if k.strip() == 'sequence': seq = int(v)
            if k.strip() == 'locktime': lt = int(v)
print(f"\nUsing pinning seq={seq} lt={lt}")

sig_nonce = h2b(state['round_sigs'][1]['sig'])
sc = find_and_delete(full_script, sig_nonce)
for idx in orig:
    sc = find_and_delete(sc, h2b(state['dummy_sigs'][1][idx]))

tx = Transaction(version=2, locktime=lt)
tx.add_input(TxIn(funding_txid, 0, b'', seq))
z = tx.sighash(0, sc, sighash_type=0x01)

print(f"\n=== CPU recomputation ===")
print(f"sighash z  = {z:064x}")
gpu_z = hit.get('sighash')
if gpu_z:
    print(f"GPU sighash matches: {f'{z:064x}' == gpu_z}")

# Recover key_nonce
r_val = state['round_sigs'][1]['r']
s_val = state['round_sigs'][1]['s']
pt = ecdsa_recover(r_val, s_val, z, recid)
if not pt:
    print("CPU ECDSA recovery FAILED")
    sys.exit(1)

cpu_qx = pt[0]
cpu_qy = pt[1]
cpu_kn = compress_pubkey(pt)
print(f"CPU qx     = {cpu_qx:064x}")
print(f"CPU qy     = {cpu_qy:064x}")
print(f"CPU kn     = {cpu_kn.hex()}")

if hit.get('qx'):
    gpu_qx_int = int(hit['qx'], 16)
    print(f"qx match: {gpu_qx_int == cpu_qx}")
if hit.get('qy'):
    gpu_qy_int = int(hit['qy'], 16)
    print(f"qy match: {gpu_qy_int == cpu_qy}")
if hit.get('key_nonce'):
    print(f"kn match: {hit['key_nonce'] == cpu_kn.hex()}")

# Compute hash
cpu_h1 = hashlib.sha256(cpu_kn).digest()
cpu_h2 = hashlib.sha256(cpu_h1).digest()
cpu_pubhash = cpu_h1 if hc == 0 else cpu_h2
print(f"CPU pubhash (hc={hc}): {cpu_pubhash.hex()}")
if hit.get('pubhash'):
    print(f"pubhash match: {hit['pubhash'] == cpu_pubhash.hex()}")
print(f"CPU pubhash[0] = 0x{cpu_pubhash[0]:02x} (must be 0x30 for valid hit)")
print(f"CPU is_valid_der: {is_valid_der_sig(cpu_pubhash)}")

if hit.get('pubhash'):
    gpu_ph = bytes.fromhex(hit['pubhash'])
    print(f"GPU pubhash[0] = 0x{gpu_ph[0]:02x} (must be 0x30 for valid hit)")
    print(f"GPU is_valid_der (CPU rerun): {is_valid_der_sig(gpu_ph)}")
    if gpu_ph[0] != 0x30:
        print()
        print("🚨 CRITICAL: GPU's pubhash[0] != 0x30 but kernel reported a hit!")
        print("   This means gpu_is_valid_der incorrectly accepts a non-0x30 hash.")
        print("   The bug is confirmed in the GPU's DER check or compiler.")
