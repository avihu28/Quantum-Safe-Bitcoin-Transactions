#!/usr/bin/env python3
"""
QSB GPU Search Orchestrator
============================
Generates script data, computes sighash values, calls the C/GPU search program.

Usage:
  python3 run_search.py bench                    # Benchmark EC recovery rate
  python3 run_search.py pin --diff 16            # Pinning search (easy)
  python3 run_search.py pin --diff 0             # Pinning search (real)
  python3 run_search.py full --diff 16           # Full search (pin + digest)
"""

import os
import sys
import time
import struct
import hashlib
import argparse
import subprocess
import itertools
import tempfile

# Add parent dir for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bitcoin_tx import (
    Transaction, TxIn, TxOut, QSBScriptBuilder,
    push_data, push_number, find_and_delete, serialize_varint,
)
from secp256k1 import (
    sha256d, ripemd160, encode_der_sig, point_mul, G, N,
)

QSB_SEARCH_BIN = os.path.join(os.path.dirname(__file__), 'qsb_search')


def make_fixed_sig(label):
    k = int.from_bytes(hashlib.sha256(f"qsb_{label}".encode()).digest(), 'big') % N
    R_pt = point_mul(k, G)
    r = R_pt[0] % N
    s = max(1, int.from_bytes(hashlib.sha256(f"qsb_{label}_s".encode()).digest()[:4], 'big') % (N // 2))
    return r, s, encode_der_sig(r, s, sighash=0x01)


def build_config_a():
    """Build Config A: n=150, t1=8+1b, t2=7+2b"""
    print("Building Config A (n=150, t1=8+1b, t2=7+2b)...")
    t0 = time.time()
    builder = QSBScriptBuilder(n=150, t1_signed=8, t1_bonus=1, t2_signed=7, t2_bonus=2)
    builder.generate_keys()
    
    pin_r, pin_s, pin_sig = make_fixed_sig("pin")
    r1_r, r1_s, r1_sig = make_fixed_sig("round1")
    r2_r, r2_s, r2_sig = make_fixed_sig("round2")
    
    full_script = builder.build_full_script(pin_sig, r1_sig, r2_sig)
    round1_script = builder.build_round_script(0, r1_sig)
    round2_script = builder.build_round_script(1, r2_sig)
    
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Script: {len(full_script)}B, R1: {len(round1_script)}B, R2: {len(round2_script)}B")
    
    return {
        'builder': builder,
        'pin_r': pin_r, 'pin_s': pin_s, 'pin_sig': pin_sig,
        'r1_r': r1_r, 'r1_s': r1_s, 'r1_sig': r1_sig,
        'r2_r': r2_r, 'r2_s': r2_s, 'r2_sig': r2_sig,
        'full_script': full_script,
        'round1_script': round1_script,
        'round2_script': round2_script,
    }


def create_transaction(full_script):
    """Create the spending transaction"""
    fake_txid = hashlib.sha256(b"qsb_funding_utxo_v1").digest()
    tx = Transaction(version=1, locktime=0)
    tx.add_input(TxIn(fake_txid, 0, b'', 0xffffffff))
    tx.add_output(TxOut(49000, b'\x76\xa9\x14' + b'\x00' * 20 + b'\x88\xac'))
    return tx


def get_tx_prefix(tx, script_code, sighash_type=0x01):
    """
    Get the serialized tx data up to (but not including) locktime.
    This is the prefix that stays fixed during pinning search.
    """
    # Serialize the sighash preimage manually
    result = struct.pack('<I', tx.version)
    result += serialize_varint(len(tx.inputs))
    for i, inp in enumerate(tx.inputs):
        result += inp.txid + struct.pack('<I', inp.vout)
        if i == 0:
            result += serialize_varint(len(script_code)) + script_code
        else:
            result += serialize_varint(0)
        result += struct.pack('<I', inp.sequence)
    result += serialize_varint(len(tx.outputs))
    for out in tx.outputs:
        result += out.serialize()
    # STOP here — don't add locktime or sighash_type
    return result


def r_s_to_hex(r, s):
    """Convert r, s integers to 32-byte hex strings"""
    return r.to_bytes(32, 'big').hex(), s.to_bytes(32, 'big').hex()


def run_bench():
    """Run the C benchmark"""
    if not os.path.exists(QSB_SEARCH_BIN):
        print(f"ERROR: {QSB_SEARCH_BIN} not found. Run: make")
        return
    subprocess.run([QSB_SEARCH_BIN, 'bench'])


def run_pinning(diff, count=1000000):
    """Run pinning search"""
    if not os.path.exists(QSB_SEARCH_BIN):
        print(f"ERROR: {QSB_SEARCH_BIN} not found. Run: make")
        return None
    
    cfg = build_config_a()
    tx = create_transaction(cfg['full_script'])
    
    # Get tx prefix (everything before locktime)
    tx_prefix = get_tx_prefix(tx, cfg['full_script'])
    
    r_hex, s_hex = r_s_to_hex(cfg['pin_r'], cfg['pin_s'])
    prefix_hex = tx_prefix.hex()
    
    print(f"\nRunning pinning search...")
    print(f"  sig_r: {r_hex[:20]}...")
    print(f"  sig_s: {s_hex[:20]}...")
    print(f"  tx_prefix: {len(tx_prefix)} bytes")
    print(f"  difficulty: 1/{diff if diff else '2^46'}")
    print(f"  count: {count}")
    
    result = subprocess.run(
        [QSB_SEARCH_BIN, 'pin', r_hex, s_hex, prefix_hex, str(diff), str(count)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result


def run_digest_search(cfg, tx, round_idx, diff, max_subsets=50000):
    """
    Compute sighash values for subsets on CPU, then send to C program for EC recovery.
    """
    if not os.path.exists(QSB_SEARCH_BIN):
        print(f"ERROR: {QSB_SEARCH_BIN} not found. Run: make")
        return None
    
    ts = cfg['builder'].t1_signed + cfg['builder'].t1_bonus if round_idx == 0 else \
         cfg['builder'].t2_signed + cfg['builder'].t2_bonus
    sig_bytes = cfg['r1_sig'] if round_idx == 0 else cfg['r2_sig']
    sig_r = cfg['r1_r'] if round_idx == 0 else cfg['r2_r']
    sig_s = cfg['r1_s'] if round_idx == 0 else cfg['r2_s']
    round_script = cfg['round1_script'] if round_idx == 0 else cfg['round2_script']
    
    print(f"\nComputing sighash values for Round {round_idx+1}...")
    print(f"  t_total={ts}, n=150, C(150,{ts}) subsets")
    print(f"  Max subsets: {max_subsets}")
    
    # Compute sighash for each subset on CPU
    t0 = time.time()
    z_values = bytearray()
    subset_list = []
    count = 0
    
    for subset in itertools.combinations(range(150), ts):
        if count >= max_subsets:
            break
        
        sc = round_script
        for idx in subset:
            sc = find_and_delete(sc, cfg['builder'].dummy_sigs[round_idx][idx])
        sc = find_and_delete(sc, sig_bytes)
        
        z = tx.sighash(0, sc, sighash_type=0x01)
        z_values.extend(z.to_bytes(32, 'big'))
        subset_list.append(subset)
        count += 1
        
        if count % 5000 == 0:
            elapsed = time.time() - t0
            print(f"    {count} sighashes computed ({count/elapsed:.0f}/s)")
    
    elapsed = time.time() - t0
    print(f"  {count} sighashes in {elapsed:.1f}s ({count/elapsed:.0f}/s)")
    
    # Write z values to temp file
    z_file = tempfile.mktemp(suffix='.bin')
    with open(z_file, 'wb') as f:
        f.write(z_values)
    
    # Call C program for batch EC recovery
    r_hex, s_hex = r_s_to_hex(sig_r, sig_s)
    
    print(f"  Running EC recovery on {count} values...")
    result = subprocess.run(
        [QSB_SEARCH_BIN, 'digest', r_hex, s_hex, z_file, str(diff)],
        capture_output=True, text=True
    )
    print(result.stdout)
    
    # Parse hits
    hits = []
    for line in result.stdout.split('\n'):
        if 'z_index=' in line:
            idx = int(line.strip().split('z_index=')[1])
            hits.append(subset_list[idx])
    
    os.unlink(z_file)
    return hits


def run_full(diff, max_subsets=50000):
    """Full pipeline: pin + digest for both rounds"""
    cfg = build_config_a()
    tx = create_transaction(cfg['full_script'])
    
    # Pinning
    pin_result = run_pinning(diff, count=100000)
    # TODO: parse locktime from output and set tx.locktime
    
    # For now, use a simple Python pinning search to get the locktime
    print("\nPython pinning (to get locktime)...")
    try:
        from secp256k1 import ecdsa_recover, compress_pubkey
        from secp256k1 import is_valid_der_sig
        check_fn = is_valid_der_sig if not diff else None
        if diff == 16:
            check_fn = lambda d: len(d) >= 9 and (d[0] >> 4) == 3
        elif diff == 256:
            check_fn = lambda d: len(d) >= 9 and d[0] == 0x30
        else:
            check_fn = lambda d: len(d) >= 9 and (d[0] >> 4) == 3  # default easy
        
        for lt in range(100000):
            tx.locktime = lt
            z = tx.sighash(0, cfg['full_script'], sighash_type=0x01)
            for flag in [0, 1]:
                Q = ecdsa_recover(cfg['pin_r'], cfg['pin_s'], z, flag)
                if Q is None: continue
                key = compress_pubkey(Q)
                h = ripemd160(key)
                if check_fn(h):
                    print(f"  Pin found: locktime={lt}")
                    tx.locktime = lt
                    break
            else:
                continue
            break
    except Exception as e:
        print(f"  Python pin failed: {e}")
        tx.locktime = 0
    
    # Digest rounds
    for rd in range(2):
        hits = run_digest_search(cfg, tx, rd, diff, max_subsets)
        if hits:
            print(f"  Round {rd+1} solution: {hits[0]}")
        else:
            print(f"  Round {rd+1}: no solution found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QSB GPU Search")
    parser.add_argument('mode', choices=['bench', 'pin', 'full'],
                       help='bench: benchmark EC rate, pin: pinning only, full: pin+digest')
    parser.add_argument('--diff', type=int, default=16,
                       help='Difficulty: 16, 256, 65536, or 0 for real')
    parser.add_argument('--count', type=int, default=1000000,
                       help='Max pinning attempts')
    parser.add_argument('--max-subsets', type=int, default=50000,
                       help='Max subsets per digest round')
    args = parser.parse_args()
    
    if args.mode == 'bench':
        run_bench()
    elif args.mode == 'pin':
        run_pinning(args.diff, args.count)
    elif args.mode == 'full':
        run_full(args.diff, args.max_subsets)
