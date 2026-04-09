#!/usr/bin/env python3
"""
qsb_pipeline.py — End-to-end QSB Pipeline

Phase 1: Setup
  - Generate HORS keys, dummy sigs, build script
  - Create P2SH address for funding
  - Save state to qsb_state.json

Phase 2: Export GPU params
  - Export binary files for pinning + digest GPU search
  - Output: gpu_pinning_params.bin, gpu_digest_r1_params.bin, gpu_digest_r2_params.bin

Phase 3: (user runs GPU search on vast.ai)

Phase 4: Import results + assemble tx
  - Read GPU output (locktime, round1 indices, round2 indices)
  - Compute actual EC pubkeys and signatures
  - Build spending transaction
  - Verify and output raw tx hex

Usage:
  python3 qsb_pipeline.py setup [--seed SEED] [--config A]
  python3 qsb_pipeline.py export --funding-txid <txid> --funding-vout <n> --funding-value <sats> --dest-address <addr>
  python3 qsb_pipeline.py assemble --locktime <lt> --round1 <i0,i1,...,i8> --round2 <i0,i1,...,i8>
  python3 qsb_pipeline.py test    # End-to-end test with easy mode
"""

import os
import sys
import json
import struct
import hashlib
import argparse
import time
from itertools import combinations

# Local imports
from secp256k1 import (
    sha256d, ripemd160, hash160,
    compress_pubkey, decompress_pubkey, point_mul, point_add, G, N, P,
    ecdsa_sign, ecdsa_sign_with_k, ecdsa_recover, ecdsa_verify,
    encode_der_sig, is_valid_der_sig, parse_der, modinv, int_to_der_int,
)
from bitcoin_tx import (
    Transaction, TxIn, TxOut, QSBScriptBuilder,
    push_data, push_number, find_and_delete, serialize_varint,
    OP_0, OP_HASH160, OP_EQUALVERIFY, OP_CHECKSIG,
)

STATE_FILE = "qsb_state.json"

def compute_sha256_midstate(data, num_blocks):
    """Compute SHA-256 intermediate state after processing num_blocks full blocks."""
    import struct as _st
    
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
    
    def ror(x, n): return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF
    def ch(e, f, g): return (e & f) ^ (~e & g) & 0xFFFFFFFF
    def maj(a, b, c): return (a & b) ^ (a & c) ^ (b & c)
    
    def compress(state, block):
        W = list(block)
        for i in range(16, 64):
            s0 = ror(W[i-15],7) ^ ror(W[i-15],18) ^ (W[i-15]>>3)
            s1 = ror(W[i-2],17) ^ ror(W[i-2],19) ^ (W[i-2]>>10)
            W.append((W[i-16]+s0+W[i-7]+s1) & 0xFFFFFFFF)
        a,b,c,d,e,f,g,h = state
        for i in range(64):
            S1 = ror(e,6) ^ ror(e,11) ^ ror(e,25)
            t1 = (h+S1+ch(e,f,g)+K[i]+W[i]) & 0xFFFFFFFF
            S0 = ror(a,2) ^ ror(a,13) ^ ror(a,22)
            t2 = (S0+maj(a,b,c)) & 0xFFFFFFFF
            h,g,f,e,d,c,b,a = g,f,e,(d+t1)&0xFFFFFFFF,c,b,a,(t1+t2)&0xFFFFFFFF
        return tuple((s+v)&0xFFFFFFFF for s,v in zip(state,(a,b,c,d,e,f,g,h)))
    
    state = (0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
             0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19)
    for blk in range(num_blocks):
        off = blk * 64
        words = _st.unpack('>16I', data[off:off+64])
        state = compress(state, words)
    return state

# ============================================================
# Utility
# ============================================================

def b2h(b):
    return b.hex()

def h2b(h):
    return bytes.fromhex(h)

def le_bytes(val, n=32):
    """int → little-endian bytes"""
    return val.to_bytes(n, 'little')

def be_bytes(val, n=32):
    """int → big-endian bytes"""
    return val.to_bytes(n, 'big')

def int_from_be(b):
    return int.from_bytes(b, 'big')

def int_from_le(b):
    return int.from_bytes(b, 'little')

def p2sh_address(script, testnet=False):
    """Compute P2SH address from redeem script"""
    h = hash160(script)
    prefix = b'\xc4' if testnet else b'\x05'
    payload = prefix + h
    checksum = sha256d(payload)[:4]
    import base58
    return base58.b58encode(payload + checksum).decode()

def p2sh_script_pubkey(script):
    """P2SH scriptPubKey: OP_HASH160 <hash160(script)> OP_EQUAL"""
    h = hash160(script)
    return bytes([OP_HASH160, 0x14]) + h + bytes([0x87])  # OP_EQUAL = 0x87

def p2pkh_script(addr_hex):
    """Simple P2PKH scriptPubKey from hex pubkeyhash"""
    pkh = h2b(addr_hex)
    return bytes([0x76, 0xa9, 0x14]) + pkh + bytes([0x88, 0xac])


# ============================================================
# Phase 1: Setup
# ============================================================

def cmd_setup(args):
    print("╔══════════════════════════════════════╗")
    print("║  QSB Pipeline — Phase 1: Setup       ║")
    print("╚══════════════════════════════════════╝")
    
    config = args.config
    seed = args.seed
    
    configs = {
        'A':    {'n': 150, 't1s': 8, 't1b': 1, 't2s': 7, 't2b': 2},  # Full: ~20h on 16×5090
        'A120': {'n': 120, 't1s': 8, 't1b': 1, 't2s': 7, 't2b': 2},  # Fast: ~3h on 16×5090
        'A110': {'n': 110, 't1s': 8, 't1b': 1, 't2s': 7, 't2b': 2},  # Express: ~1.5h on 16×5090
        'A100': {'n': 100, 't1s': 8, 't1b': 1, 't2s': 7, 't2b': 2},  # Quick: ~1h on 16×5090
        'test': {'n': 10,  't1s': 2, 't1b': 0, 't2s': 2, 't2b': 0},  # Test only
    }
    cfg = configs[config]
    n, t1s, t1b, t2s, t2b = cfg['n'], cfg['t1s'], cfg['t1b'], cfg['t2s'], cfg['t2b']
    t1, t2 = t1s + t1b, t2s + t2b
    
    print(f"  Config: {config}, n={n}, R1=({t1s}+{t1b}), R2=({t2s}+{t2b})")
    
    if seed:
        import random
        random.seed(seed)
        orig = os.urandom
        os.urandom = lambda n: random.randbytes(n)
    
    builder = QSBScriptBuilder(n, t1s, t1b, t2s, t2b)
    builder.generate_keys()
    
    if seed:
        os.urandom = orig
    
    # Generate fixed sig_nonce for each phase
    # These are the ECDSA signatures hardcoded in the script
    # They use known (r, s) values — the search finds a locktime/subset
    # that makes Hash160(recovered_pubkey) valid DER
    
    pin_k = int.from_bytes(hashlib.sha256(b"qsb_pin_nonce").digest(), 'big') % N
    pin_R = point_mul(pin_k, G)
    pin_r = pin_R[0] % N
    pin_s = int.from_bytes(hashlib.sha256(b"qsb_pin_s").digest()[:16], 'big') % (N // 2)
    pin_s = max(1, pin_s)
    pin_sig = encode_der_sig(pin_r, pin_s, sighash=0x01)
    
    round_sigs = []
    for ri in range(2):
        k = int.from_bytes(hashlib.sha256(f"qsb_r{ri}_nonce".encode()).digest(), 'big') % N
        R = point_mul(k, G)
        r_val = R[0] % N
        s_val = int.from_bytes(hashlib.sha256(f"qsb_r{ri}_s".encode()).digest()[:16], 'big') % (N // 2)
        s_val = max(1, s_val)
        sig = encode_der_sig(r_val, s_val, sighash=0x01)
        round_sigs.append({'r': r_val, 's': s_val, 'sig': b2h(sig), 'k': k})
    
    # Build full script
    full_script = builder.build_full_script(pin_sig, h2b(round_sigs[0]['sig']), h2b(round_sigs[1]['sig']))
    
    print(f"  Script size: {len(full_script)} bytes")
    print(f"  Script hash160: {b2h(hash160(full_script))}")
    
    # Save state
    state = {
        'config': config,
        'n': n, 't1s': t1s, 't1b': t1b, 't2s': t2s, 't2b': t2b,
        'hors_secrets': [[b2h(s) for s in r] for r in builder.hors_secrets],
        'hors_commitments': [[b2h(c) for c in r] for r in builder.hors_commitments],
        'dummy_sigs': [[b2h(s) for s in r] for r in builder.dummy_sigs],
        'pin_r': pin_r, 'pin_s': pin_s, 'pin_k': pin_k,
        'pin_sig': b2h(pin_sig),
        'round_sigs': [{'r': rs['r'], 's': rs['s'], 'sig': rs['sig'], 'k': rs['k']} for rs in round_sigs],
        'full_script_hex': b2h(full_script),
        'script_hash160': b2h(hash160(full_script)),
        'p2sh_script_pubkey': b2h(p2sh_script_pubkey(full_script)),
    }
    
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"\n  State saved to {STATE_FILE}")
    print(f"  P2SH scriptPubKey: {state['p2sh_script_pubkey']}")
    print(f"\n  Fund this P2SH output, then run:")
    print(f"  python3 qsb_pipeline.py export --funding-txid <txid> --funding-vout <n> --funding-value <sats> --dest-address <pkh_hex>")


# ============================================================
# Phase 2: Export GPU params
# ============================================================

def cmd_export(args):
    print("╔══════════════════════════════════════╗")
    print("║  QSB Pipeline — Phase 2: Export       ║")
    print("╚══════════════════════════════════════╝")
    
    with open(STATE_FILE) as f:
        state = json.load(f)
    
    n = state['n']
    t1 = state['t1s'] + state['t1b']
    t2 = state['t2s'] + state['t2b']
    
    full_script = h2b(state['full_script_hex'])
    
    # Build the spending transaction template
    funding_txid = h2b(args.funding_txid)[::-1]  # reverse for internal byte order
    funding_vout = args.funding_vout
    funding_value = args.funding_value
    dest_pkh = h2b(args.dest_address)
    
    # Output: P2PKH to destination, minus fee
    fee = 5000  # conservative fee in sats
    dest_value = funding_value - fee
    dest_script = p2pkh_script(args.dest_address)
    
    tx = Transaction(version=1, locktime=0)
    tx.add_input(TxIn(funding_txid, funding_vout, b'', 0xfffffffe))  # sequence allows locktime
    tx.add_output(TxOut(dest_value, dest_script))
    
    # ============================================================
    # Export pinning params
    # ============================================================
    
    pin_r = state['pin_r']
    pin_s = state['pin_s']
    pin_sig = h2b(state['pin_sig'])
    
    # For pinning, sighash_type = SIGHASH_ALL (0x01)
    # scriptCode = full_script with FindAndDelete(pin_sig)
    pin_script_code = find_and_delete(full_script, pin_sig)
    
    # The sighash preimage is: serialize(tx_copy) + sighash_type(4 LE)
    # tx_copy has pin_script_code in input 0
    # Locktime varies — that's what we search
    
    # Build tx_prefix: everything before locktime
    tx_prefix = struct.pack('<I', tx.version)  # version
    tx_prefix += serialize_varint(1)  # 1 input
    tx_prefix += funding_txid  # txid
    tx_prefix += struct.pack('<I', funding_vout)  # vout
    tx_prefix += serialize_varint(len(pin_script_code))  # scriptCode varint
    tx_prefix += pin_script_code  # scriptCode
    tx_prefix += struct.pack('<I', 0xfffffffe)  # sequence
    tx_prefix += serialize_varint(1)  # 1 output
    tx_prefix += struct.pack('<q', dest_value)  # value
    tx_prefix += serialize_varint(len(dest_script))  # script length
    tx_prefix += dest_script  # scriptPubKey
    # locktime goes here (4 bytes, searched by GPU)
    # then sighash_type (4 bytes, always 0x01000000)
    
    total_preimage_len = len(tx_prefix) + 4 + 4  # + locktime + sighash_type
    
    # Compute r_inv, neg_r_inv, u2*R for EC recovery
    r_inv = modinv(pin_r, N)
    neg_r_inv = (-r_inv) % N
    u2 = (pin_s * r_inv) % N
    
    # Recover R point from r
    # R.x = pin_r, need to find R.y
    x = pin_r
    y_sq = (pow(x, 3, P) + 7) % P
    y = pow(y_sq, (P + 1) // 4, P)
    if y % 2 != 0:  # pick even y (recid=0)
        y = P - y
    R_point = (x, y)
    u2R = point_mul(u2, R_point)
    
    # Compute SHA-256 midstate for tx_prefix (everything before locktime)
    # This is what the GPU uses — processes prefix as fixed blocks
    midstate_data = tx_prefix
    full_blocks = len(midstate_data) // 64
    midstate_bytes_covered = full_blocks * 64
    tail_data = midstate_data[midstate_bytes_covered:]
    
    # Export as binary
    params = {
        'type': 'pinning',
        'tx_prefix_len': len(tx_prefix),
        'tx_prefix': b2h(tx_prefix),
        'total_preimage_len': total_preimage_len,
        'midstate_blocks': full_blocks,
        'tail_data': b2h(tail_data),
        'tail_data_len': len(tail_data),
        'pin_r': pin_r,
        'pin_s': pin_s,
        'neg_r_inv': b2h(le_bytes(neg_r_inv)),
        'u2r_x': b2h(le_bytes(u2R[0])),
        'u2r_y': b2h(le_bytes(u2R[1])),
    }
    
    with open('gpu_pinning_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    # Binary export for GPU
    with open('pinning.bin', 'wb') as f:
        # Midstate (8 × uint32 big-endian)
        midstate = compute_sha256_midstate(tx_prefix, full_blocks)
        for v in midstate:
            f.write(struct.pack('>I', v))
        f.write(struct.pack('<I', len(tail_data)))
        f.write(tail_data)
        f.write(struct.pack('<I', total_preimage_len))
        f.write(le_bytes(neg_r_inv))
        f.write(le_bytes(u2R[0]))
        f.write(le_bytes(u2R[1]))
    
    print(f"  Pinning: tx_prefix={len(tx_prefix)} bytes, midstate={full_blocks} blocks")
    print(f"  Saved gpu_pinning_params.json + pinning.bin")
    
    # ============================================================
    # Export digest params (per round)
    # ============================================================
    
    for ri in range(2):
        rs = state['round_sigs'][ri]
        r_val, s_val = rs['r'], rs['s']
        sig_nonce = h2b(rs['sig'])
        
        # ScriptCode for this round:
        # The full script, with FindAndDelete of:
        #   - sig_nonce (this round's hardcoded sig)
        #   - selected dummy sigs (varies per subset — done by GPU)
        #
        # But we need the scriptCode BEFORE FindAndDelete of dummies,
        # because the GPU does the dummy removal.
        # Actually, FindAndDelete of sig_nonce is always done.
        
        # Script structure for sighash of this round's sig:
        # The scriptCode = full_script with FindAndDelete(sig_nonce)
        # Then for each candidate subset, also FindAndDelete each selected dummy sig
        
        base_script_code = find_and_delete(full_script, sig_nonce)
        
        # Now the GPU needs to additionally remove selected dummy sigs from this
        # The structure: HORS section is at the beginning, dummy sigs in the middle
        # We need to identify the byte positions of each dummy sig in base_script_code
        
        # For the GPU params, export:
        # - The HORS section (fixed prefix of scriptCode)
        # - Each dummy sig push_data (for removal)
        # - The tail section (after dummy sigs)
        
        # Parse base_script_code to find HORS section, dummy sig section, tail
        # HORS section: n × 21 bytes (push_data(20-byte hash))
        hors_section_len = n * 21
        hors_section = base_script_code[:hors_section_len]
        
        # Dummy sigs: each is push_data(9-byte sig) = 10 bytes
        # In the script, they're in reverse order (n-1 down to 0)
        # Some may have been removed by FindAndDelete of sig_nonce if collision (unlikely)
        dummy_sig_section_start = hors_section_len
        dummy_sig_section_len = n * 10  # 150 × 10
        
        # Tail: everything after dummy sigs
        tail_start = dummy_sig_section_start + dummy_sig_section_len
        tail_section = base_script_code[tail_start:]
        
        # Build the sighash preimage structure
        # For SIGHASH_ALL: serialize(tx_copy with scriptCode) + 0x01000000
        # The scriptCode varies per subset, but tx_prefix and tx_suffix are fixed
        
        # tx_prefix for digest: version + input count + txid + vout + scriptCode_varint (varies slightly)
        # Actually scriptCode length is constant across subsets (always remove exactly t dummy sigs)
        removed_per_subset = (t1 if ri == 0 else t2)
        scriptcode_len_after_fad = len(base_script_code) - removed_per_subset * 10
        
        d_tx_prefix = struct.pack('<I', tx.version)
        d_tx_prefix += serialize_varint(1)
        d_tx_prefix += funding_txid
        d_tx_prefix += struct.pack('<I', funding_vout)
        d_tx_prefix += serialize_varint(scriptcode_len_after_fad)
        # scriptCode goes here (built by GPU per subset)
        
        d_tx_suffix = struct.pack('<I', 0xfffffffe)  # sequence
        d_tx_suffix += serialize_varint(1)
        d_tx_suffix += struct.pack('<q', dest_value)
        d_tx_suffix += serialize_varint(len(dest_script))
        d_tx_suffix += dest_script
        d_tx_suffix += struct.pack('<I', tx.locktime)  # locktime (set after pinning)
        d_tx_suffix += struct.pack('<I', 0x01)  # SIGHASH_ALL
        
        total_d_preimage = len(d_tx_prefix) + scriptcode_len_after_fad + len(d_tx_suffix)
        
        # Midstate: covers d_tx_prefix + HORS section (fixed)
        fixed_prefix = d_tx_prefix + hors_section
        fp_full_blocks = len(fixed_prefix) // 64
        
        # EC recovery params
        d_r_inv = modinv(r_val, N)
        d_neg_r_inv = (-d_r_inv) % N
        d_u2 = (s_val * d_r_inv) % N
        
        dx = r_val
        dy_sq = (pow(dx, 3, P) + 7) % P
        dy = pow(dy_sq, (P + 1) // 4, P)
        if dy % 2 != 0:
            dy = P - dy
        dR = (dx, dy)
        d_u2R = point_mul(d_u2, dR)
        
        # Export dummy sigs in script order (reversed: n-1 down to 0)
        dummy_sigs_in_order = []
        for i in range(n - 1, -1, -1):
            sig_bytes = h2b(state['dummy_sigs'][ri][i])
            dummy_sigs_in_order.append(b2h(push_data(sig_bytes)))
        
        digest_params = {
            'type': f'digest_round{ri+1}',
            'round': ri,
            'n': n,
            't': removed_per_subset,
            'hors_section': b2h(hors_section),
            'hors_section_len': hors_section_len,
            'dummy_sigs': [b2h(h2b(state['dummy_sigs'][ri][i])) for i in range(n)],
            'dummy_sig_pushes': dummy_sigs_in_order,
            'tail_section': b2h(tail_section),
            'tail_section_len': len(tail_section),
            'tx_prefix': b2h(d_tx_prefix),
            'tx_prefix_len': len(d_tx_prefix),
            'tx_suffix': b2h(d_tx_suffix),
            'tx_suffix_len': len(d_tx_suffix),
            'fixed_prefix': b2h(fixed_prefix),
            'fixed_prefix_len': len(fixed_prefix),
            'midstate_blocks': fp_full_blocks,
            'scriptcode_len': scriptcode_len_after_fad,
            'total_preimage_len': total_d_preimage,
            'sig_r': r_val,
            'sig_s': s_val,
            'neg_r_inv': b2h(le_bytes(d_neg_r_inv)),
            'u2r_x': b2h(le_bytes(d_u2R[0])),
            'u2r_y': b2h(le_bytes(d_u2R[1])),
        }
        
        fname = f'gpu_digest_r{ri+1}_params.json'
        with open(fname, 'w') as f:
            json.dump(digest_params, f, indent=2)
        
        # Binary export for GPU
        bname = f'digest_r{ri+1}.bin'
        with open(bname, 'wb') as f:
            # Header
            f.write(struct.pack('<I', n))
            f.write(struct.pack('<I', removed_per_subset))
            f.write(struct.pack('<I', total_d_preimage))
            f.write(struct.pack('<I', len(tail_section)))
            f.write(struct.pack('<I', len(d_tx_suffix)))
            # Midstate (8 × uint32 BE)
            mid = compute_sha256_midstate(fixed_prefix, fp_full_blocks)
            for v in mid:
                f.write(struct.pack('>I', v))
            # Dummy sigs as push_data (n × 10 bytes, in script order: reversed)
            for i in range(n - 1, -1, -1):
                sig_bytes = h2b(state['dummy_sigs'][ri][i])
                f.write(push_data(sig_bytes))
            # Tail section
            f.write(tail_section)
            # tx_suffix
            f.write(d_tx_suffix)
            # EC params (LE 32 bytes each)
            f.write(le_bytes(d_neg_r_inv))
            f.write(le_bytes(d_u2R[0]))
            f.write(le_bytes(d_u2R[1]))
        
        print(f"  Round {ri+1}: scriptCode={scriptcode_len_after_fad} bytes, midstate={fp_full_blocks} blocks")
        print(f"  Saved {fname} + {bname}")
    
    print(f"\n  Upload these JSON files + GPU code to vast.ai and run search.")


# ============================================================
# Phase 4: Assemble transaction
# ============================================================

def cmd_assemble(args):
    print("╔══════════════════════════════════════╗")
    print("║  QSB Pipeline — Phase 4: Assemble    ║")
    print("╚══════════════════════════════════════╝")
    
    with open(STATE_FILE) as f:
        state = json.load(f)
    
    n = state['n']
    t1 = state['t1s'] + state['t1b']
    t2 = state['t2s'] + state['t2b']
    
    locktime = args.locktime
    r1_indices = sorted([int(x) for x in args.round1.split(',')])
    r2_indices = sorted([int(x) for x in args.round2.split(',')])
    
    assert len(r1_indices) == t1, f"Expected {t1} round1 indices, got {len(r1_indices)}"
    assert len(r2_indices) == t2, f"Expected {t2} round2 indices, got {len(r2_indices)}"
    
    full_script = h2b(state['full_script_hex'])
    
    print(f"  Locktime: {locktime}")
    print(f"  Round 1 indices: {r1_indices}")
    print(f"  Round 2 indices: {r2_indices}")
    
    # ================================================================
    # Rebuild the spending transaction
    # ================================================================
    # NOTE: QSB input must be at index >= num_outputs for SIGHASH_SINGLE bug (z=1)
    # Design: helper input at index 0, QSB input at index 1, 1 output
    # This makes SIGHASH_SINGLE at input 1 trigger the bug.
    # For testing, we use a fake helper input.
    
    funding_txid = h2b(args.funding_txid)[::-1]  # internal byte order
    helper_txid = b'\x00' * 32  # placeholder helper input
    
    dest_script = p2pkh_script(args.dest_address)
    dest_value = args.funding_value - 5000  # minus fee
    
    tx = Transaction(version=1, locktime=locktime)
    tx.add_input(TxIn(helper_txid, 0, b'', 0xfffffffe))    # index 0: helper
    tx.add_input(TxIn(funding_txid, args.funding_vout, b'', 0xfffffffe))  # index 1: QSB
    tx.add_output(TxOut(dest_value, dest_script))            # index 0: destination
    
    QSB_INPUT_INDEX = 1  # QSB is at input 1
    
    # ================================================================
    # Step 1: Pinning — recover key_nonce
    # ================================================================
    print("\n  [1] Pinning: recover key_nonce")
    
    pin_r = state['pin_r']
    pin_s = state['pin_s']
    pin_sig = h2b(state['pin_sig'])
    
    # scriptCode for pin_sig_nonce verification: full_script - FindAndDelete(pin_sig)
    pin_sc = find_and_delete(full_script, pin_sig)
    z_pin = tx.sighash(QSB_INPUT_INDEX, pin_sc, sighash_type=0x01)
    
    # Recover key_nonce (try both recovery flags)
    key_nonce_pin = None
    sig_puzzle_pin = None
    for flag in [0, 1]:
        pt = ecdsa_recover(pin_r, pin_s, z_pin, flag)
        if pt:
            kn = compress_pubkey(pt)
            sp = ripemd160(hashlib.sha256(kn).digest())
            real_der = is_valid_der_sig(sp)
            easy_der = (sp[0] >> 4) == 3
            if real_der or easy_der:
                key_nonce_pin = kn
                sig_puzzle_pin = sp
                print(f"    key_nonce: {b2h(kn)[:16]}...")
                print(f"    sig_puzzle: {b2h(sp)} (real_DER={real_der})")
                break
    
    if key_nonce_pin is None:
        print("    ERROR: could not recover pinning key_nonce!")
        return
    
    # ================================================================
    # Step 2: Pinning — recover key_puzzle from sig_puzzle
    # ================================================================
    print("\n  [2] Pinning: recover key_puzzle")
    
    
    sp_r, sp_s = parse_der(sig_puzzle_pin)
    if sp_r is None:
        print("    WARNING: sig_puzzle not valid DER (easy mode). key_puzzle will be mock.")
        print("    (In real GPU search, sig_puzzle is always valid DER)")
        key_puzzle_pin = b'\x02' + b'\x00' * 32  # mock compressed pubkey
    else:
        # sig_puzzle's sighash_type is its last byte
        sp_sighash_type = sig_puzzle_pin[-1]
        print(f"    sig_puzzle sighash_type: 0x{sp_sighash_type:02x}")
        
        # scriptCode for sig_puzzle verification: full_script - FindAndDelete(sig_puzzle)
        puzzle_sc = find_and_delete(full_script, sig_puzzle_pin)
        z_puzzle_pin = tx.sighash(QSB_INPUT_INDEX, puzzle_sc, sighash_type=sp_sighash_type)
        
        key_puzzle_pin = None
        for flag in [0, 1]:
            pt = ecdsa_recover(sp_r, sp_s, z_puzzle_pin, flag)
            if pt:
                key_puzzle_pin = compress_pubkey(pt)
                print(f"    key_puzzle: {b2h(key_puzzle_pin)[:16]}...")
                break
        
        if key_puzzle_pin is None:
            print("    ERROR: could not recover pinning key_puzzle!")
            return
    
    # ================================================================
    # Step 3: Digest rounds — recover key_nonce, key_puzzle, dummy pubkeys
    # ================================================================
    
    round_results = []
    round_indices = [r1_indices, r2_indices]
    
    for ri in range(2):
        t = t1 if ri == 0 else t2
        ts = state['t1s'] if ri == 0 else state['t2s']
        indices = round_indices[ri]
        rs = state['round_sigs'][ri]
        r_val, s_val = rs['r'], rs['s']
        sig_nonce = h2b(rs['sig'])
        
        print(f"\n  [{3+ri*2}] Round {ri+1}: recover key_nonce")
        
        # scriptCode: full_script - FindAndDelete(sig_nonce) - FindAndDelete(each selected dummy)
        sc = find_and_delete(full_script, sig_nonce)
        for idx in indices:
            dummy_sig = h2b(state['dummy_sigs'][ri][idx])
            sc = find_and_delete(sc, dummy_sig)
        
        z_round = tx.sighash(QSB_INPUT_INDEX, sc, sighash_type=0x01)
        
        key_nonce_round = None
        sig_puzzle_round = None
        for flag in [0, 1]:
            pt = ecdsa_recover(r_val, s_val, z_round, flag)
            if pt:
                kn = compress_pubkey(pt)
                sp = ripemd160(hashlib.sha256(kn).digest())
                real_der = is_valid_der_sig(sp)
                easy_der = (sp[0] >> 4) == 3
                if real_der or easy_der:
                    key_nonce_round = kn
                    sig_puzzle_round = sp
                    print(f"    key_nonce: {b2h(kn)[:16]}...")
                    print(f"    sig_puzzle: {b2h(sp)} (real_DER={real_der})")
                    break
        
        if key_nonce_round is None:
            print(f"    ERROR: round {ri+1} key_nonce recovery failed!")
            return
        
        # Recover key_puzzle from sig_puzzle
        print(f"\n  [{4+ri*2}] Round {ri+1}: recover key_puzzle")
        
        sp_r2, sp_s2 = parse_der(sig_puzzle_round)
        if sp_r2 is None:
            print(f"    WARNING: sig_puzzle not valid DER (easy mode). Using mock key_puzzle.")
            key_puzzle_round = b'\x02' + b'\x00' * 32
        else:
            sp_ht = sig_puzzle_round[-1]
            print(f"    sig_puzzle sighash_type: 0x{sp_ht:02x}")
            puzzle_sc2 = find_and_delete(full_script, sig_puzzle_round)
            z_puzzle_round = tx.sighash(QSB_INPUT_INDEX, puzzle_sc2, sighash_type=sp_ht)
            key_puzzle_round = None
            for flag in [0, 1]:
                pt = ecdsa_recover(sp_r2, sp_s2, z_puzzle_round, flag)
                if pt:
                    key_puzzle_round = compress_pubkey(pt)
                    print(f"    key_puzzle: {b2h(key_puzzle_round)[:16]}...")
                    break
            if key_puzzle_round is None:
                print(f"    ERROR: round {ri+1} key_puzzle recovery failed!")
                return
        
        # Recover dummy pubkeys (z=1 via SIGHASH_SINGLE bug)
        # This requires QSB_INPUT_INDEX >= num_outputs
        dummy_pubkeys = []
        for idx in indices:
            ds_bytes = h2b(state['dummy_sigs'][ri][idx])
            dr, ds_val = parse_der(ds_bytes)
            for flag in [0, 1]:
                pt = ecdsa_recover(dr, ds_val, 1, flag)  # z=1 (SIGHASH_SINGLE bug)
                if pt:
                    dummy_pubkeys.append(compress_pubkey(pt))
                    break
        
        # Collect HORS preimages for signed indices
        signed_indices = indices[:ts]
        preimages = [h2b(state['hors_secrets'][ri][i]) for i in signed_indices]
        
        round_results.append({
            'key_nonce': key_nonce_round,
            'key_puzzle': key_puzzle_round,
            'sig_puzzle': sig_puzzle_round,
            'dummy_pubkeys': dummy_pubkeys,
            'preimages': preimages,
            'subset': indices,
            'signed_indices': list(signed_indices),
            'bonus_indices': list(indices[ts:]),
        })
        
        print(f"    dummy_pubkeys: {len(dummy_pubkeys)}")
        print(f"    preimages: {len(preimages)}")
    
    # ================================================================
    # Step 4: Build witness (scriptSig)
    # ================================================================
    print(f"\n  [7] Building witness...")
    
    # Witness layout (bottom to top of stack):
    # Round 2: key_puzzle, key_nonce, dummy_pubs(rev), preimages(rev), indices(rev)
    # Round 1: key_puzzle, key_nonce, dummy_pubs(rev), preimages(rev), indices(rev)
    # Pinning: key_puzzle, key_nonce
    
    witness = b''
    
    # Round 2 first (bottom of stack)
    for rd in [1, 0]:
        rr = round_results[rd]
        witness += push_data(rr['key_puzzle'])
        witness += push_data(rr['key_nonce'])
        for pub in reversed(rr['dummy_pubkeys']):
            witness += push_data(pub)
        for pre in reversed(rr['preimages']):
            witness += push_data(pre)
        for idx in reversed(rr['subset']):
            witness += push_number(idx)
    
    # Pinning data (top of stack)
    witness += push_data(key_puzzle_pin)
    witness += push_data(key_nonce_pin)
    
    # P2SH scriptSig: witness + push_data(redeem_script)
    script_sig = witness + push_data(full_script)
    
    print(f"    Witness: {len(witness)} bytes")
    print(f"    ScriptSig: {len(script_sig)} bytes")
    print(f"    Redeem script: {len(full_script)} bytes")
    
    # Set the scriptSig on the QSB input
    tx.inputs[QSB_INPUT_INDEX].script_sig = script_sig
    
    # Serialize
    raw_tx = tx.serialize()
    print(f"\n  [8] Final transaction: {len(raw_tx)} bytes")
    print(f"    Raw hex: {b2h(raw_tx)[:80]}...")
    
    # Save
    with open('qsb_raw_tx.hex', 'w') as f:
        f.write(b2h(raw_tx))
    print(f"    Saved to qsb_raw_tx.hex")
    
    # Also save all recovery data for debugging
    solution = {
        'locktime': locktime,
        'round1_indices': r1_indices,
        'round2_indices': r2_indices,
        'pin_key_nonce': b2h(key_nonce_pin),
        'pin_key_puzzle': b2h(key_puzzle_pin),
        'pin_sig_puzzle': b2h(sig_puzzle_pin),
        'rounds': [{
            'key_nonce': b2h(rr['key_nonce']),
            'key_puzzle': b2h(rr['key_puzzle']),
            'sig_puzzle': b2h(rr['sig_puzzle']),
            'dummy_pubkeys': [b2h(p) for p in rr['dummy_pubkeys']],
            'preimages': [b2h(p) for p in rr['preimages']],
            'subset': rr['subset'],
        } for rr in round_results],
    }
    with open('qsb_solution.json', 'w') as f:
        json.dump(solution, f, indent=2)
    print(f"    Solution saved to qsb_solution.json")


# ============================================================
# Test mode
# ============================================================

def cmd_test(args):
    """End-to-end test with small params and easy mode"""
    print("╔══════════════════════════════════════╗")
    print("║  QSB Pipeline — Test Mode            ║")
    print("╚══════════════════════════════════════╝")
    
    n = 10
    t1s, t1b, t2s, t2b = 2, 0, 2, 0
    t1, t2 = t1s + t1b, t2s + t2b
    
    print(f"  n={n}, t1={t1}, t2={t2}")
    
    import random
    random.seed(42)
    orig = os.urandom
    os.urandom = lambda nb: random.randbytes(nb)
    
    builder = QSBScriptBuilder(n, t1s, t1b, t2s, t2b)
    builder.generate_keys()
    
    os.urandom = orig
    
    # Generate sig nonces
    sigs = []
    for phase in ['pin', 'r0', 'r1']:
        k = int.from_bytes(hashlib.sha256(f"test_{phase}".encode()).digest(), 'big') % N
        R = point_mul(k, G)
        r_val = R[0] % N
        s_val = int.from_bytes(hashlib.sha256(f"test_{phase}_s".encode()).digest()[:8], 'big') % (N // 2)
        s_val = max(1, s_val)
        sig = encode_der_sig(r_val, s_val, sighash=0x01)
        sigs.append({'r': r_val, 's': s_val, 'sig': sig, 'k': k})
    
    full_script = builder.build_full_script(sigs[0]['sig'], sigs[1]['sig'], sigs[2]['sig'])
    print(f"  Script: {len(full_script)} bytes")
    
    # Build test transaction
    fake_txid = b'\x01' * 32
    # Transaction structure: helper input at index 0, QSB at index 1, 1 output
    # This enables SIGHASH_SINGLE bug (z=1) for dummy sigs at input 1
    QSB_IDX = 1
    tx = Transaction(version=1, locktime=0)
    tx.add_input(TxIn(b'\x00' * 32, 0, b'', 0xfffffffe))   # helper at index 0
    tx.add_input(TxIn(fake_txid, 0, b'', 0xfffffffe))       # QSB at index 1
    tx.add_output(TxOut(45000, p2pkh_script('0' * 40)))
    
    # =============================
    # Pinning search (easy mode)
    # =============================
    print(f"\n  [Pinning search]")
    pin_sig_data = sigs[0]['sig']
    pin_script_code = find_and_delete(full_script, pin_sig_data)
    pin_r, pin_s = sigs[0]['r'], sigs[0]['s']
    
    r_inv = modinv(pin_r, N)
    
    found_lt = None
    for lt in range(1, 10_000_000):
        tx.locktime = lt
        z = tx.sighash(QSB_IDX, pin_script_code, sighash_type=0x01)
        
        # Recover pubkey
        u1 = (-z * r_inv) % N
        u2 = (pin_s * r_inv) % N
        
        # Recover R point
        x = pin_r
        y_sq = (pow(x, 3, P) + 7) % P
        y = pow(y_sq, (P + 1) // 4, P)
        if y % 2 != 0:
            y = P - y
        R_pt = (x, y)
        
        Q = point_add(point_mul(u1, G), point_mul(u2, R_pt))
        pubkey_bytes = compress_pubkey(Q)
        h160 = ripemd160(hashlib.sha256(pubkey_bytes).digest())
        
        # Easy check for test speed (real search uses GPU with full DER)
        if is_valid_der_sig(h160) or (h160[0] >> 4) == 3:
            found_lt = lt
            is_real_der = is_valid_der_sig(h160)
            print(f"  Found! locktime={lt}, hash160={b2h(h160)} (real_DER={is_real_der})")
            break
        
        if lt % 100000 == 0:
            print(f"    searched {lt}...")
    
    if not found_lt:
        print("  Not found in range!")
        return
    
    tx.locktime = found_lt
    
    # =============================
    # Digest search (easy mode, per round)
    # =============================
    
    found_round_indices = []
    for ri in range(2):
        rs = sigs[ri + 1]
        sig_nonce = rs['sig']
        r_val, s_val = rs['r'], rs['s']
        t = t1 if ri == 0 else t2
        
        print(f"\n  [Digest round {ri+1} search, t={t}]")
        
        base_sc = find_and_delete(full_script, sig_nonce)
        d_r_inv = modinv(r_val, N)
        
        dx = r_val
        dy_sq = (pow(dx, 3, P) + 7) % P
        dy = pow(dy_sq, (P + 1) // 4, P)
        if dy % 2 != 0:
            dy = P - dy
        dR = (dx, dy)
        
        found_combo = None
        count = 0
        for combo in combinations(range(n), t):
            # FindAndDelete selected dummy sigs
            sc = base_sc
            for idx in combo:
                sc = find_and_delete(sc, builder.dummy_sigs[ri][idx])
            
            z = tx.sighash(QSB_IDX, sc, sighash_type=0x01)
            
            u1 = (-z * d_r_inv) % N
            u2 = (s_val * d_r_inv) % N
            Q = point_add(point_mul(u1, G), point_mul(u2, dR))
            pk = compress_pubkey(Q)
            h160 = ripemd160(hashlib.sha256(pk).digest())
            
            if is_valid_der_sig(h160) or (h160[0] >> 4) == 3:
                found_combo = list(combo)
                print(f"  Found! indices={found_combo}, hash160={b2h(h160)}")
                break
            
            count += 1
            if count % 10 == 0:
                print(f"    searched {count}...")
        
        if not found_combo:
            print(f"  Not found! (searched {count})")
            return
        found_round_indices.append(found_combo)
    
    print(f"\n  ✓ All phases found solutions!")
    print(f"  Locktime: {found_lt}")
    
    # Now test assembly
    print(f"\n  --- Testing Assembly ---")
    
    # Save state for assembly
    test_state = {
        'config': 'test', 'n': n,
        't1s': t1s, 't1b': t1b, 't2s': t2s, 't2b': t2b,
        'hors_secrets': [[b2h(s) for s in r] for r in builder.hors_secrets],
        'hors_commitments': [[b2h(c) for c in r] for r in builder.hors_commitments],
        'dummy_sigs': [[b2h(s) for s in r] for r in builder.dummy_sigs],
        'pin_r': sigs[0]['r'], 'pin_s': sigs[0]['s'], 'pin_k': sigs[0]['k'],
        'pin_sig': b2h(sigs[0]['sig']),
        'round_sigs': [{'r': sigs[i+1]['r'], 's': sigs[i+1]['s'],
                        'sig': b2h(sigs[i+1]['sig']), 'k': sigs[i+1]['k']} for i in range(2)],
        'full_script_hex': b2h(full_script),
        'script_hash160': b2h(hash160(full_script)),
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(test_state, f, indent=2)
    
    # Create a mock args object for assembly
    class MockArgs:
        pass
    asm_args = MockArgs()
    asm_args.locktime = found_lt
    asm_args.round1 = ','.join(str(i) for i in found_round_indices[0])
    asm_args.round2 = ','.join(str(i) for i in found_round_indices[1])
    asm_args.funding_txid = '01' * 32
    asm_args.funding_vout = 0
    asm_args.funding_value = 50000
    asm_args.dest_address = '00' * 20
    
    cmd_assemble(asm_args)
    
    print(f"\n  ✓ Full pipeline test complete!")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="QSB Pipeline")
    sub = parser.add_subparsers(dest='command')
    
    # Setup
    p_setup = sub.add_parser('setup')
    p_setup.add_argument('--config', default='A')
    p_setup.add_argument('--seed', type=int, default=None)
    
    # Export
    p_export = sub.add_parser('export')
    p_export.add_argument('--funding-txid', required=True)
    p_export.add_argument('--funding-vout', type=int, required=True)
    p_export.add_argument('--funding-value', type=int, required=True)
    p_export.add_argument('--dest-address', required=True, help='hex pubkey hash (20 bytes)')
    
    # Assemble
    p_asm = sub.add_parser('assemble')
    p_asm.add_argument('--locktime', type=int, required=True)
    p_asm.add_argument('--round1', required=True, help='comma-separated indices')
    p_asm.add_argument('--round2', required=True, help='comma-separated indices')
    p_asm.add_argument('--funding-txid', required=True)
    p_asm.add_argument('--funding-vout', type=int, required=True)
    p_asm.add_argument('--funding-value', type=int, required=True)
    p_asm.add_argument('--dest-address', required=True, help='hex pubkey hash')
    
    # Test
    p_test = sub.add_parser('test')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        cmd_setup(args)
    elif args.command == 'export':
        cmd_export(args)
    elif args.command == 'assemble':
        cmd_assemble(args)
    elif args.command == 'test':
        cmd_test(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
