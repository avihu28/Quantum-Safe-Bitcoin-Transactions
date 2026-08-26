#!/usr/bin/env python3
"""
qsb_pipeline.py — End-to-end QSB Pipeline

Phase 1: Setup
  - Generate HORS keys, dummy sigs, build script
  - Create bare script output for funding
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
  python3 qsb_pipeline.py export \
    --funding-txid <txid> --funding-vout 0 --funding-value <sats> \
    --extra-input-txid <txid> --extra-input-vout <n> --extra-input-value <sats> \
    --output-value <sats> --output-address bc1q...
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
    encode_der_sig, is_valid_der_sig, modinv, int_to_der_int,
)
from bitcoin_tx import (
    Transaction, TxIn, TxOut, QSBScriptBuilder,
    push_data, push_number, find_and_delete, serialize_varint,
    OP_0, OP_HASH160, OP_EQUALVERIFY, OP_CHECKSIG,
)

STATE_FILE = "qsb_state.json"


def puzzle_hash(pubkey_bytes, hash_mode='ripemd160', hash_choice=0):
    """Compute the puzzle hash for a compressed pubkey.

    This MUST match exactly what the locking script computes at the puzzle
    step, because the result is interpreted as an ECDSA signature (sig_puzzle)
    whose DER parsing must succeed.

    Script op        → hash we must compute here
    OP_RIPEMD160     → ripemd160(pubkey_bytes)          # single-hash ripemd160 mode
    OP_SHA256        → sha256(pubkey_bytes)             # single-hash sha256 mode
    OP_IF SHA256 ENDIF OP_SHA256 → sha256(pubkey)       # double mode, bit=0
                                 or sha256(sha256(pk))  # double mode, bit=1

    Returns (hash_bytes, is_valid_der, hash_choice_used).
    """
    if hash_mode == 'ripemd160':
        h = ripemd160(pubkey_bytes)
        return h, is_valid_der_sig(h), 0
    elif hash_mode == 'sha256':
        h = hashlib.sha256(pubkey_bytes).digest()
        return h, is_valid_der_sig(h), 0
    elif hash_mode == 'sha256_double':
        # Try SHA-256 first (bit=0 — IF branch skipped)
        h1 = hashlib.sha256(pubkey_bytes).digest()
        if is_valid_der_sig(h1):
            return h1, True, 0
        # Try SHA-256(SHA-256) (bit=1 — IF branch taken)
        h2 = hashlib.sha256(h1).digest()
        if is_valid_der_sig(h2):
            return h2, True, 1
        # Neither worked — return first hash with failure
        return h1, False, 0
    else:
        raise ValueError(f"Unknown hash_mode: {hash_mode}")

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

def _count_pattern(haystack, needle):
    """Count non-overlapping occurrences of `needle` in `haystack`."""
    if not needle:
        return 0
    count = 0
    i = 0
    while i + len(needle) <= len(haystack):
        if haystack[i:i + len(needle)] == needle:
            count += 1
            i += len(needle)
        else:
            i += 1
    return count

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


def p2wpkh_script(pkh_hex):
    """P2WPKH scriptPubKey: OP_0 <20-byte pubkeyhash>"""
    pkh = h2b(pkh_hex)
    if len(pkh) != 20:
        raise ValueError(f"P2WPKH pubkeyhash must be 20 bytes, got {len(pkh)}")
    return bytes([0x00, 0x14]) + pkh


def bech32_decode_pkh(addr):
    """Decode a bech32 (segwit v0) address to its 20-byte pubkeyhash.
    Lightweight, no external deps."""
    CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
    addr = addr.lower()
    if addr.count("1") < 1:
        raise ValueError(f"invalid bech32: {addr}")
    pos = addr.rfind("1")
    hrp = addr[:pos]
    data = addr[pos+1:]
    if hrp not in ("bc", "tb", "bcrt"):
        raise ValueError(f"unexpected hrp: {hrp}")
    decoded = []
    for c in data:
        if c not in CHARSET:
            raise ValueError(f"invalid char in bech32: {c}")
        decoded.append(CHARSET.index(c))
    if len(decoded) < 6:
        raise ValueError("bech32 too short")
    # Skip checksum verification (assumes valid input — appropriate for
    # advanced user-supplied addresses; bitcoin-cli will catch invalid ones).
    payload = decoded[:-6]
    witver = payload[0]
    if witver != 0:
        raise ValueError(f"only segwit v0 supported (got v{witver})")
    # Decode 5-bit groups → 8-bit bytes
    bits = 0
    acc = 0
    out = bytearray()
    for v in payload[1:]:
        acc = (acc << 5) | v
        bits += 5
        if bits >= 8:
            bits -= 8
            out.append((acc >> bits) & 0xff)
    if len(out) != 20:
        raise ValueError(f"unexpected program length {len(out)} (expected 20)")
    return bytes(out)


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
        # Config A: n=150, (8+1b, 7+2b), sha256 single-hash pinning
        #   201/201 ops exact, 9783 bytes, ~2^116 pre-image, ~2^88 collision,
        #   1x grinds, honest work ~2^47. ALL-SHA256 → simplest GPU kernels.
        'A':    {'n': 150, 't1s': 8, 't1b': 1, 't2s': 7, 't2b': 2, 'hash_mode': 'sha256'},
        # Config Ar: same shape but with ripemd160 for pinning (paper's spec).
        #   Same 201/201 ops. ~1 bit more pre-image but needs a RIPEMD160 GPU kernel.
        'Ar':   {'n': 150, 't1s': 8, 't1b': 1, 't2s': 7, 't2b': 2, 'hash_mode': 'ripemd160'},
        # Config S: smaller n, sha256 — kept for fallback but Config A is preferred.
        'S':    {'n': 140, 't1s': 8, 't1b': 1, 't2s': 7, 't2b': 2, 'hash_mode': 'sha256'},
        # Config D: the original failing config (221/201 ops). DO NOT USE.
        'D':    {'n': 130, 't1s': 8, 't1b': 1, 't2s': 8, 't2b': 1, 'hash_mode': 'sha256_double'},
        # Config test: tiny n for end-to-end self-test.
        'test': {'n': 10,  't1s': 2, 't1b': 0, 't2s': 2, 't2b': 0, 'hash_mode': 'sha256'},
    }
    cfg = configs[config]
    n, t1s, t1b, t2s, t2b = cfg['n'], cfg['t1s'], cfg['t1b'], cfg['t2s'], cfg['t2b']
    hash_mode = cfg.get('hash_mode', 'ripemd160')
    t1, t2 = t1s + t1b, t2s + t2b
    
    print(f"  Config: {config}, n={n}, R1=({t1s}+{t1b}), R2=({t2s}+{t2b}), hash={hash_mode}")
    
    if seed:
        import random
        random.seed(seed)
        orig = os.urandom
        os.urandom = lambda n: bytes([random.getrandbits(8) for _ in range(n)])
    
    builder = QSBScriptBuilder(n, t1s, t1b, t2s, t2b, hash_mode=hash_mode)
    builder.generate_keys()
    
    if seed:
        os.urandom = orig
    
    # Generate fixed sig_nonce for each phase
    # These are the ECDSA signatures hardcoded in the script
    # They use known (r, s) values — the search finds a locktime/subset
    # that makes SHA256(recovered_pubkey) valid DER
    
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
    
    static_ops = QSBScriptBuilder.count_opcodes(full_script)
    runtime_ops, multisig_ns = QSBScriptBuilder.count_opcodes_runtime(full_script)
    print(f"  Script size: {len(full_script)} bytes (limit 10000)")
    print(f"  Opcodes: static={static_ops}, runtime={runtime_ops} / 201 (multisig_Ns={multisig_ns})")

    # HARD FAIL: do not let a consensus-invalid script pass setup.
    if runtime_ops > 201:
        raise RuntimeError(
            f"Script exceeds MAX_OPS_PER_SCRIPT: {runtime_ops} > 201. "
            "This would fail consensus (SCRIPT_ERR_OP_COUNT). "
            "Change config (see README) before proceeding.")
    if len(full_script) > 10000:
        raise RuntimeError(
            f"Script exceeds MAX_SCRIPT_SIZE: {len(full_script)} > 10000 bytes.")
    
    # Save state
    state = {
        'config': config,
        'hash_mode': hash_mode,
        'n': n, 't1s': t1s, 't1b': t1b, 't2s': t2s, 't2b': t2b,
        'hors_secrets': [[b2h(s) for s in r] for r in builder.hors_secrets],
        'hors_commitments': [[b2h(c) for c in r] for r in builder.hors_commitments],
        'dummy_sigs': [[b2h(s) for s in r] for r in builder.dummy_sigs],
        'pin_r': pin_r, 'pin_s': pin_s, 'pin_k': pin_k,
        'pin_sig': b2h(pin_sig),
        'round_sigs': [{'r': rs['r'], 's': rs['s'], 'sig': rs['sig'], 'k': rs['k']} for rs in round_sigs],
        'full_script_hex': b2h(full_script),
    }
    
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    
    # Also save the bare scriptPubKey for funding
    spk_file = "qsb_scriptpubkey.hex"
    with open(spk_file, 'w') as f:
        f.write(b2h(full_script))
    
    print(f"\n  State saved to {STATE_FILE}")
    print(f"  Bare scriptPubKey saved to {spk_file} ({len(full_script)} bytes)")
    print(f"\n  NOTE: This is a bare script output (not P2SH). To fund it:")
    print(f"  1. Create a raw tx with this scriptPubKey as an output:")
    print(f"     bitcoin-cli createrawtransaction '[{{...}}]' '[{{\"data\":\"...\"}}]'")
    print(f"     (Or use the fund command below)")
    print(f"  2. Then run:")
    print(f"  python3 qsb_pipeline.py export \\")
    print(f"      --funding-txid <txid> --funding-vout 0 --funding-value <sats> \\")
    print(f"      --extra-input-txid <txid> --extra-input-vout <n> --extra-input-value <sats> \\")
    print(f"      --output-value <sats> --output-address bc1q...")


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
    
    # ========================================================================
    # Spending tx structure: 2 inputs + 1 output
    #   - input[0]: extra UTXO that user controls (e.g., the change output of
    #     the funding tx). User signs this separately (cmd_assemble leaves it
    #     unsigned for the user to sign with their wallet).
    #   - input[1]: THE QSB OUTPUT (the funding tx's vout=funding_vout)
    #   - output[0]: send (input0_value + funding_value - fee) to user_address
    #
    # Why this shape:
    #   * Consensus requires ≥1 output (bad-txns-vout-empty otherwise)
    #   * QSB at input index 1 keeps things simple — kernel computes legacy
    #     SIGHASH_ALL preimage with input[0]'s scriptSig zeroed (per legacy
    #     sighash semantics), so user signing input[0] later doesn't affect
    #     the QSB sighash
    # ========================================================================
    funding_txid = h2b(args.funding_txid)[::-1]  # reverse for internal byte order
    funding_vout = args.funding_vout
    funding_value = args.funding_value
    
    extra_txid = h2b(args.extra_input_txid)[::-1]
    extra_vout = args.extra_input_vout
    extra_value = args.extra_input_value
    extra_seq = args.extra_input_sequence
    
    output_value = args.output_value
    # Decode destination (P2WPKH preferred — segwit, ~22 byte scriptPubKey)
    if args.output_address.startswith(("bc1", "tb1", "bcrt1")):
        out_pkh = bech32_decode_pkh(args.output_address)
        output_script = p2wpkh_script(out_pkh.hex())
    else:
        # Treat as 20-byte hex pubkeyhash → P2PKH
        output_script = p2pkh_script(args.output_address)
    
    print(f"  Spending-tx structure:")
    print(f"    input[0]: extra UTXO {args.extra_input_txid}:{extra_vout} ({extra_value} sats)")
    print(f"    input[1]: QSB         {args.funding_txid}:{funding_vout} ({funding_value} sats)")
    print(f"    output[0]: {output_value} sats → {args.output_address}")
    fee = (extra_value + funding_value) - output_value
    print(f"    fee: {fee} sats")
    if fee < 0:
        print(f"    ✗ ERROR: output value exceeds inputs! Reduce --output-value.")
        return
    if fee > 100_000:
        print(f"    ⚠ WARN: fee is huge ({fee} sats = ${fee*1e-8 * 100_000:.2f} @ $100k/BTC).")
    
    # The QSB input is at index 1.
    QSB_INPUT_INDEX = 1
    
    # Build the spending transaction template (used for sighash computation)
    tx = Transaction(version=args.version, locktime=args.locktime)
    tx.add_input(TxIn(extra_txid, extra_vout, b'', extra_seq))           # input[0]
    tx.add_input(TxIn(funding_txid, funding_vout, b'', args.sequence))    # input[1] — QSB
    tx.add_output(TxOut(output_value, output_script))
    
    # Pre-built fragments used in BOTH pinning and digest preimages.
    # 
    # Legacy SIGHASH_ALL sighash for input[1] (the QSB) substitutes input[0]'s
    # scriptSig with empty (1 byte 0x00). So tx_in[0] in the preimage is:
    #   txid(32) + vout(4) + 0x00(empty_script_len) + sequence(4) = 41 bytes
    txin0_for_preimage = (
        extra_txid +
        struct.pack('<I', extra_vout) +
        b'\x00' +
        struct.pack('<I', extra_seq)
    )
    # tx_in[1] up to (but not including) the scriptcode body
    txin1_prefix = (
        funding_txid +
        struct.pack('<I', funding_vout)
    )
    # The single output's serialized form (8 byte value + varint script_len + script)
    serialized_output = (
        struct.pack('<q', output_value) +
        serialize_varint(len(output_script)) +
        output_script
    )
    
    # ============================================================
    # Export pinning params
    # ============================================================
    
    pin_r = state['pin_r']
    pin_s = state['pin_s']
    pin_sig = h2b(state['pin_sig'])

    # For pinning, sighash_type = SIGHASH_ALL (0x01)
    # scriptCode = full_script with FindAndDelete(pin_sig)
    pin_script_code = find_and_delete(full_script, pin_sig)

    # Layout:
    #   prefix:
    #     version(4) | n_in_varint=2 | tx_in[0]_full(41) | tx_in[1].txid+vout(36) |
    #     scriptcode_len_varint | scriptcode
    #   suffix:
    #     tx_in[1].sequence(4) | n_out_varint=1 | serialized_output | locktime(4) | sighash(4)
    pin_prefix = struct.pack('<I', tx.version)
    pin_prefix += serialize_varint(2)                  # 2 inputs
    pin_prefix += txin0_for_preimage                    # input[0] (empty scriptSig)
    pin_prefix += txin1_prefix                          # input[1] up to scriptcode
    pin_prefix += serialize_varint(len(pin_script_code))
    pin_prefix += pin_script_code

    # The kernel splices sequence (input[1].sequence) and locktime, both vary.

    # Compute r_inv, neg_r_inv, u2*R for EC recovery
    r_inv = modinv(pin_r, N)
    neg_r_inv = (-r_inv) % N
    u2 = (pin_s * r_inv) % N

    # Recover R point from r (even y, recid=0)
    x = pin_r
    y_sq = (pow(x, 3, P) + 7) % P
    y = pow(y_sq, (P + 1) // 4, P)
    if y % 2 != 0:
        y = P - y
    R_point = (x, y)
    u2R = point_mul(u2, R_point)

    # Midstate covers full 64-byte blocks of pin_prefix; trailing bytes become
    # the FIRST bytes of the kernel's combined-suffix buffer.
    full_blocks = len(pin_prefix) // 64
    prefix_remainder_pin = pin_prefix[full_blocks * 64:]

    # Combined-suffix layout:
    #   [prefix_remainder] [QSB_sequence (4)] [out_count varint=1] [out (8 + varint + script)] [locktime (4)] [sighash (4)]
    combined_suffix = bytearray(prefix_remainder_pin)
    seq_offset_in_suffix = len(combined_suffix)
    combined_suffix += struct.pack('<I', args.sequence)        # QSB input's sequence (kernel varies)
    combined_suffix += serialize_varint(1)                      # 1 output
    combined_suffix += serialized_output                        # the output (fixed)
    lt_offset_in_suffix = len(combined_suffix)
    combined_suffix += struct.pack('<I', args.locktime)         # locktime (kernel varies)
    combined_suffix += struct.pack('<I', 0x01)                  # SIGHASH_ALL
    combined_suffix = bytes(combined_suffix)

    total_preimage_len = full_blocks * 64 + len(combined_suffix)

    # JSON export
    params = {
        'type': 'pinning',
        'hash_mode': state.get('hash_mode', 'sha256'),
        'pin_prefix': b2h(pin_prefix),
        'pin_prefix_len': len(pin_prefix),
        'combined_suffix': b2h(combined_suffix),
        'combined_suffix_len': len(combined_suffix),
        'seq_offset': seq_offset_in_suffix,
        'lt_offset': lt_offset_in_suffix,
        'total_preimage_len': total_preimage_len,
        'midstate_blocks': full_blocks,
        'prefix_remainder_len': len(prefix_remainder_pin),
        # Spending-tx structure (saved so cmd_assemble can reconstruct identically)
        'spending_tx': {
            'version': args.version,
            'extra_input': {
                'txid': args.extra_input_txid,
                'vout': extra_vout,
                'value': extra_value,
                'sequence': extra_seq,
            },
            'qsb_input': {
                'txid': args.funding_txid,
                'vout': funding_vout,
                'value': funding_value,
                'sequence': args.sequence,
            },
            'output': {
                'value': output_value,
                'address': args.output_address,
                'script_pubkey': output_script.hex(),
            },
            'locktime': args.locktime,
            'sighash_type': 0x01,
            'qsb_input_index': QSB_INPUT_INDEX,
        },
        # Backward-compat alias for emulator / verify_hit (they read 'tx_prefix')
        'tx_prefix': b2h(pin_prefix),
        'tx_prefix_len': len(pin_prefix),
        'pin_r': pin_r,
        'pin_s': pin_s,
        'neg_r_inv': b2h(le_bytes(neg_r_inv)),
        'u2r_x': b2h(le_bytes(u2R[0])),
        'u2r_y': b2h(le_bytes(u2R[1])),
    }

    with open('gpu_pinning_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    # Binary export for GPU — same layout as before, just with the new
    # combined_suffix that includes the output.
    with open('pinning.bin', 'wb') as f:
        midstate = compute_sha256_midstate(pin_prefix, full_blocks)
        for v in midstate:
            f.write(struct.pack('>I', v))
        f.write(struct.pack('<I', len(combined_suffix)))
        f.write(combined_suffix)
        f.write(struct.pack('<I', total_preimage_len))
        f.write(struct.pack('<I', seq_offset_in_suffix))
        f.write(struct.pack('<I', lt_offset_in_suffix))
        f.write(le_bytes(neg_r_inv))
        f.write(le_bytes(u2R[0]))
        f.write(le_bytes(u2R[1]))

    print(f"  Pinning: pin_prefix={len(pin_prefix)} bytes, midstate={full_blocks} blocks, "
          f"combined_suffix={len(combined_suffix)} "
          f"(remainder={len(prefix_remainder_pin)}, seq@{seq_offset_in_suffix}, "
          f"lt@{lt_offset_in_suffix})")
    print(f"  Saved gpu_pinning_params.json + pinning.bin")
    
    # ============================================================
    # Export digest params (per round)
    # ============================================================
    
    # Reconstruct the builder so we can ask it for exact section sizes
    builder = QSBScriptBuilder(
        n, state['t1s'], state['t1b'], state['t2s'], state['t2b'],
        hash_mode=state.get('hash_mode', 'sha256'))
    # Populate HORS commitments and dummy sigs from saved state
    builder.hors_commitments = [
        [h2b(c) for c in state['hors_commitments'][r]] for r in range(2)
    ]
    builder.dummy_sigs = [
        [h2b(s) for s in state['dummy_sigs'][r]] for r in range(2)
    ]
    # Rebuild the subscripts to determine offsets in the canonical layout
    pin_sub = builder.build_pinning_script(pin_sig)
    r1_sub = builder.build_round_script(0, h2b(state['round_sigs'][0]['sig']))
    r2_sub = builder.build_round_script(1, h2b(state['round_sigs'][1]['sig']))
    # Sanity: concatenation must equal full_script
    assert pin_sub + r1_sub + r2_sub == full_script, \
        "Builder subscripts do not reassemble to full_script — layout drift"

    round_offsets = [len(pin_sub), len(pin_sub) + len(r1_sub)]
    round_subscripts = [r1_sub, r2_sub]

    for ri in range(2):
        rs = state['round_sigs'][ri]
        r_val, s_val = rs['r'], rs['s']
        sig_nonce = h2b(rs['sig'])

        # base_script_code = full_script with this round's sig_nonce removed via F&D.
        # F&D is applied to the WHOLE script; the sig_nonce push lives somewhere inside
        # this round's subscript (specifically, just before the signed selections).
        # We compute the offsets of HORS and dummies in base_script_code directly.
        base_script_code = find_and_delete(full_script, sig_nonce)

        # === CORRECT LAYOUT COMPUTATION ===
        # In full_script (before F&D):
        #   [pin_sub] [r1_sub] [r2_sub]
        # Round R's HORS section begins at offset `round_offsets[R]`, size n*21.
        # Round R's dummy section begins at `round_offsets[R] + n*21`, size n*10.
        #
        # After F&D of sig_nonce (located inside this round, AFTER dummies), the
        # offsets for HORS and dummies are UNCHANGED — sig_nonce appears later in
        # the script, so removing it doesn't shift the earlier bytes.
        #
        # Caveat: if sig_nonce happens to appear elsewhere (in pin_sub, in another
        # round, or inside a HORS hash or dummy sig by coincidence), F&D would
        # remove those too and shift offsets. We assert below that only a single
        # occurrence existed — the one we expect.
        full_occurrences = _count_pattern(full_script, push_data(sig_nonce))
        if full_occurrences != 1:
            raise RuntimeError(
                f"sig_nonce for round {ri+1} appears {full_occurrences} times in "
                f"full_script — FindAndDelete would remove all of them and corrupt "
                f"the byte layout. Regenerate sig_nonce (change the seed).")

        pre_hors_len = round_offsets[ri]    # everything before THIS round's HORS
        hors_len = n * 21
        dummies_len = n * 10
        # After-dummies-in-this-round + subsequent rounds (with sig_nonce already
        # removed via F&D).
        post_dummies_start = pre_hors_len + hors_len + dummies_len
        tail_section = base_script_code[post_dummies_start:]
        # The pre-HORS bytes from base_script_code (unchanged by F&D since sig_nonce
        # is AFTER this point in the script)
        pre_hors_section = base_script_code[:pre_hors_len]
        # HORS bytes (unchanged by F&D)
        hors_section = base_script_code[pre_hors_len:pre_hors_len + hors_len]

        # Sanity: reassembled scriptcode (with no dummies removed) should equal
        # base_script_code.
        dummies_section = base_script_code[pre_hors_len + hors_len:post_dummies_start]
        assert pre_hors_section + hors_section + dummies_section + tail_section == base_script_code

        # removed_per_subset = number of dummy sigs the GPU subtracts per candidate
        removed_per_subset = (t1 if ri == 0 else t2)
        scriptcode_len_after_fad = len(base_script_code) - removed_per_subset * 10

        # Spending-tx prefix for THIS round (same shape as pinning).
        # Layout: version | n_in=2 | tx_in[0]_full(empty script) | tx_in[1].txid+vout |
        #         scriptcode_len_varint | scriptcode_body...
        d_tx_prefix = struct.pack('<I', tx.version)
        d_tx_prefix += serialize_varint(2)
        d_tx_prefix += txin0_for_preimage
        d_tx_prefix += txin1_prefix
        d_tx_prefix += serialize_varint(scriptcode_len_after_fad)
        # scriptCode body goes here (built by GPU per subset)
        
        # Suffix: tx_in[1].sequence | n_out=1 | output | locktime | sighash
        d_tx_suffix = struct.pack('<I', args.sequence)         # QSB sequence (kernel varies)
        d_tx_suffix += serialize_varint(1)                      # 1 output
        d_tx_suffix += serialized_output                        # output (fixed)
        d_tx_suffix += struct.pack('<I', args.locktime)         # locktime (kernel varies)
        d_tx_suffix += struct.pack('<I', 0x01)                  # SIGHASH_ALL
        
        total_d_preimage = len(d_tx_prefix) + scriptcode_len_after_fad + len(d_tx_suffix)
        
        # Midstate: covers d_tx_prefix + pre_hors_section + hors_section.
        # Both pre_hors_section (pin script + earlier rounds' scripts) and hors_section
        # are FIXED across subsets, so they can be folded into the SHA-256 midstate.
        # The GPU doesn't need to see them — it just continues SHA-256 from the midstate.
        fixed_prefix = d_tx_prefix + pre_hors_section + hors_section
        fp_full_blocks = len(fixed_prefix) // 64

        # Remainder = trailing bytes of fixed_prefix that didn't fit in a full 64-byte
        # block (0..63 bytes). GPU processes these, then the filtered dummies, then
        # tail, then tx_suffix.
        prefix_remainder = fixed_prefix[fp_full_blocks * 64:]

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
            'hash_mode': state.get('hash_mode', 'sha256'),
            # Layout (all fixed; GPU only receives midstate covering these):
            'pre_hors_section': b2h(pre_hors_section),
            'pre_hors_section_len': len(pre_hors_section),
            'hors_section': b2h(hors_section),
            'hors_section_len': hors_len,
            'dummies_section_len': dummies_len,
            # Individual dummy sig pushes (for GPU's per-subset reconstruction):
            'dummy_sigs': [b2h(h2b(state['dummy_sigs'][ri][i])) for i in range(n)],
            'dummy_sig_pushes': dummy_sigs_in_order,
            # Bytes AFTER this round's dummy section (post-dummy of this round + next round):
            'tail_section': b2h(tail_section),
            'tail_section_len': len(tail_section),
            # Preimage boundaries:
            'tx_prefix': b2h(d_tx_prefix),
            'tx_prefix_len': len(d_tx_prefix),
            'tx_suffix': b2h(d_tx_suffix),
            'tx_suffix_len': len(d_tx_suffix),
            'fixed_prefix': b2h(fixed_prefix),
            'fixed_prefix_len': len(fixed_prefix),
            'midstate_blocks': fp_full_blocks,
            'prefix_remainder': b2h(prefix_remainder),
            'prefix_remainder_len': len(prefix_remainder),
            'scriptcode_len': scriptcode_len_after_fad,
            'total_preimage_len': total_d_preimage,
            'sig_r': r_val,
            'sig_s': s_val,
            'neg_r_inv': b2h(le_bytes(d_neg_r_inv)),
            'u2r_x': b2h(le_bytes(d_u2R[0])),
            'u2r_y': b2h(le_bytes(d_u2R[1])),
            # Spending-tx structure (matches gpu_pinning_params.json — saved
            # so cmd_assemble can reconstruct the exact 2-in/1-out tx)
            'spending_tx': {
                'version': args.version,
                'extra_input': {
                    'txid': args.extra_input_txid,
                    'vout': extra_vout,
                    'value': extra_value,
                    'sequence': extra_seq,
                },
                'qsb_input': {
                    'txid': args.funding_txid,
                    'vout': funding_vout,
                    'value': funding_value,
                    'sequence': args.sequence,
                },
                'output': {
                    'value': output_value,
                    'address': args.output_address,
                    'script_pubkey': output_script.hex(),
                },
                'locktime': args.locktime,
                'sighash_type': 0x01,
                'qsb_input_index': QSB_INPUT_INDEX,
            },
        }

        fname = f'gpu_digest_r{ri+1}_params.json'
        with open(fname, 'w') as f:
            json.dump(digest_params, f, indent=2)

        # Binary export for GPU — new format includes prefix_remainder
        bname = f'digest_r{ri+1}.bin'
        with open(bname, 'wb') as f:
            # Header
            f.write(struct.pack('<I', n))
            f.write(struct.pack('<I', removed_per_subset))
            f.write(struct.pack('<I', total_d_preimage))
            f.write(struct.pack('<I', len(tail_section)))
            f.write(struct.pack('<I', len(d_tx_suffix)))
            f.write(struct.pack('<I', len(prefix_remainder)))
            # Midstate (8 × uint32 BE)
            mid = compute_sha256_midstate(fixed_prefix, fp_full_blocks)
            for v in mid:
                f.write(struct.pack('>I', v))
            # Remainder bytes (may be 0..63 bytes, variable)
            f.write(prefix_remainder)
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

        print(f"  Round {ri+1}: scriptCode={scriptcode_len_after_fad} bytes, "
              f"midstate={fp_full_blocks} blocks, remainder={len(prefix_remainder)} bytes")
        print(f"  Saved {fname} + {bname}")
    
    print(f"\n  Upload these JSON files + GPU code to vast.ai and run search.")


# ============================================================
# Phase 4: Assemble transaction
# ============================================================

def parse_der(sig_bytes):
    """Parse DER-encoded signature into (r, s) integers.
    Handles optional sighash suffix byte.
    Returns (None, None) if not valid DER."""
    if len(sig_bytes) < 8:
        return None, None
    d = sig_bytes
    try:
        if d[0] != 0x30: return None, None
        tl = d[1]
        # Check if there's a sighash byte after DER
        if tl + 2 == len(d) - 1:
            d = d[:-1]  # strip sighash
        idx = 2
        if d[idx] != 0x02: return None, None
        idx += 1
        rl = d[idx]; idx += 1
        r = int.from_bytes(d[idx:idx+rl], 'big')
        idx += rl
        if d[idx] != 0x02: return None, None
        idx += 1
        sl = d[idx]; idx += 1
        s = int.from_bytes(d[idx:idx+sl], 'big')
        return r, s
    except (IndexError, ValueError):
        return None, None


def cmd_assemble(args):
    print("╔══════════════════════════════════════╗")
    print("║  QSB Pipeline — Phase 4: Assemble    ║")
    print("╚══════════════════════════════════════╝")
    
    with open(STATE_FILE) as f:
        state = json.load(f)
    
    n = state['n']
    t1 = state['t1s'] + state['t1b']
    t2 = state['t2s'] + state['t2b']
    hash_mode = state.get('hash_mode', 'sha256')

    # Curve constants — used in multiple places below for r/r+N fallback
    from secp256k1 import N as _CURVE_N, P as _CURVE_P
    
    locktime = args.locktime
    sequence = args.sequence
    r1_indices = sorted([int(x) for x in args.round1.split(',')])
    r2_indices = sorted([int(x) for x in args.round2.split(',')])
    
    assert len(r1_indices) == t1, f"Expected {t1} round1 indices, got {len(r1_indices)}"
    assert len(r2_indices) == t2, f"Expected {t2} round2 indices, got {len(r2_indices)}"
    
    full_script = h2b(state['full_script_hex'])

    # Corrected witness index encoding (single-hash modes): the selection loop
    # reads model-derived dummy-depths, NOT raw pool indices. Rebuild a builder
    # holding the pool data and compute the indices exactly as build_full_script
    # positioned them. sha256_double keeps the legacy raw-index behaviour.
    witness_indices = None
    if hash_mode in ('ripemd160', 'sha256'):
        _wb = QSBScriptBuilder(n, state['t1s'], state['t1b'], state['t2s'],
                               state['t2b'], hash_mode=hash_mode)
        _wb.hors_commitments = [[h2b(c) for c in state['hors_commitments'][r]] for r in range(2)]
        _wb.dummy_sigs = [[h2b(s) for s in state['dummy_sigs'][r]] for r in range(2)]
        witness_indices = _wb.compute_witness_indices({0: r1_indices, 1: r2_indices})

    print(f"  Locktime: {locktime}")
    print(f"  Sequence: {sequence} (0x{sequence:08X})")
    print(f"  Round 1 indices: {r1_indices}")
    print(f"  Round 2 indices: {r2_indices}")
    
    # Build the spending transaction: 2 inputs + 1 output, QSB at index 1.
    # input[0] is the user's extra UTXO — we leave its scriptSig empty here.
    # The user signs input[0] separately (e.g., via bitcoin-cli
    # signrawtransactionwithwallet). Legacy SIGHASH_ALL on input[1] zeroes
    # input[0]'s scriptSig in the preimage, so signing input[0] later does
    # not affect the QSB sighash.
    funding_txid = h2b(args.funding_txid)[::-1]
    extra_txid = h2b(args.extra_input_txid)[::-1]
    
    if args.output_address.startswith(("bc1", "tb1", "bcrt1")):
        out_pkh = bech32_decode_pkh(args.output_address)
        output_script = p2wpkh_script(out_pkh.hex())
    else:
        output_script = p2pkh_script(args.output_address)
    
    tx = Transaction(version=args.version, locktime=locktime)
    tx.add_input(TxIn(extra_txid, args.extra_input_vout, b'', args.extra_input_sequence))
    tx.add_input(TxIn(funding_txid, args.funding_vout, b'', sequence))
    tx.add_output(TxOut(args.output_value, output_script))
    
    QSB_INPUT_INDEX = 1
    
    print(f"  Spending tx structure:")
    print(f"    input[0]: extra UTXO {args.extra_input_txid}:{args.extra_input_vout}")
    print(f"              (UNSIGNED — user must sign this)")
    print(f"    input[1]: QSB         {args.funding_txid}:{args.funding_vout}")
    print(f"    output[0]: {args.output_value} sats → {args.output_address}")
    
    # Step 1: Pinning — recover key_nonce
    print("\n  [1] Pinning: recover key_nonce")
    
    pin_r = state['pin_r']
    pin_s = state['pin_s']
    pin_sig = h2b(state['pin_sig'])
    
    pin_sc = find_and_delete(full_script, pin_sig)
    z_pin = tx.sighash(QSB_INPUT_INDEX, pin_sc, sighash_type=0x01)
    
    key_nonce_pin = None
    sig_puzzle_pin = None
    # Try BOTH r values (r and r+N if r+N < P) and BOTH recovery flags.
    # For each candidate, use puzzle_hash() so we ONLY accept hashes the script
    # can actually verify (single SHA-256 for Config A; the script never
    # computes h2, so an h2-valid sig would be a fake hit).
    from secp256k1 import N as _CURVE_N, P as _CURVE_P
    r_tries = [pin_r] + ([pin_r + _CURVE_N] if pin_r + _CURVE_N < _CURVE_P else [])
    for r_try in r_tries:
        for flag in [0, 1]:
            pt = ecdsa_recover(r_try, pin_s, z_pin, flag)
            if not pt:
                continue
            kn = compress_pubkey(pt)
            h, valid, hc = puzzle_hash(kn, hash_mode)
            if valid:
                key_nonce_pin = kn
                sig_puzzle_pin = h
                hc_label = (' (sha256²)' if hash_mode == 'sha256_double' and hc == 1
                            else f' ({hash_mode})')
                print(f"    key_nonce: {b2h(kn)[:16]}... "
                      f"(flag={flag}, r{'+N' if r_try != pin_r else ''}{hc_label})")
                break
        if key_nonce_pin is not None:
            break

    if key_nonce_pin is None:
        print("    ERROR: could not recover pinning key_nonce!")
        return
    
    # Step 2: Pinning — recover key_puzzle
    print("\n  [2] Pinning: recover key_puzzle")
    
    sp_r, sp_s = parse_der(sig_puzzle_pin)
    if sp_r is None:
        print("    ERROR: sig_puzzle not valid DER!")
        return
    
    sp_sighash_type = sig_puzzle_pin[-1]
    print(f"    sig_puzzle sighash_type: 0x{sp_sighash_type:02x}")
    
    puzzle_sc = find_and_delete(full_script, sig_puzzle_pin)
    z_puzzle_pin = tx.sighash(QSB_INPUT_INDEX, puzzle_sc, sighash_type=sp_sighash_type)
    
    key_puzzle_pin = None
    for flag in [0, 1]:
        pt = ecdsa_recover(sp_r, sp_s, z_puzzle_pin, flag)
        if pt:
            key_puzzle_pin = compress_pubkey(pt)
            print(f"    key_puzzle: {b2h(key_puzzle_pin)[:16]}... (flag={flag}, r as parsed)")
            break
    if key_puzzle_pin is None:
        from secp256k1 import N as _CURVE_N, P as _CURVE_P
        if sp_r + _CURVE_N < _CURVE_P:
            for flag in [0, 1]:
                pt = ecdsa_recover(sp_r + _CURVE_N, sp_s, z_puzzle_pin, flag)
                if pt:
                    key_puzzle_pin = compress_pubkey(pt)
                    print(f"    key_puzzle: {b2h(key_puzzle_pin)[:16]}... (flag={flag}, r+N)")
                    break
    
    if key_puzzle_pin is None:
        print("    ERROR: could not recover pinning key_puzzle!")
        return
    
    # Steps 3-6: Digest rounds
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
        
        sc = find_and_delete(full_script, sig_nonce)
        for idx in indices:
            dummy_sig = h2b(state['dummy_sigs'][ri][idx])
            sc = find_and_delete(sc, dummy_sig)
        
        z_round = tx.sighash(QSB_INPUT_INDEX, sc, sighash_type=0x01)
        
        key_nonce_round = None
        sig_puzzle_round = None
        # Try BOTH r values (r and r+N) — the GPU kernel does this, so we must too
        r_round_tries = [r_val] + ([r_val + _CURVE_N]
                                    if r_val + _CURVE_N < _CURVE_P else [])
        for r_try in r_round_tries:
            for flag in [0, 1]:
                pt = ecdsa_recover(r_try, s_val, z_round, flag)
                if pt:
                    kn = compress_pubkey(pt)
                    sp, real_der, hc = puzzle_hash(kn, hash_mode)
                    if real_der:
                        key_nonce_round = kn
                        sig_puzzle_round = sp
                        print(f"    key_nonce: {b2h(kn)[:16]}... "
                              f"(flag={flag}, r{'+N' if r_try != r_val else ''}, hc={hc})")
                        break
            if key_nonce_round is not None:
                break

        if key_nonce_round is None:
            print(f"    ERROR: round {ri+1} key_nonce recovery failed!")
            return
        
        print(f"\n  [{4+ri*2}] Round {ri+1}: recover key_puzzle")
        
        sp_r2, sp_s2 = parse_der(sig_puzzle_round)
        if sp_r2 is None:
            print(f"    ERROR: round {ri+1} sig_puzzle not valid DER!")
            return
        
        sp_ht = sig_puzzle_round[-1]
        print(f"    sig_puzzle sighash_type: 0x{sp_ht:02x}")
        puzzle_sc2 = find_and_delete(full_script, sig_puzzle_round)
        z_puzzle_round = tx.sighash(QSB_INPUT_INDEX, puzzle_sc2, sighash_type=sp_ht)
        key_puzzle_round = None
        # Try parsed r first
        for flag in [0, 1]:
            pt = ecdsa_recover(sp_r2, sp_s2, z_puzzle_round, flag)
            if pt:
                key_puzzle_round = compress_pubkey(pt)
                print(f"    key_puzzle: {b2h(key_puzzle_round)[:16]}... (flag={flag}, r as parsed)")
                break
        # If r isn't on curve, try r+N as fallback
        if key_puzzle_round is None:
            from secp256k1 import N as _CURVE_N, P as _CURVE_P
            if sp_r2 + _CURVE_N < _CURVE_P:
                for flag in [0, 1]:
                    pt = ecdsa_recover(sp_r2 + _CURVE_N, sp_s2, z_puzzle_round, flag)
                    if pt:
                        key_puzzle_round = compress_pubkey(pt)
                        print(f"    key_puzzle: {b2h(key_puzzle_round)[:16]}... (flag={flag}, r+N)")
                        break
        if key_puzzle_round is None:
            print(f"    ERROR: round {ri+1} key_puzzle recovery failed!")
            return
        
        # Recover dummy pubkeys via the SIGHASH_SINGLE bug. Bitcoin Core's bug
        # value is 2**248 (uint256::ONE's little-endian bytes read big-endian by
        # secp256k1), NOT the integer 1 — must match or the dummies fail consensus.
        SIGHASH_SINGLE_BUG_Z = 1 << 248
        dummy_pubkeys = []
        for idx in indices:
            ds_bytes = h2b(state['dummy_sigs'][ri][idx])
            dr, ds_val = parse_der(ds_bytes)
            if dr is None:
                print(f"    ERROR: dummy sig {idx} not valid DER!")
                return
            recovered = None
            for flag in [0, 1]:
                pt = ecdsa_recover(dr, ds_val, SIGHASH_SINGLE_BUG_Z, flag)
                if pt:
                    recovered = compress_pubkey(pt)
                    break
            if recovered is None:
                from secp256k1 import N as _CURVE_N, P as _CURVE_P
                if dr + _CURVE_N < _CURVE_P:
                    for flag in [0, 1]:
                        pt = ecdsa_recover(dr + _CURVE_N, ds_val, SIGHASH_SINGLE_BUG_Z, flag)
                        if pt:
                            recovered = compress_pubkey(pt)
                            break
            if recovered is None:
                print(f"    ERROR: failed to recover dummy pubkey for idx {idx}!")
                return
            dummy_pubkeys.append(recovered)
        
        signed_indices = indices[:ts]
        preimages = [h2b(state['hors_secrets'][ri][i]) for i in signed_indices]
        
        round_results.append({
            'key_nonce': key_nonce_round,
            'key_puzzle': key_puzzle_round,
            'sig_puzzle': sig_puzzle_round,
            'dummy_pubkeys': dummy_pubkeys,
            'preimages': preimages,
            'subset': indices,
        })
        
        print(f"    dummy_pubkeys: {len(dummy_pubkeys)}")
        print(f"    preimages: {len(preimages)}")
    
    # Build witness
    print(f"\n  [7] Building witness...")
    
    witness = b''
    for rd in [1, 0]:
        rr = round_results[rd]
        witness += push_data(rr['key_puzzle'])
        witness += push_data(rr['key_nonce'])
        for pub in reversed(rr['dummy_pubkeys']):
            witness += push_data(pub)
        for pre in reversed(rr['preimages']):
            witness += push_data(pre)
        if witness_indices is not None:
            # model-derived index numbers (NOT raw pool indices); pushed so that
            # selection 0's index ends on top, matching _seed_witness order.
            ivs = witness_indices[rd]
            for j in range(len(ivs) - 1, -1, -1):
                witness += push_number(ivs[j])
        else:
            for idx in reversed(rr['subset']):
                witness += push_number(idx)
    
    witness += push_data(key_puzzle_pin)
    witness += push_data(key_nonce_pin)
    
    script_sig = witness
    
    print(f"    Witness: {len(witness)} bytes")
    print(f"    ScriptSig: {len(script_sig)} bytes")
    
    tx.inputs[QSB_INPUT_INDEX].script_sig = script_sig
    
    raw_tx = tx.serialize()
    print(f"\n  [8] Final transaction: {len(raw_tx)} bytes")
    print(f"    Raw hex: {b2h(raw_tx)[:80]}...")
    
    with open('qsb_raw_tx.hex', 'w') as f:
        f.write(b2h(raw_tx))
    print(f"    Saved to qsb_raw_tx.hex")
    
    solution = {
        'locktime': locktime,
        'sequence': sequence,
        'pin_key_nonce': b2h(key_nonce_pin),
        'pin_key_puzzle': b2h(key_puzzle_pin),
        'rounds': [{
            'key_nonce': b2h(rr['key_nonce']),
            'key_puzzle': b2h(rr['key_puzzle']),
            'subset': rr['subset'],
            'dummy_pubkeys': [b2h(p) for p in rr['dummy_pubkeys']],
            'preimages': [b2h(p) for p in rr['preimages']],
        } for rr in round_results],
    }
    with open('qsb_solution.json', 'w') as f:
        json.dump(solution, f, indent=2)
    print(f"    Solution saved to qsb_solution.json")
    
    print(f"\n  ✓ Spending transaction assembled (input[0] UNSIGNED)")
    print(f"")
    print(f"  Next step: sign input[0] with your wallet, then broadcast.")
    print(f"")
    print(f"  Option A — bitcoin-cli (loaded wallet has the privkey for the extra UTXO):")
    print(f"    bitcoin-cli signrawtransactionwithwallet $(cat qsb_raw_tx.hex) \\")
    print(f"      '[{{\"txid\":\"{args.extra_input_txid}\",\"vout\":{args.extra_input_vout},\\")
    print(f"        \"scriptPubKey\":\"<scriptPubKey of extra UTXO, hex>\",\\")
    print(f"        \"amount\":{args.extra_input_value/1e8:.8f}}}]'")
    print(f"")
    print(f"  Option B — sparrow / electrum / cold wallet:")
    print(f"    Import qsb_raw_tx.hex as a partially-signed tx, sign input 0, export.")
    print(f"")
    print(f"  Then broadcast via Slipstream (handles non-standard txs):")
    print(f"    curl -X POST https://slipstream.mara.com/api/transactions \\")
    print(f"      -H 'Content-Type: application/json' \\")
    print(f"      -d '{{\"tx_hex\":\"<fully signed hex>\"}}'")


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
    hash_mode = 'sha256'  # use sha256 for test (faster than grinding for ripemd)
    
    print(f"  n={n}, t1={t1}, t2={t2}, hash={hash_mode}")
    
    import random
    random.seed(42)
    orig = os.urandom
    os.urandom = lambda nb: bytes([random.getrandbits(8) for _ in range(nb)])
    
    builder = QSBScriptBuilder(n, t1s, t1b, t2s, t2b, hash_mode=hash_mode)
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
        ph, ph_der, ph_hc = puzzle_hash(pubkey_bytes, hash_mode)
        
        # Easy check for test speed (real search uses GPU with full DER)
        if ph_der or (ph[0] >> 4) == 3:
            found_lt = lt
            is_real_der = ph_der
            print(f"  Found! locktime={lt}, hash={b2h(ph)} (real_DER={is_real_der}, hc={ph_hc})")
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
            ph, ph_der, ph_hc = puzzle_hash(pk, hash_mode)
            
            if ph_der or (ph[0] >> 4) == 3:
                found_combo = list(combo)
                print(f"  Found! indices={found_combo}, hash={b2h(ph)} (hc={ph_hc})")
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
        'config': 'test', 'n': n, 'hash_mode': hash_mode,
        't1s': t1s, 't1b': t1b, 't2s': t2s, 't2b': t2b,
        'hors_secrets': [[b2h(s) for s in r] for r in builder.hors_secrets],
        'hors_commitments': [[b2h(c) for c in r] for r in builder.hors_commitments],
        'dummy_sigs': [[b2h(s) for s in r] for r in builder.dummy_sigs],
        'pin_r': sigs[0]['r'], 'pin_s': sigs[0]['s'], 'pin_k': sigs[0]['k'],
        'pin_sig': b2h(sigs[0]['sig']),
        'round_sigs': [{'r': sigs[i+1]['r'], 's': sigs[i+1]['s'],
                        'sig': b2h(sigs[i+1]['sig']), 'k': sigs[i+1]['k']} for i in range(2)],
        'full_script_hex': b2h(full_script),
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(test_state, f, indent=2)
    
    # Create a mock args object for assembly
    class MockArgs:
        pass
    asm_args = MockArgs()
    asm_args.locktime = found_lt
    asm_args.sequence = 0xfffffffe
    asm_args.version = 1
    asm_args.round1 = ','.join(str(i) for i in found_round_indices[0])
    asm_args.round2 = ','.join(str(i) for i in found_round_indices[1])
    asm_args.funding_txid = '01' * 32
    asm_args.funding_vout = 0
    asm_args.funding_value = 50000
    # Match the test tx's input[0] and output[0] exactly so the assembled
    # tx reproduces the same sighash that cmd_test used in its search.
    asm_args.extra_input_txid = '00' * 32
    asm_args.extra_input_vout = 0
    asm_args.extra_input_value = 30000
    asm_args.extra_input_sequence = 0xfffffffe
    asm_args.output_value = 45000  # MUST match the cmd_test search tx output
    asm_args.output_address = '00' * 20  # P2PKH '0'*40 → 20-byte hex pubkeyhash
    
    cmd_assemble(asm_args)
    
    print(f"\n  ✓ Full pipeline test complete!")


# ============================================================
# Main
# ============================================================

def cmd_fund(args):
    """Create an unsigned funding transaction that sends to the bare QSB script output.

    Supports either P2PKH (20-byte hex pubkeyhash) or P2WPKH (bc1q... bech32)
    change addresses. The change scriptPubKey is auto-detected from the format.

    The input is signed externally (e.g., bitcoin-cli signrawtransactionwithwallet
    or your hardware wallet). For P2WPKH inputs, your wallet handles witness
    signing automatically.

    For multiple input UTXOs, use --extra-input multiple times. The TX will be
    constructed with all inputs in the given order.
    """
    print("╔══════════════════════════════════════╗")
    print("║  QSB Pipeline — Create Funding Tx    ║")
    print("╚══════════════════════════════════════╝")

    with open(STATE_FILE) as f:
        state = json.load(f)

    full_script = h2b(state['full_script_hex'])

    # Primary input
    input_txid = h2b(args.input_txid)[::-1]
    input_vout = args.input_vout
    input_value = args.input_value
    inputs = [(input_txid, input_vout, input_value, args.input_txid)]

    # Extra inputs (optional, for combining UTXOs)
    if args.extra_input:
        for spec in args.extra_input:
            parts = spec.split(":")
            if len(parts) != 3:
                print(f"  ERROR: --extra-input must be 'txid:vout:value', got '{spec}'")
                return
            ex_txid, ex_vout, ex_value = parts
            ex_vout, ex_value = int(ex_vout), int(ex_value)
            inputs.append((h2b(ex_txid)[::-1], ex_vout, ex_value, ex_txid))

    total_in = sum(v for _, _, v, _ in inputs)
    qsb_value = args.qsb_value
    fee = args.fee
    change_value = total_in - qsb_value - fee

    # Detect change address format
    if args.change_address.startswith(("bc1", "tb1", "bcrt1")):
        change_pkh = bech32_decode_pkh(args.change_address)
        change_script = p2wpkh_script(change_pkh.hex())
        change_kind = "P2WPKH (bech32)"
    else:
        # Treat as 20-byte hex pubkeyhash → P2PKH
        change_pkh = h2b(args.change_address)
        if len(change_pkh) != 20:
            print(f"  ERROR: change-address must be bech32 (bc1q...) or 20-byte hex, got {len(change_pkh)} bytes")
            return
        change_script = p2pkh_script(args.change_address)
        change_kind = "P2PKH (hex hash)"

    print(f"  Inputs:")
    total = 0
    for _, ex_vout, ex_value, ex_txid in inputs:
        print(f"    {ex_txid[:16]}…:{ex_vout} ({ex_value} sats)")
        total += ex_value
    print(f"  Total in: {total} sats")
    print(f"  QSB output: {qsb_value} sats ({len(full_script)} byte bare script)")
    print(f"  Change: {change_value} sats → {args.change_address} ({change_kind})")
    print(f"  Fee: {fee} sats")

    if change_value < 0:
        print(f"\n  ERROR: insufficient funds! Need {qsb_value + fee} sats, have {total_in}")
        return

    # Build unsigned transaction
    tx = Transaction(version=2, locktime=0)
    for txid_le, vout, _, _ in inputs:
        tx.add_input(TxIn(txid_le, vout, b'', 0xfffffffd))

    # Output 0: bare QSB script (scriptPubKey = full locking script)
    tx.add_output(TxOut(qsb_value, full_script))

    # Output 1: change
    if change_value > 546:  # dust threshold
        tx.add_output(TxOut(change_value, change_script))
    else:
        print(f"  (no change output — {change_value} sats below dust threshold; "
              f"raise fee or reduce qsb-value to avoid leaving dust as fee)")

    raw_unsigned = tx.serialize()

    fname = "qsb_funding_unsigned.hex"
    with open(fname, 'w') as f:
        f.write(b2h(raw_unsigned))

    print(f"\n  Unsigned tx: {len(raw_unsigned)} bytes")
    print(f"  Saved to {fname}")
    print(f"\n  ⚠ IMPORTANT: This is a NON-STANDARD transaction (bare script output).")
    print(f"  Standard Bitcoin nodes will NOT relay it.")
    print(f"  Submit via MARA Slipstream or another non-standard mempool service.")
    print(f"\n  To sign with bitcoin-cli (if you have your privkey loaded):")
    print(f"    bitcoin-cli signrawtransactionwithwallet $(cat {fname})")
    print(f"\n  Or with descriptor + UTXOs explicitly:")
    print(f"    bitcoin-cli signrawtransactionwithkey <hex> '[\"<wif>\"]' \\")
    print(f"        '[{{\"txid\":\"...\",\"vout\":N,\"scriptPubKey\":\"...\",\"amount\":N.NN}}]'")
    print(f"\n  Once signed, broadcast via Slipstream:")
    print(f"    curl -X POST https://slipstream.mara.com/api/transactions \\")
    print(f"        -H 'Content-Type: application/json' \\")
    print(f"        -d '{{\"tx_hex\":\"<signed-hex>\"}}'")

    print(f"\n  After confirmation, note the txid and use:")
    print(f"  python3 qsb_pipeline.py export \\")
    print(f"      --funding-txid <txid> --funding-vout 0 --funding-value <sats> \\")
    print(f"      --extra-input-txid <txid> --extra-input-vout <n> --extra-input-value <sats> \\")
    print(f"      --output-value <sats> --output-address bc1q...")


def main():
    parser = argparse.ArgumentParser(description="QSB Pipeline")
    sub = parser.add_subparsers(dest='command')
    
    # Setup
    p_setup = sub.add_parser('setup')
    p_setup.add_argument('--config', default='A')
    p_setup.add_argument('--seed', type=int, default=None)
    
    # Export
    p_export = sub.add_parser('export')
    p_export.add_argument('--funding-txid', required=True,
                          help='Funding tx txid (the QSB output is at vout=funding-vout)')
    p_export.add_argument('--funding-vout', type=int, required=True)
    p_export.add_argument('--funding-value', type=int, required=True)
    # New: extra UTXO that funds input[0] of the spending tx so we have ≥1
    # output-and-input shape (consensus-valid 2-in/1-out tx with QSB at idx 1)
    p_export.add_argument('--extra-input-txid', required=True,
                          help='txid of an extra UTXO YOU control, used as input[0] '
                               'of the spending tx (typically the change output of '
                               'the funding tx)')
    p_export.add_argument('--extra-input-vout', type=int, required=True)
    p_export.add_argument('--extra-input-value', type=int, required=True)
    p_export.add_argument('--extra-input-sequence', type=lambda x: int(x, 0),
                          default=0xfffffffd,
                          help='default 0xfffffffd (RBF off, locktime active)')
    # New: where the spending tx sends its single output
    p_export.add_argument('--output-value', type=int, required=True,
                          help='value of the single output in sats; '
                               'fee = (extra_input_value + funding_value) - output_value')
    p_export.add_argument('--output-address', required=True,
                          help='destination (bech32 P2WPKH preferred, e.g. bc1q...) '
                               'or 20-byte hex pubkeyhash for P2PKH')
    p_export.add_argument('--locktime', type=int, default=0)
    p_export.add_argument('--sequence', type=lambda x: int(x, 0), default=0xfffffffe,
                          help='QSB input sequence (kernel varies). 0xHEX or decimal.')
    p_export.add_argument('--version', type=int, default=2)
    
    # Assemble
    p_asm = sub.add_parser('assemble')
    p_asm.add_argument('--locktime', type=int, required=True)
    p_asm.add_argument('--sequence', type=lambda x: int(x, 0), required=True,
                       help='0xHEX or decimal')
    p_asm.add_argument('--version', type=int, default=2, help='tx version')
    p_asm.add_argument('--round1', required=True, help='R1 STATE indices, csv')
    p_asm.add_argument('--round2', required=True, help='R2 STATE indices, csv')
    p_asm.add_argument('--funding-txid', required=True)
    p_asm.add_argument('--funding-vout', type=int, required=True)
    p_asm.add_argument('--funding-value', type=int, required=True)
    p_asm.add_argument('--extra-input-txid', required=True,
                       help='same value as used at export time')
    p_asm.add_argument('--extra-input-vout', type=int, required=True)
    p_asm.add_argument('--extra-input-value', type=int, required=True)
    p_asm.add_argument('--extra-input-sequence', type=lambda x: int(x, 0),
                       default=0xfffffffd)
    p_asm.add_argument('--output-value', type=int, required=True)
    p_asm.add_argument('--output-address', required=True)
    
    # Test
    p_test = sub.add_parser('test')
    
    # Fund — create unsigned funding transaction
    p_fund = sub.add_parser('fund')
    p_fund.add_argument('--input-txid', required=True, help='primary UTXO txid to spend')
    p_fund.add_argument('--input-vout', type=int, required=True, help='primary UTXO output index')
    p_fund.add_argument('--input-value', type=int, required=True, help='primary UTXO value in sats')
    p_fund.add_argument('--extra-input', action='append', default=[],
                        metavar='TXID:VOUT:VALUE',
                        help='additional UTXO to combine, format txid:vout:value (can repeat)')
    p_fund.add_argument('--qsb-value', type=int, required=True, help='Amount to send to QSB output (sats)')
    p_fund.add_argument('--change-address', required=True,
                        help='Change address: bech32 (bc1q...) for P2WPKH, or 20-byte hex for P2PKH')
    p_fund.add_argument('--fee', type=int, default=3000,
                        help='fee in sats (default 3000; raise for high-fee periods)')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        cmd_setup(args)
    elif args.command == 'export':
        cmd_export(args)
    elif args.command == 'assemble':
        cmd_assemble(args)
    elif args.command == 'test':
        cmd_test(args)
    elif args.command == 'fund':
        cmd_fund(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
