#!/usr/bin/env python3
"""
QSB Funding Transaction Builder

Creates a funding transaction that sends Bitcoin from a P2WPKH (bc1q) address
to a bare script output containing the QSB locking script.

Usage:
    python3 fund_qsb.py --wif <private_key_WIF> --amount <sats> [--fee-rate <sat/vB>]

Requirements:
    pip install requests

The script:
  1. Loads the QSB script from qsb_state.json
  2. Derives the public key from your WIF private key
  3. Fetches UTXOs from mempool.space
  4. Builds a SegWit (P2WPKH) spending transaction with bare script output
  5. Signs using BIP143
  6. Submits via MARA Slipstream API (accepts non-standard txs)
"""

import sys
import os
import json
import hashlib
import struct
import argparse

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from secp256k1 import point_mul, G, N, P as CURVE_P, compress_pubkey, modinv

# ============================================================
# Helpers
# ============================================================

def sha256(data):
    return hashlib.sha256(data).digest()

def sha256d(data):
    return sha256(sha256(data))

def hash160(data):
    return hashlib.new('ripemd160', sha256(data)).digest()

def b2h(b):
    return b.hex()

def h2b(h):
    return bytes.fromhex(h)

# ============================================================
# Bech32 decoding (for bc1q addresses)
# ============================================================

BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

def bech32_polymod(values):
    GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = ((chk & 0x1ffffff) << 5) ^ v
        for i in range(5):
            chk ^= GEN[i] if ((b >> i) & 1) else 0
    return chk

def bech32_decode(bech):
    if bech != bech.lower() and bech != bech.upper():
        return None, None, None
    bech = bech.lower()
    pos = bech.rfind('1')
    if pos < 1 or pos + 7 > len(bech):
        return None, None, None
    hrp = bech[:pos]
    data = [BECH32_CHARSET.find(c) for c in bech[pos+1:]]
    if -1 in data:
        return None, None, None
    hrp_expand = [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]
    if bech32_polymod(hrp_expand + data) != 1:
        return None, None, None
    return hrp, data[0], data[1:-6]

def convertbits(data, frombits, tobits, pad=True):
    acc, bits, ret = 0, 0, []
    maxv = (1 << tobits) - 1
    for value in data:
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad and bits:
        ret.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        return None
    return ret

def decode_bech32_address(addr):
    """Decode bc1q... address → (witness_version, witness_program_bytes)"""
    hrp, ver, data = bech32_decode(addr)
    if hrp is None:
        raise ValueError(f"Invalid bech32 address: {addr}")
    decoded = convertbits(data, 5, 8, False)
    if decoded is None or len(decoded) < 2:
        raise ValueError(f"Invalid witness program")
    return ver, bytes(decoded)

# ============================================================
# WIF decoding
# ============================================================

B58_CHARS = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

def b58decode(s):
    n = 0
    for c in s:
        n = n * 58 + B58_CHARS.index(c)
    result = n.to_bytes(max(1, (n.bit_length() + 7) // 8), 'big')
    # Preserve leading zeros
    pad = 0
    for c in s:
        if c == '1':
            pad += 1
        else:
            break
    return b'\x00' * pad + result

def wif_to_privkey(wif):
    """Decode WIF private key → (privkey_int, compressed)"""
    raw = b58decode(wif)
    # Verify checksum
    payload = raw[:-4]
    checksum = raw[-4:]
    if sha256d(payload)[:4] != checksum:
        raise ValueError("Invalid WIF checksum")
    
    if payload[0] == 0x80:  # mainnet
        pass
    elif payload[0] == 0xef:  # testnet
        pass
    else:
        raise ValueError(f"Unknown WIF version: 0x{payload[0]:02x}")
    
    if len(payload) == 34 and payload[-1] == 0x01:
        # Compressed
        return int.from_bytes(payload[1:33], 'big'), True
    elif len(payload) == 33:
        # Uncompressed
        return int.from_bytes(payload[1:33], 'big'), False
    else:
        raise ValueError(f"Invalid WIF payload length: {len(payload)}")

# ============================================================
# BIP39 + BIP32 (seed phrase → private key)
# ============================================================

def mnemonic_to_seed(mnemonic, passphrase=""):
    """BIP39: mnemonic words → 64-byte seed"""
    import hmac
    mnemonic_bytes = mnemonic.encode('utf-8')
    salt = ('mnemonic' + passphrase).encode('utf-8')
    return hashlib.pbkdf2_hmac('sha512', mnemonic_bytes, salt, 2048)

def hmac_sha512(key, data):
    import hmac
    return hmac.new(key, data, hashlib.sha512).digest()

def bip32_master_key(seed):
    """BIP32: seed → (master_privkey, master_chaincode)"""
    I = hmac_sha512(b"Bitcoin seed", seed)
    return I[:32], I[32:]

def bip32_derive_child(privkey_bytes, chaincode, index):
    """BIP32: derive child key at index (hardened if index >= 0x80000000)"""
    if index >= 0x80000000:
        # Hardened: HMAC(key=chaincode, data=0x00 || privkey || index)
        data = b'\x00' + privkey_bytes + struct.pack('>I', index)
    else:
        # Normal: HMAC(key=chaincode, data=pubkey || index)
        k_int = int.from_bytes(privkey_bytes, 'big')
        pub = compress_pubkey(point_mul(k_int, G))
        data = pub + struct.pack('>I', index)
    
    I = hmac_sha512(chaincode, data)
    child_key = (int.from_bytes(I[:32], 'big') + int.from_bytes(privkey_bytes, 'big')) % N
    child_chaincode = I[32:]
    return child_key.to_bytes(32, 'big'), child_chaincode

def derive_from_path(seed, path):
    """Derive private key from BIP32 path string like m/84'/0'/0'/0/0"""
    privkey, chaincode = bip32_master_key(seed)
    
    parts = path.strip().split('/')
    assert parts[0] == 'm', f"Path must start with 'm', got '{parts[0]}'"
    
    for part in parts[1:]:
        if part.endswith("'") or part.endswith("h"):
            index = int(part[:-1]) + 0x80000000  # hardened
        else:
            index = int(part)
        privkey, chaincode = bip32_derive_child(privkey, chaincode, index)
    
    return int.from_bytes(privkey, 'big')

def seed_to_privkey(mnemonic, path="m/84'/0'/0'/0/0"):
    """Full BIP39+BIP32: mnemonic → private key at path"""
    seed = mnemonic_to_seed(mnemonic)
    return derive_from_path(seed, path)

# ============================================================
# SegWit transaction building + BIP143 signing
# ============================================================

def serialize_varint(n):
    if n < 0xfd:
        return bytes([n])
    elif n <= 0xffff:
        return b'\xfd' + struct.pack('<H', n)
    elif n <= 0xffffffff:
        return b'\xfe' + struct.pack('<I', n)
    else:
        return b'\xff' + struct.pack('<Q', n)

def encode_der_sig(r, s, sighash=0x01):
    """Encode ECDSA signature as DER + sighash byte"""
    def int_bytes(v):
        b = v.to_bytes(32, 'big').lstrip(b'\x00') or b'\x00'
        if b[0] & 0x80:
            b = b'\x00' + b
        return b
    rb = int_bytes(r)
    sb = int_bytes(s)
    der = b'\x30' + bytes([len(rb) + len(sb) + 4])
    der += b'\x02' + bytes([len(rb)]) + rb
    der += b'\x02' + bytes([len(sb)]) + sb
    return der + bytes([sighash])

def ecdsa_sign(privkey, z):
    """Sign message hash z with private key"""
    import random
    while True:
        k = random.randrange(1, N)
        R = point_mul(k, G)
        r = R[0] % N
        if r == 0:
            continue
        s = (modinv(k, N) * (z + r * privkey)) % N
        if s == 0:
            continue
        # Low-S normalization
        if s > N // 2:
            s = N - s
        return r, s

def build_and_sign_funding_tx(privkey, compressed, pubkey_hash, utxos, 
                               qsb_script, qsb_value, fee_rate=10):
    """Build and sign a P2WPKH → bare script funding transaction.
    
    Returns (raw_tx_hex, txid).
    """
    pubkey_point = point_mul(privkey, G)
    pubkey_bytes = compress_pubkey(pubkey_point)
    derived_hash = hash160(pubkey_bytes)
    
    if derived_hash != pubkey_hash:
        raise ValueError(f"Private key doesn't match address!\n"
                        f"  Expected: {b2h(pubkey_hash)}\n"
                        f"  Got:      {b2h(derived_hash)}")
    
    # Select UTXOs (simple: use all, or enough to cover)
    total_in = 0
    selected = []
    for u in utxos:
        selected.append(u)
        total_in += u['value']
        if total_in >= qsb_value + 5000:  # rough estimate
            break
    
    if total_in < qsb_value + 1000:
        raise ValueError(f"Insufficient funds: have {total_in}, need {qsb_value} + fees")
    
    n_inputs = len(selected)
    
    # Estimate size for fee calculation:
    # SegWit tx overhead: ~10 bytes
    # Per P2WPKH input: ~41 (non-witness) + ~107 (witness) bytes → ~69 vbytes
    # Bare script output: ~9 + len(qsb_script) bytes
    # Change output (P2WPKH): ~31 bytes
    vsize_est = 10 + n_inputs * 69 + 9 + len(qsb_script) + 31
    fee = vsize_est * fee_rate
    
    change_value = total_in - qsb_value - fee
    
    print(f"  Inputs: {n_inputs} UTXOs, total {total_in} sats")
    print(f"  QSB output: {qsb_value} sats")
    print(f"  Fee: {fee} sats ({fee_rate} sat/vB, ~{vsize_est} vB)")
    print(f"  Change: {change_value} sats")
    
    if change_value < 0:
        raise ValueError(f"Insufficient funds after fee: need {fee} more sats")
    
    # === Build transaction ===
    version = struct.pack('<I', 1)
    marker = b'\x00'  # SegWit marker
    flag = b'\x01'     # SegWit flag
    locktime = struct.pack('<I', 0)
    
    # Inputs
    vin_count = serialize_varint(n_inputs)
    vin = b''
    for u in selected:
        vin += h2b(u['txid'])[::-1]  # txid LE
        vin += struct.pack('<I', u['vout'])
        vin += b'\x00'  # empty scriptSig (SegWit)
        vin += struct.pack('<I', 0xffffffff)
    
    # Outputs
    outputs = []
    
    # Output 0: QSB bare script
    out0 = struct.pack('<q', qsb_value)
    out0 += serialize_varint(len(qsb_script))
    out0 += qsb_script
    outputs.append(out0)
    
    # Output 1: change (P2WPKH — native SegWit)
    if change_value > 546:  # above dust
        change_spk = b'\x00\x14' + pubkey_hash  # OP_0 <20-byte hash>
        out1 = struct.pack('<q', change_value)
        out1 += serialize_varint(len(change_spk))
        out1 += change_spk
        outputs.append(out1)
    
    vout_count = serialize_varint(len(outputs))
    vout = b''.join(outputs)
    
    # === BIP143 signing ===
    
    # hashPrevouts
    prevouts = b''
    for u in selected:
        prevouts += h2b(u['txid'])[::-1] + struct.pack('<I', u['vout'])
    hash_prevouts = sha256d(prevouts)
    
    # hashSequence
    sequences = b''
    for _ in selected:
        sequences += struct.pack('<I', 0xffffffff)
    hash_sequence = sha256d(sequences)
    
    # hashOutputs
    hash_outputs = sha256d(vout)
    
    # Sign each input
    witnesses = []
    for i, u in enumerate(selected):
        # BIP143 scriptCode for P2WPKH: OP_DUP OP_HASH160 <20> OP_EQUALVERIFY OP_CHECKSIG
        script_code = b'\x76\xa9\x14' + pubkey_hash + b'\x88\xac'
        
        outpoint = h2b(u['txid'])[::-1] + struct.pack('<I', u['vout'])
        amount = struct.pack('<q', u['value'])
        
        # BIP143 sighash preimage
        preimage = struct.pack('<I', 1)       # version
        preimage += hash_prevouts
        preimage += hash_sequence
        preimage += outpoint
        preimage += serialize_varint(len(script_code)) + script_code
        preimage += amount
        preimage += struct.pack('<I', 0xffffffff)  # sequence
        preimage += hash_outputs
        preimage += locktime
        preimage += struct.pack('<I', 1)      # SIGHASH_ALL
        
        z = int.from_bytes(sha256d(preimage), 'big')
        r, s = ecdsa_sign(privkey, z)
        sig = encode_der_sig(r, s, sighash=0x01)
        
        # Witness: [sig, pubkey]
        wit = b'\x02'  # 2 witness items
        wit += serialize_varint(len(sig)) + sig
        wit += serialize_varint(len(pubkey_bytes)) + pubkey_bytes
        witnesses.append(wit)
    
    # === Serialize full SegWit transaction ===
    raw_tx = version + marker + flag
    raw_tx += vin_count + vin
    raw_tx += vout_count + vout
    for w in witnesses:
        raw_tx += w
    raw_tx += locktime
    
    # Compute txid (without witness data)
    tx_no_witness = version + vin_count + vin + vout_count + vout + locktime
    txid = sha256d(tx_no_witness)[::-1]  # display byte order
    
    return b2h(raw_tx), b2h(txid), fee, change_value

# ============================================================
# API calls
# ============================================================

def fetch_utxos(address):
    """Fetch UTXOs from mempool.space API"""
    import requests
    url = f"https://mempool.space/api/address/{address}/utxo"
    print(f"  Fetching UTXOs from mempool.space...")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    utxos = resp.json()
    confirmed = [u for u in utxos if u['status']['confirmed']]
    print(f"  Found {len(confirmed)} confirmed UTXOs ({sum(u['value'] for u in confirmed)} sats total)")
    return confirmed

def submit_slipstream(raw_hex, dry_run=False):
    """Submit transaction via MARA Slipstream API.
    
    POST /api/transactions
    Body: {"tx_hex": "<hex>"}
    Response: {"status": "success", "message": "<txid>"}
    """
    import requests
    base = "https://slipstream.mara.com"
    
    # First test mempool acceptance
    print(f"\n  Testing mempool acceptance...")
    try:
        test_resp = requests.post(f"{base}/api/mempool/tests",
                                  json={"tx_hexes": [raw_hex]}, timeout=30)
        print(f"  Mempool test: {test_resp.status_code} {test_resp.text[:300]}")
    except Exception as e:
        print(f"  Mempool test failed: {e}")
    
    if dry_run:
        return False
    
    # Submit
    print(f"\n  Submitting to Slipstream ({len(raw_hex)//2} bytes)...")
    resp = requests.post(f"{base}/api/transactions",
                         json={"tx_hex": raw_hex}, timeout=30)
    
    if resp.status_code == 200:
        result = resp.json()
        if result.get('status') == 'success':
            print(f"  ✓ Accepted! TXID: {result.get('message')}")
            return True
        else:
            print(f"  ✗ Error: {result.get('message')}")
            return False
    else:
        print(f"  ✗ Rejected: {resp.status_code} {resp.text[:300]}")
        return False

def get_slipstream_rates():
    """Get Slipstream fee rate info from /api/rates."""
    import requests
    try:
        resp = requests.get("https://slipstream.mara.com/api/rates", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except:
        return None

def get_slipstream_block_info():
    """Get Slipstream block info from /api/block-info."""
    import requests
    try:
        resp = requests.get("https://slipstream.mara.com/api/block-info", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except:
        return None

def get_fee_estimate():
    """Get current fee rate from mempool.space"""
    import requests
    try:
        resp = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=10)
        resp.raise_for_status()
        fees = resp.json()
        return fees
    except:
        return None

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="QSB Funding Transaction Builder")
    key_group = parser.add_mutually_exclusive_group(required=True)
    key_group.add_argument('--wif', help='Private key in WIF format')
    key_group.add_argument('--seed', help='BIP39 seed phrase (12 or 24 words, quoted)')
    parser.add_argument('--path', default="m/84'/0'/0'/0/0", help='BIP32 derivation path (default: m/84\'/0\'/0\'/0/0)')
    parser.add_argument('--amount', type=int, default=10000, help='Amount to send to QSB output (sats, default 10000)')
    parser.add_argument('--fee-rate', type=int, default=0, help='Fee rate (sat/vB, 0=auto)')
    parser.add_argument('--state-file', default='qsb_state.json', help='State file from setup')
    parser.add_argument('--dry-run', action='store_true', help='Build tx but do not submit')
    args = parser.parse_args()
    
    print("╔══════════════════════════════════════════╗")
    print("║  QSB Funding Transaction Builder          ║")
    print("╚══════════════════════════════════════════╝")
    
    # Load QSB script
    with open(args.state_file) as f:
        state = json.load(f)
    qsb_script = h2b(state['full_script_hex'])
    print(f"\n  QSB script: {len(qsb_script)} bytes")
    
    # Derive private key
    if args.wif:
        privkey, compressed = wif_to_privkey(args.wif)
        if not compressed:
            print("  ERROR: uncompressed key. P2WPKH requires compressed keys.")
            return
        print(f"  Key source: WIF")
    else:
        mnemonic = args.seed.strip()
        words = mnemonic.split()
        if len(words) not in (12, 24):
            print(f"  ERROR: expected 12 or 24 words, got {len(words)}")
            return
        privkey = seed_to_privkey(mnemonic, args.path)
        compressed = True
        print(f"  Key source: BIP39 seed ({len(words)} words)")
        print(f"  Derivation path: {args.path}")
    
    # Derive address
    pubkey_point = point_mul(privkey, G)
    pubkey_bytes = compress_pubkey(pubkey_point)
    pkh = hash160(pubkey_bytes)
    
    # Encode as bc1q address for display
    print(f"  Pubkey hash: {b2h(pkh)}")
    
    # Fetch UTXOs
    # Construct bc1q address from pubkey hash
    # (for API query — we need the address string)
    witness_prog = list(pkh)
    conv = convertbits(witness_prog, 8, 5, True)
    # Bech32 encoding
    hrp = 'bc'
    hrp_expand = [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]
    values = [0] + conv  # witness version 0
    polymod_input = hrp_expand + values + [0, 0, 0, 0, 0, 0]
    polymod = bech32_polymod(polymod_input) ^ 1
    checksum = [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]
    addr = hrp + '1' + ''.join(BECH32_CHARSET[d] for d in values + checksum)
    print(f"  Address: {addr}")
    
    utxos = fetch_utxos(addr)
    if not utxos:
        print("  ERROR: no confirmed UTXOs found!")
        return
    
    # Fee rate — Slipstream requires max(2× mempool priority, 2 sat/vB)
    fee_rate = args.fee_rate
    if fee_rate == 0:
        # Try Slipstream's own rate API first
        ss_info = get_slipstream_block_info()
        if ss_info:
            fee_rate = int(ss_info.get('submit_fee_rate', 4)) + 1  # +1 buffer
            print(f"\n  Slipstream submit_fee_rate: {ss_info.get('submit_fee_rate')} sat/vB")
            print(f"  Slipstream fee_rate (to be mined): {ss_info.get('fee_rate')} sat/vB")
            print(f"  Using: {fee_rate} sat/vB")
        else:
            # Fallback to mempool.space
            fees = get_fee_estimate()
            if fees:
                mempool_rate = fees.get('halfHourFee', 5)
                fee_rate = max(2 * mempool_rate, 4)
                print(f"\n  Mempool half-hour fee: {mempool_rate} sat/vB")
                print(f"  Slipstream min (2× mempool): {fee_rate} sat/vB")
            else:
                fee_rate = 10
                print(f"  Could not fetch fee estimates, using {fee_rate} sat/vB")
    
    # Build and sign
    print(f"\n  Building funding transaction...")
    raw_hex, txid, fee, change = build_and_sign_funding_tx(
        privkey, compressed, pkh, utxos, qsb_script, args.amount, fee_rate
    )
    
    print(f"\n  Transaction ID: {txid}")
    print(f"  Raw tx size: {len(raw_hex)//2} bytes")
    
    # Save
    with open('qsb_funding_tx.hex', 'w') as f:
        f.write(raw_hex)
    print(f"  Saved to qsb_funding_tx.hex")
    
    # Test mempool acceptance
    print(f"\n  Testing mempool acceptance...")
    submit_slipstream(raw_hex, dry_run=True)
    
    # Submit
    if args.dry_run:
        print(f"\n  DRY RUN — not submitting.")
        print(f"  To submit manually:")
        print(f"  curl -X POST https://slipstream.mara.com/api/transactions \\")
        print(f"    -H 'Content-Type: application/json' \\")
        print(f"    -d '{{\"tx_hex\":\"<contents of qsb_funding_tx.hex>\"}}'")
    else:
        confirm = input(f"\n  Submit {args.amount} sats to QSB output via Slipstream? (yes/no): ")
        if confirm.lower() in ('yes', 'y'):
            if submit_slipstream(raw_hex):
                print(f"\n  ✓ Transaction submitted!")
                print(f"  TXID: {txid}")
                print(f"  Monitor: https://mempool.space/tx/{txid}")
                print(f"\n  After confirmation, run:")
                print(f"  python3 qsb_pipeline.py export --funding-txid {txid} --funding-vout 0 --funding-value {args.amount} --dest-address <your_pkh_hex>")
            else:
                print(f"\n  Transaction saved to qsb_funding_tx.hex")
                print(f"  You can resubmit manually.")
        else:
            print("  Cancelled.")

if __name__ == '__main__':
    main()
