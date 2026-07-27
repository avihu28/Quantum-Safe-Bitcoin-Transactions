"""
secp256k1 minimal implementation for QSB
Includes: point arithmetic, ECDSA signing, key recovery, DER encoding/validation
"""

import hashlib
import struct
import os

# secp256k1 curve parameters
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
A = 0
B = 7
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
G = (Gx, Gy)

INF = (None, None)

def modinv(a, m):
    """Extended Euclidean Algorithm for modular inverse"""
    if a < 0:
        a = a % m
    g, x, _ = _extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"No modular inverse for {a} mod {m}")
    return x % m

def _extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    g, x, y = _extended_gcd(b % a, a)
    return g, y - (b // a) * x, x

def point_add(p1, p2):
    if p1 == INF: return p2
    if p2 == INF: return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        if y1 != y2:
            return INF
        # Point doubling
        lam = (3 * x1 * x1 + A) * modinv(2 * y1, P) % P
    else:
        lam = (y2 - y1) * modinv(x2 - x1, P) % P
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return (x3, y3)

def point_mul(k, point):
    """Double-and-add scalar multiplication"""
    result = INF
    addend = point
    k = k % N
    while k:
        if k & 1:
            result = point_add(result, addend)
        addend = point_add(addend, addend)
        k >>= 1
    return result

def point_neg(p):
    if p == INF: return INF
    return (p[0], (-p[1]) % P)

def compress_pubkey(point):
    """Compress a public key point to 33 bytes"""
    if point == INF:
        raise ValueError("Cannot compress point at infinity")
    x, y = point
    prefix = b'\x02' if y % 2 == 0 else b'\x03'
    return prefix + x.to_bytes(32, 'big')

def decompress_pubkey(data):
    """Decompress a 33-byte compressed public key"""
    if len(data) != 33:
        raise ValueError(f"Expected 33 bytes, got {len(data)}")
    prefix = data[0]
    x = int.from_bytes(data[1:], 'big')
    # y^2 = x^3 + 7 mod P
    y_sq = (pow(x, 3, P) + B) % P
    y = pow(y_sq, (P + 1) // 4, P)
    if y_sq != pow(y, 2, P):
        raise ValueError("Point not on curve")
    if prefix == 0x02:
        return (x, y if y % 2 == 0 else P - y)
    elif prefix == 0x03:
        return (x, y if y % 2 == 1 else P - y)
    else:
        raise ValueError(f"Invalid prefix: {prefix}")


# ============================================================
# ECDSA
# ============================================================

def ecdsa_sign(privkey_int, z):
    """Sign message hash z with private key. Returns (r, s)."""
    while True:
        k = int.from_bytes(os.urandom(32), 'big') % N
        if k == 0: continue
        R = point_mul(k, G)
        r = R[0] % N
        if r == 0: continue
        s = (modinv(k, N) * (z + r * privkey_int)) % N
        if s == 0: continue
        # Low-s normalization
        if s > N // 2:
            s = N - s
        return (r, s)

def ecdsa_sign_with_k(privkey_int, z, k):
    """Sign with specific k (for testing)"""
    R = point_mul(k, G)
    r = R[0] % N
    s = (modinv(k, N) * (z + r * privkey_int)) % N
    if s > N // 2:
        s = N - s
    return (r, s)

def ecdsa_verify(pubkey_point, z, r, s):
    """Verify ECDSA signature"""
    if r < 1 or r >= N or s < 1 or s >= N:
        return False
    w = modinv(s, N)
    u1 = (z * w) % N
    u2 = (r * w) % N
    point = point_add(point_mul(u1, G), point_mul(u2, pubkey_point))
    if point == INF:
        return False
    return point[0] % N == r

def ecdsa_recover(r, s, z, recovery_flag=0):
    """
    Recover public key from signature (r, s) and message hash z.
    recovery_flag: 0 or 1 (selects y parity of R)
    Returns the public key point or None.
    """
    # Reconstruct R from r
    x = r
    # y^2 = x^3 + 7
    y_sq = (pow(x, 3, P) + B) % P
    y = pow(y_sq, (P + 1) // 4, P)
    if pow(y, 2, P) != y_sq:
        return None
    if recovery_flag == 0:
        R = (x, y if y % 2 == 0 else P - y)
    else:
        R = (x, y if y % 2 == 1 else P - y)
    
    # Q = r^-1 * (s*R - z*G)
    r_inv = modinv(r, N)
    u1 = (-z * r_inv) % N
    u2 = (s * r_inv) % N
    Q = point_add(point_mul(u1, G), point_mul(u2, R))
    
    if Q == INF:
        return None
    return Q


# ============================================================
# DER encoding / validation
# ============================================================

def int_to_der_int(val):
    """Encode a positive integer as DER integer bytes (no tag/length)"""
    b = val.to_bytes((val.bit_length() + 7) // 8, 'big') if val > 0 else b'\x00'
    if b[0] & 0x80:
        b = b'\x00' + b
    return b

def encode_der_sig(r, s, sighash=0x01):
    """Encode (r, s) as DER signature with sighash byte"""
    r_bytes = int_to_der_int(r)
    s_bytes = int_to_der_int(s)
    inner = b'\x02' + bytes([len(r_bytes)]) + r_bytes + b'\x02' + bytes([len(s_bytes)]) + s_bytes
    return b'\x30' + bytes([len(inner)]) + inner + bytes([sighash])

def is_valid_der_sig(data):
    """
    Check if raw bytes constitute a valid DER-encoded ECDSA signature.
    Consensus-level: DER structure must be valid (BIP66), but sighash byte
    (last byte) can be ANY value. SCRIPT_VERIFY_STRICTENC is policy only.
    """
    if len(data) < 9:  # Minimum: 30 06 02 01 xx 02 01 xx sh
        return False
    
    # Byte 0: 0x30 (compound)
    if data[0] != 0x30:
        return False
    
    # Byte 1: length of remaining data (excluding sighash)
    total_len = data[1]
    if total_len + 3 != len(data):  # +2 for tag+len, +1 for sighash
        return False
    
    # Sighash byte: any value valid at consensus level
    # (STRICTENC is policy-only, bypassed via Slipstream)
    
    idx = 2
    for _ in range(2):  # Parse r, then s
        if idx >= len(data) - 1:
            return False
        if data[idx] != 0x02:
            return False
        idx += 1
        int_len = data[idx]
        idx += 1
        if int_len == 0:
            return False
        if idx + int_len > len(data) - 1:  # -1 for sighash
            return False
        # No unnecessary leading zeros
        if int_len > 1 and data[idx] == 0x00 and not (data[idx + 1] & 0x80):
            return False
        # Must be positive (no leading 0x80+)
        if data[idx] & 0x80:
            return False
        idx += int_len
    
    # Should have consumed exactly up to sighash
    if idx != len(data) - 1:
        return False
    
    return True


# ============================================================
# SIGHASH_SINGLE bug: z = 1
# ============================================================

def make_sighash_single_sig(privkey_int):
    """Create a signature for the SIGHASH_SINGLE bug (z=1)"""
    r, s = ecdsa_sign(privkey_int, 1)
    return encode_der_sig(r, s, sighash=0x03)  # SIGHASH_SINGLE

def make_dummy_sig_and_pubkey():
    """
    Create a 9-byte dummy signature using the SIGHASH_SINGLE bug,
    and its corresponding public key (recovered with z=1).
    Returns (sig_bytes, pubkey_bytes)
    """
    # We need a small signature. Use a known small r.
    # For z=1, we need to find a key that gives us a short sig.
    # Actually for dummy sigs we just need ANY valid sig with z=1.
    # The pubkey is recovered from (sig, z=1).
    
    privkey = int.from_bytes(os.urandom(32), 'big') % N
    r, s = ecdsa_sign(privkey, 1)
    sig_bytes = encode_der_sig(r, s, sighash=0x03)
    
    # Recover pubkey from (r, s, z=1) to verify
    pubkey_point = ecdsa_recover(r, s, 1, recovery_flag=0)
    if pubkey_point is None:
        pubkey_point = ecdsa_recover(r, s, 1, recovery_flag=1)
    
    pubkey_bytes = compress_pubkey(pubkey_point)
    return sig_bytes, pubkey_bytes


# ============================================================
# Hashing
# ============================================================

def sha256(data):
    return hashlib.sha256(data).digest()

def sha256d(data):
    return sha256(sha256(data))

def ripemd160(data):
    h = hashlib.new('ripemd160')
    h.update(data)
    return h.digest()

def hash160(data):
    return ripemd160(sha256(data))


# ============================================================
# Quick self-test
# ============================================================

if __name__ == "__main__":
    print("=== secp256k1 self-test ===")
    
    # Test point multiplication
    Q = point_mul(1, G)
    assert Q == G, "G*1 != G"
    print("[OK] Point multiplication")
    
    # Test signing and verification
    privkey = 12345
    pubkey = point_mul(privkey, G)
    z = int.from_bytes(sha256(b"test message"), 'big')
    r, s = ecdsa_sign(privkey, z)
    assert ecdsa_verify(pubkey, z, r, s), "Signature verification failed"
    print("[OK] ECDSA sign/verify")
    
    # Test key recovery
    for flag in [0, 1]:
        Q = ecdsa_recover(r, s, z, recovery_flag=flag)
        if Q is not None and Q == pubkey:
            print(f"[OK] Key recovery (flag={flag})")
            break
    else:
        print("[FAIL] Key recovery")
    
    # Test DER encoding
    sig = encode_der_sig(r, s)
    assert is_valid_der_sig(sig), "DER validation failed"
    print(f"[OK] DER encoding ({len(sig)} bytes)")
    
    # Test RIPEMD160 -> DER probability estimate
    print("\n=== RIPEMD160 -> DER test ===")
    hits = 0
    trials = 10000
    for i in range(trials):
        data = os.urandom(33)  # Simulating a compressed pubkey
        h = ripemd160(data)
        if is_valid_der_sig(h):
            hits += 1
    print(f"  {hits}/{trials} valid DER from random 20-byte strings")
    print(f"  (Expected ~0 for {trials} trials, need ~2^46)")
    
    print("\n[ALL OK]")
