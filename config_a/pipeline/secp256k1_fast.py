"""
Fast secp256k1 operations using coincurve (C bindings to libsecp256k1).
Drop-in replacement for secp256k1.py — same API, ~10,000x faster.

Install: pip install coincurve
"""

import hashlib
import struct
import os

try:
    import coincurve
    from coincurve import PrivateKey, PublicKey
    HAS_COINCURVE = True
except ImportError:
    HAS_COINCURVE = False
    print("WARNING: coincurve not installed. Falling back to pure Python.")
    print("Install with: pip install coincurve")
    from secp256k1 import *  # fallback

if HAS_COINCURVE:

    # Curve parameters (for reference / compatibility)
    N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    B = 7

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
    # Key operations
    # ============================================================

    def compress_pubkey(point):
        """Compress (x, y) tuple to 33 bytes"""
        if point is None or point == (None, None):
            raise ValueError("Cannot compress point at infinity")
        x, y = point
        prefix = b'\x02' if y % 2 == 0 else b'\x03'
        return prefix + x.to_bytes(32, 'big')

    def decompress_pubkey(data):
        """Decompress 33-byte key to (x, y) tuple"""
        pub = PublicKey(data)
        # Get uncompressed (65 bytes: 04 + x + y)
        uncompressed = pub.format(compressed=False)
        x = int.from_bytes(uncompressed[1:33], 'big')
        y = int.from_bytes(uncompressed[33:65], 'big')
        return (x, y)

    def point_mul(k, point):
        """Scalar multiplication using coincurve"""
        if point is None or point == (None, None):
            return (None, None)
        # Convert point to PublicKey
        x, y = point
        uncompressed = b'\x04' + x.to_bytes(32, 'big') + y.to_bytes(32, 'big')
        pub = PublicKey(uncompressed)
        # Multiply
        k_bytes = (k % N).to_bytes(32, 'big')
        result = pub.multiply(k_bytes)
        # Parse result
        unc = result.format(compressed=False)
        rx = int.from_bytes(unc[1:33], 'big')
        ry = int.from_bytes(unc[33:65], 'big')
        return (rx, ry)

    def point_add(p1, p2):
        """Point addition using coincurve"""
        if p1 is None or p1 == (None, None):
            return p2
        if p2 is None or p2 == (None, None):
            return p1
        x1, y1 = p1
        x2, y2 = p2
        unc1 = b'\x04' + x1.to_bytes(32, 'big') + y1.to_bytes(32, 'big')
        unc2 = b'\x04' + x2.to_bytes(32, 'big') + y2.to_bytes(32, 'big')
        pub1 = PublicKey(unc1)
        pub2 = PublicKey(unc2)
        result = pub1.combine([pub2])
        unc = result.format(compressed=False)
        rx = int.from_bytes(unc[1:33], 'big')
        ry = int.from_bytes(unc[33:65], 'big')
        return (rx, ry)

    def point_neg(p):
        if p is None or p == (None, None):
            return (None, None)
        return (p[0], (-p[1]) % P)

    # Generator point
    G = decompress_pubkey(bytes.fromhex(
        '0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798'
    ))

    INF = (None, None)

    def modinv(a, m):
        if a < 0: a = a % m
        g, x, _ = _extended_gcd(a, m)
        if g != 1: raise ValueError(f"No modular inverse")
        return x % m

    def _extended_gcd(a, b):
        if a == 0: return b, 0, 1
        g, x, y = _extended_gcd(b % a, a)
        return g, y - (b // a) * x, x

    # ============================================================
    # ECDSA
    # ============================================================

    def ecdsa_sign(privkey_int, z):
        """Sign message hash z. Returns (r, s)."""
        pk = PrivateKey(privkey_int.to_bytes(32, 'big'))
        z_bytes = z.to_bytes(32, 'big')
        # coincurve signs a hash directly
        sig_der = pk.sign(z_bytes, hasher=None)
        # Parse DER to get r, s
        r, s = _parse_der_rs(sig_der)
        # Low-s normalization
        if s > N // 2:
            s = N - s
        return (r, s)

    def ecdsa_verify(pubkey_point, z, r, s):
        """Verify ECDSA signature"""
        x, y = pubkey_point
        unc = b'\x04' + x.to_bytes(32, 'big') + y.to_bytes(32, 'big')
        pub = PublicKey(unc)
        sig_der = _encode_der_rs(r, s)
        z_bytes = z.to_bytes(32, 'big')
        try:
            return pub.verify(sig_der, z_bytes, hasher=None)
        except Exception:
            return False

    def ecdsa_recover(r, s, z, recovery_flag=0):
        """
        Recover public key from (r, s, z).
        Returns (x, y) tuple or None.
        """
        sig_bytes = _encode_recoverable_sig(r, s, recovery_flag)
        z_bytes = z.to_bytes(32, 'big')
        try:
            pub = PublicKey.from_signature_and_message(sig_bytes, z_bytes, hasher=None)
            unc = pub.format(compressed=False)
            x = int.from_bytes(unc[1:33], 'big')
            y = int.from_bytes(unc[33:65], 'big')
            return (x, y)
        except Exception:
            return None

    def ecdsa_recover_compressed(r, s, z, recovery_flag=0):
        """
        Recover public key from (r, s, z).
        Returns 33-byte compressed pubkey directly (fastest path).
        """
        sig_bytes = _encode_recoverable_sig(r, s, recovery_flag)
        z_bytes = z.to_bytes(32, 'big')
        try:
            pub = PublicKey.from_signature_and_message(sig_bytes, z_bytes, hasher=None)
            return pub.format(compressed=True)
        except Exception:
            return None

    def _encode_recoverable_sig(r, s, recovery_flag):
        """Encode (r, s, flag) as 65-byte recoverable signature"""
        return r.to_bytes(32, 'big') + s.to_bytes(32, 'big') + bytes([recovery_flag])

    def _parse_der_rs(der_bytes):
        """Parse r, s from DER signature (no sighash byte)"""
        idx = 2  # skip 30 len
        assert der_bytes[idx] == 0x02
        r_len = der_bytes[idx + 1]; idx += 2
        r = int.from_bytes(der_bytes[idx:idx + r_len], 'big'); idx += r_len
        assert der_bytes[idx] == 0x02
        s_len = der_bytes[idx + 1]; idx += 2
        s = int.from_bytes(der_bytes[idx:idx + s_len], 'big')
        return r, s

    def _encode_der_rs(r, s):
        """Encode (r, s) as DER signature (no sighash byte)"""
        def int_to_der(val):
            b = val.to_bytes((val.bit_length() + 7) // 8, 'big') if val > 0 else b'\x00'
            if b[0] & 0x80: b = b'\x00' + b
            return b
        r_b = int_to_der(r)
        s_b = int_to_der(s)
        inner = b'\x02' + bytes([len(r_b)]) + r_b + b'\x02' + bytes([len(s_b)]) + s_b
        return b'\x30' + bytes([len(inner)]) + inner

    # ============================================================
    # DER encoding for Bitcoin (with sighash byte)
    # ============================================================

    def int_to_der_int(val):
        b = val.to_bytes((val.bit_length() + 7) // 8, 'big') if val > 0 else b'\x00'
        if b[0] & 0x80: b = b'\x00' + b
        return b

    def encode_der_sig(r, s, sighash=0x01):
        r_bytes = int_to_der_int(r)
        s_bytes = int_to_der_int(s)
        inner = b'\x02' + bytes([len(r_bytes)]) + r_bytes + b'\x02' + bytes([len(s_bytes)]) + s_bytes
        return b'\x30' + bytes([len(inner)]) + inner + bytes([sighash])

    def is_valid_der_sig(data):
        if len(data) < 9: return False
        if data[0] != 0x30: return False
        total_len = data[1]
        if total_len + 3 != len(data): return False
        sighash = data[-1]
        if sighash == 0 or sighash > 0x83: return False
        idx = 2
        for _ in range(2):
            if idx >= len(data) - 1: return False
            if data[idx] != 0x02: return False
            idx += 1
            int_len = data[idx]; idx += 1
            if int_len == 0: return False
            if idx + int_len > len(data) - 1: return False
            if int_len > 1 and data[idx] == 0x00 and not (data[idx + 1] & 0x80): return False
            if data[idx] & 0x80: return False
            idx += int_len
        if idx != len(data) - 1: return False
        return True

    # ============================================================
    # Benchmark
    # ============================================================

    if __name__ == "__main__":
        import time

        print("=== coincurve fast secp256k1 ===")

        # Benchmark key recovery
        privkey = 12345
        pubkey = point_mul(privkey, G)
        z = int.from_bytes(sha256(b"test"), 'big')
        r, s = ecdsa_sign(privkey, z)

        # Verify
        assert ecdsa_verify(pubkey, z, r, s), "Verify failed"
        print("[OK] Sign/Verify")

        # Recovery
        Q = ecdsa_recover(r, s, z, 0)
        if Q != pubkey:
            Q = ecdsa_recover(r, s, z, 1)
        assert Q == pubkey, f"Recovery failed: {Q} != {pubkey}"
        print("[OK] Key recovery")

        # Benchmark recovery
        t0 = time.time()
        count = 10000
        for i in range(count):
            z_test = int.from_bytes(sha256(i.to_bytes(4, 'big')), 'big')
            key = ecdsa_recover_compressed(r, s, z_test, 0)
        elapsed = time.time() - t0
        print(f"[BENCH] {count} recoveries: {elapsed:.2f}s ({count/elapsed:.0f}/s)")

        # Benchmark recovery + ripemd160
        t0 = time.time()
        for i in range(count):
            z_test = int.from_bytes(sha256(i.to_bytes(4, 'big')), 'big')
            key = ecdsa_recover_compressed(r, s, z_test, 0)
            if key:
                h = ripemd160(key)
        elapsed = time.time() - t0
        print(f"[BENCH] {count} recover+ripemd: {elapsed:.2f}s ({count/elapsed:.0f}/s)")

        # DER check
        sig = encode_der_sig(r, s)
        assert is_valid_der_sig(sig)
        print("[OK] DER validation")

        print("\n[ALL OK]")
