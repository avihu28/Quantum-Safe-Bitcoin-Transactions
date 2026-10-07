#!/usr/bin/env python3
"""
Test suite for QSB pipeline components.

Covers: secp256k1 primitives, DER encoding/parsing, ECDSA sign/verify/recover,
        Bitcoin transaction serialization, FindAndDelete, sighash computation,
        and script building.

Run:
    cd pipeline && python3 -m pytest test_qsb.py -v
    # or without pytest:
    cd pipeline && python3 test_qsb.py
"""

import hashlib
import os
import struct
import sys
import unittest

from secp256k1 import (
    P, N, G, INF,
    modinv, point_add, point_mul, point_neg,
    compress_pubkey, decompress_pubkey,
    ecdsa_sign, ecdsa_sign_with_k, ecdsa_verify, ecdsa_recover,
    encode_der_sig, is_valid_der_sig, parse_der, int_to_der_int,
    sha256, sha256d, ripemd160, hash160,
)
from bitcoin_tx import (
    Transaction, TxIn, TxOut, QSBScriptBuilder,
    push_data, push_number, find_and_delete, serialize_varint,
    _valid_small_r_values, _encode_9byte_sig,
)


# ============================================================
# secp256k1 field & curve arithmetic
# ============================================================

class TestFieldArithmetic(unittest.TestCase):

    def test_modinv_basic(self):
        """modinv(a, m) * a == 1 (mod m)"""
        for a in [1, 2, 7, 123456789, N - 1]:
            inv = modinv(a, N)
            self.assertEqual((a * inv) % N, 1)

    def test_modinv_negative(self):
        """modinv handles negative inputs"""
        inv = modinv(-3, N)
        self.assertEqual((-3 * inv) % N, 1)

    def test_modinv_no_inverse(self):
        """modinv raises for non-coprime inputs"""
        with self.assertRaises(ValueError):
            modinv(0, N)

    def test_generator_on_curve(self):
        """G satisfies y^2 = x^3 + 7 (mod P)"""
        x, y = G
        self.assertEqual(pow(y, 2, P), (pow(x, 3, P) + 7) % P)

    def test_point_add_identity(self):
        """P + INF = P"""
        self.assertEqual(point_add(G, INF), G)
        self.assertEqual(point_add(INF, G), G)
        self.assertEqual(point_add(INF, INF), INF)

    def test_point_add_inverse(self):
        """P + (-P) = INF"""
        neg_G = point_neg(G)
        self.assertEqual(point_add(G, neg_G), INF)

    def test_point_mul_identity(self):
        """1*G = G"""
        self.assertEqual(point_mul(1, G), G)

    def test_point_mul_zero(self):
        """0*G = INF"""
        self.assertEqual(point_mul(0, G), INF)

    def test_point_mul_order(self):
        """N*G = INF (curve order)"""
        self.assertEqual(point_mul(N, G), INF)

    def test_point_mul_double(self):
        """2*G = G + G"""
        double = point_mul(2, G)
        added = point_add(G, G)
        self.assertEqual(double, added)

    def test_point_mul_associative(self):
        """(a*b)*G = a*(b*G)"""
        a, b = 137, 251
        lhs = point_mul((a * b) % N, G)
        rhs = point_mul(a, point_mul(b, G))
        self.assertEqual(lhs, rhs)

    def test_point_neg(self):
        """Negation flips y-coordinate mod P"""
        x, y = G
        nx, ny = point_neg(G)
        self.assertEqual(x, nx)
        self.assertEqual((y + ny) % P, 0)

    def test_point_neg_inf(self):
        self.assertEqual(point_neg(INF), INF)


# ============================================================
# Key compression / decompression
# ============================================================

class TestPubkeyCompression(unittest.TestCase):

    def test_compress_decompress_roundtrip(self):
        """compress then decompress recovers the original point"""
        for k in [1, 2, 42, 999999, N - 1]:
            point = point_mul(k, G)
            compressed = compress_pubkey(point)
            self.assertEqual(len(compressed), 33)
            recovered = decompress_pubkey(compressed)
            self.assertEqual(recovered, point)

    def test_compress_prefix(self):
        """Even y -> 0x02, odd y -> 0x03"""
        for k in range(1, 20):
            point = point_mul(k, G)
            compressed = compress_pubkey(point)
            expected_prefix = 0x02 if point[1] % 2 == 0 else 0x03
            self.assertEqual(compressed[0], expected_prefix)

    def test_compress_inf_raises(self):
        with self.assertRaises(ValueError):
            compress_pubkey(INF)

    def test_decompress_invalid_length(self):
        with self.assertRaises(ValueError):
            decompress_pubkey(b'\x02' + b'\x00' * 31)

    def test_decompress_invalid_prefix(self):
        data = b'\x04' + G[0].to_bytes(32, 'big')
        with self.assertRaises(ValueError):
            decompress_pubkey(data)


# ============================================================
# ECDSA sign / verify / recover
# ============================================================

class TestECDSA(unittest.TestCase):

    def setUp(self):
        self.privkey = 0xDEADBEEF
        self.pubkey = point_mul(self.privkey, G)
        self.z = int.from_bytes(sha256(b"test message for QSB"), 'big')

    def test_sign_verify(self):
        """Sign then verify succeeds"""
        r, s = ecdsa_sign(self.privkey, self.z)
        self.assertTrue(ecdsa_verify(self.pubkey, self.z, r, s))

    def test_verify_wrong_message(self):
        """Verify fails on different message hash"""
        r, s = ecdsa_sign(self.privkey, self.z)
        wrong_z = int.from_bytes(sha256(b"wrong message"), 'big')
        self.assertFalse(ecdsa_verify(self.pubkey, wrong_z, r, s))

    def test_verify_wrong_key(self):
        """Verify fails with wrong public key"""
        r, s = ecdsa_sign(self.privkey, self.z)
        wrong_pub = point_mul(0xCAFE, G)
        self.assertFalse(ecdsa_verify(wrong_pub, self.z, r, s))

    def test_low_s_normalization(self):
        """Signatures always have s <= N/2"""
        for _ in range(10):
            r, s = ecdsa_sign(self.privkey, self.z)
            self.assertLessEqual(s, N // 2)

    def test_sign_with_k(self):
        """Deterministic signing with known k"""
        k = 42
        r, s = ecdsa_sign_with_k(self.privkey, self.z, k)
        self.assertTrue(ecdsa_verify(self.pubkey, self.z, r, s))
        # Same k -> same r (r = (k*G).x mod N)
        R = point_mul(k, G)
        self.assertEqual(r, R[0] % N)

    def test_recover_pubkey(self):
        """Key recovery returns the correct public key"""
        r, s = ecdsa_sign(self.privkey, self.z)
        for flag in [0, 1]:
            Q = ecdsa_recover(r, s, self.z, recovery_flag=flag)
            if Q == self.pubkey:
                break
        self.assertEqual(Q, self.pubkey)

    def test_recover_both_flags(self):
        """Both recovery flags produce valid public keys (if they exist)"""
        r, s = ecdsa_sign(self.privkey, self.z)
        for flag in [0, 1]:
            Q = ecdsa_recover(r, s, self.z, recovery_flag=flag)
            if Q is not None:
                self.assertTrue(ecdsa_verify(Q, self.z, r, s))

    def test_verify_boundary_values(self):
        """Verify rejects r or s outside [1, N-1]"""
        self.assertFalse(ecdsa_verify(self.pubkey, self.z, 0, 1))
        self.assertFalse(ecdsa_verify(self.pubkey, self.z, N, 1))
        self.assertFalse(ecdsa_verify(self.pubkey, self.z, 1, 0))
        self.assertFalse(ecdsa_verify(self.pubkey, self.z, 1, N))


# ============================================================
# DER encoding / validation / parsing
# ============================================================

class TestDER(unittest.TestCase):

    def test_encode_valid(self):
        """Encoded signature passes validation"""
        sig = encode_der_sig(12345, 67890)
        self.assertTrue(is_valid_der_sig(sig))

    def test_encode_large_values(self):
        """Full-size r, s encode correctly"""
        r = N - 1
        s = N // 2
        sig = encode_der_sig(r, s)
        self.assertTrue(is_valid_der_sig(sig))

    def test_minimum_9_byte_sig(self):
        """9-byte sig with r=1, s=1 is valid DER"""
        sig = _encode_9byte_sig(1, 1, sighash=0x03)
        self.assertEqual(len(sig), 9)
        self.assertTrue(is_valid_der_sig(sig))

    def test_parse_der_roundtrip(self):
        """parse_der recovers original (r, s) from encode_der_sig"""
        test_cases = [
            (1, 1),
            (12345, 67890),
            (N - 1, N // 2),
            (0x7F, 0x7F),
            (0x80, 0x80),
        ]
        for r, s in test_cases:
            sig = encode_der_sig(r, s)
            pr, ps = parse_der(sig)
            self.assertEqual(pr, r, "r mismatch for ({}, {})".format(r, s))
            self.assertEqual(ps, s, "s mismatch for ({}, {})".format(r, s))

    def test_parse_der_invalid(self):
        """parse_der returns (None, None) for invalid data"""
        r, s = parse_der(b'\x00' * 20)
        self.assertIsNone(r)
        self.assertIsNone(s)

    def test_parse_der_too_short(self):
        r, s = parse_der(b'\x30\x06\x02\x01')
        self.assertIsNone(r)

    def test_valid_der_rejects_wrong_tag(self):
        """First byte must be 0x30"""
        sig = encode_der_sig(1, 1)
        bad = b'\x31' + sig[1:]
        self.assertFalse(is_valid_der_sig(bad))

    def test_valid_der_rejects_wrong_length(self):
        """Length byte must match actual content"""
        sig = encode_der_sig(1, 1)
        bad = sig[:1] + bytes([sig[1] + 1]) + sig[2:]
        self.assertFalse(is_valid_der_sig(bad))

    def test_valid_der_rejects_negative_int(self):
        """Leading 0x80 bit means negative -- invalid"""
        bad = bytes([0x30, 0x06, 0x02, 0x01, 0x80, 0x02, 0x01, 0x01, 0x01])
        self.assertFalse(is_valid_der_sig(bad))

    def test_valid_der_rejects_unnecessary_padding(self):
        """Leading 0x00 only allowed if next byte has high bit"""
        bad = bytes([0x30, 0x07, 0x02, 0x02, 0x00, 0x01, 0x02, 0x01, 0x01, 0x01])
        self.assertFalse(is_valid_der_sig(bad))

    def test_int_to_der_int_padding(self):
        """Values >= 0x80 get a leading zero byte"""
        b = int_to_der_int(0x80)
        self.assertEqual(b[0], 0x00)
        self.assertEqual(b[1], 0x80)

    def test_int_to_der_int_no_padding(self):
        """Values < 0x80 are a single byte"""
        b = int_to_der_int(0x7F)
        self.assertEqual(len(b), 1)
        self.assertEqual(b[0], 0x7F)

    def test_valid_small_r_values(self):
        """All returned r values are valid x-coordinates on secp256k1"""
        valid = _valid_small_r_values()
        self.assertGreater(len(valid), 0)
        for r in valid:
            self.assertGreaterEqual(r, 1)
            self.assertLessEqual(r, 127)
            y_sq = (pow(r, 3, P) + 7) % P
            y = pow(y_sq, (P + 1) // 4, P)
            self.assertEqual(pow(y, 2, P), y_sq,
                             "r={} is not a valid x-coordinate".format(r))


# ============================================================
# Hashing
# ============================================================

class TestHashing(unittest.TestCase):

    def test_sha256_known_vector(self):
        """SHA-256 of empty string matches known hash"""
        expected = bytes.fromhex(
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855"
        )
        self.assertEqual(sha256(b""), expected)

    def test_sha256d(self):
        """Double SHA-256 = SHA-256(SHA-256(x))"""
        data = b"test"
        self.assertEqual(sha256d(data), sha256(sha256(data)))

    def test_ripemd160_known_vector(self):
        """RIPEMD-160 of empty string"""
        expected = bytes.fromhex("9c1185a5c5e9fc54612808977ee8f548b2258d31")
        self.assertEqual(ripemd160(b""), expected)

    def test_hash160(self):
        """HASH160 = RIPEMD-160(SHA-256(x))"""
        data = b"hello"
        self.assertEqual(hash160(data), ripemd160(sha256(data)))


# ============================================================
# Transaction serialization
# ============================================================

class TestTransaction(unittest.TestCase):

    def test_serialize_varint(self):
        """Varint encoding for various sizes"""
        self.assertEqual(serialize_varint(0), b'\x00')
        self.assertEqual(serialize_varint(0xfc), b'\xfc')
        self.assertEqual(serialize_varint(0xfd), b'\xfd\xfd\x00')
        self.assertEqual(serialize_varint(0xffff), b'\xfd\xff\xff')
        self.assertEqual(serialize_varint(0x10000), b'\xfe\x00\x00\x01\x00')

    def test_transaction_deterministic(self):
        """Same inputs produce identical serialization"""
        def make_tx():
            tx = Transaction(version=1, locktime=0)
            tx.add_input(TxIn(b'\x01' * 32, 0, b'', 0xffffffff))
            tx.add_output(TxOut(50000, b'\x76\xa9\x14' + b'\x00' * 20 + b'\x88\xac'))
            return tx.serialize()
        self.assertEqual(make_tx(), make_tx())

    def test_sighash_all(self):
        """SIGHASH_ALL produces consistent hash"""
        tx = Transaction(version=1, locktime=0)
        tx.add_input(TxIn(b'\x01' * 32, 0, b'', 0xffffffff))
        tx.add_output(TxOut(50000, b'\x00' * 25))
        script_code = b'\x76\xa9\x14' + b'\x00' * 20 + b'\x88\xac'
        z1 = tx.sighash(0, script_code, sighash_type=0x01)
        z2 = tx.sighash(0, script_code, sighash_type=0x01)
        self.assertEqual(z1, z2)
        self.assertIsInstance(z1, int)
        self.assertGreater(z1, 0)

    def test_sighash_changes_with_locktime(self):
        """Different locktime produces different sighash"""
        script_code = b'\x00' * 25
        tx1 = Transaction(version=1, locktime=0)
        tx1.add_input(TxIn(b'\x01' * 32, 0, b'', 0xffffffff))
        tx1.add_output(TxOut(50000, b'\x00' * 25))
        tx2 = Transaction(version=1, locktime=1)
        tx2.add_input(TxIn(b'\x01' * 32, 0, b'', 0xffffffff))
        tx2.add_output(TxOut(50000, b'\x00' * 25))
        self.assertNotEqual(
            tx1.sighash(0, script_code),
            tx2.sighash(0, script_code),
        )

    def test_sighash_single_bug(self):
        """SIGHASH_SINGLE with input_index >= len(outputs) returns 1"""
        tx = Transaction(version=1, locktime=0)
        tx.add_input(TxIn(b'\x00' * 32, 0, b'', 0xffffffff))
        tx.add_input(TxIn(b'\x01' * 32, 0, b'', 0xffffffff))
        tx.add_output(TxOut(50000, b'\x00' * 25))
        # Input 1, only 1 output -> SIGHASH_SINGLE bug
        z = tx.sighash(1, b'\x00' * 25, sighash_type=0x03)
        self.assertEqual(z, 1)


# ============================================================
# Script operations
# ============================================================

class TestScript(unittest.TestCase):

    def test_push_data_small(self):
        """Data <= 75 bytes uses direct length prefix"""
        data = b'\xaa\xbb'
        result = push_data(data)
        self.assertEqual(result, bytes([2, 0xaa, 0xbb]))

    def test_push_data_medium(self):
        """Data 76-255 bytes uses OP_PUSHDATA1"""
        data = b'\x00' * 100
        result = push_data(data)
        self.assertEqual(result[0], 0x4c)  # OP_PUSHDATA1
        self.assertEqual(result[1], 100)

    def test_push_data_large(self):
        """Data 256-65535 bytes uses OP_PUSHDATA2"""
        data = b'\x00' * 300
        result = push_data(data)
        self.assertEqual(result[0], 0x4d)  # OP_PUSHDATA2
        size = struct.unpack('<H', result[1:3])[0]
        self.assertEqual(size, 300)

    def test_push_data_empty(self):
        """Empty data produces OP_0"""
        self.assertEqual(push_data(b''), bytes([0x00]))

    def test_push_number_small(self):
        """Numbers 1-16 use OP_1..OP_16"""
        self.assertEqual(push_number(0), bytes([0x00]))
        self.assertEqual(push_number(1), bytes([0x51]))  # OP_1
        self.assertEqual(push_number(16), bytes([0x60]))  # OP_16

    def test_find_and_delete(self):
        """Removes all occurrences of push_data(pattern)"""
        a = push_data(b'\xaa\xbb')
        b_data = push_data(b'\xcc\xdd')
        script = a + b_data + a
        result = find_and_delete(script, b'\xaa\xbb')
        self.assertEqual(result, b_data)

    def test_find_and_delete_no_match(self):
        """No match leaves script unchanged"""
        script = push_data(b'\xaa\xbb')
        result = find_and_delete(script, b'\xff\xff')
        self.assertEqual(result, script)

    def test_find_and_delete_empty_script(self):
        """Empty script stays empty"""
        self.assertEqual(find_and_delete(b'', b'\xaa'), b'')


# ============================================================
# QSB script builder
# ============================================================

class TestQSBScriptBuilder(unittest.TestCase):

    def setUp(self):
        """Build a small config for fast tests"""
        import random
        random.seed(42)
        orig = os.urandom
        os.urandom = lambda n: random.randbytes(n)

        self.builder = QSBScriptBuilder(n=10, t1_signed=2, t2_signed=2)
        self.builder.generate_keys()

        os.urandom = orig

    def test_hors_secrets_count(self):
        """Each round has n HORS secrets"""
        for r in range(2):
            self.assertEqual(len(self.builder.hors_secrets[r]), 10)
            self.assertEqual(len(self.builder.hors_commitments[r]), 10)

    def test_hors_commitment_is_hash160(self):
        """commitment = HASH160(secret)"""
        for r in range(2):
            for i in range(10):
                expected = hash160(self.builder.hors_secrets[r][i])
                self.assertEqual(self.builder.hors_commitments[r][i], expected)

    def test_dummy_sigs_are_9_bytes(self):
        """All dummy sigs are exactly 9 bytes and valid DER"""
        for r in range(2):
            for sig in self.builder.dummy_sigs[r]:
                self.assertEqual(len(sig), 9)
                self.assertTrue(is_valid_der_sig(sig))

    def test_dummy_sigs_unique_per_round(self):
        """No duplicate dummy sigs within a round"""
        for r in range(2):
            sigs = self.builder.dummy_sigs[r]
            self.assertEqual(len(sigs), len(set(map(bytes, sigs))))

    def test_full_script_builds(self):
        """Full script builds without error and is non-empty"""
        pin_sig = encode_der_sig(111, 222, sighash=0x01)
        r1_sig = encode_der_sig(333, 444, sighash=0x01)
        r2_sig = encode_der_sig(555, 666, sighash=0x01)
        script = self.builder.build_full_script(pin_sig, r1_sig, r2_sig)
        self.assertGreater(len(script), 100)

    def test_full_script_deterministic(self):
        """Same builder produces same script"""
        pin_sig = encode_der_sig(111, 222, sighash=0x01)
        r1_sig = encode_der_sig(333, 444, sighash=0x01)
        r2_sig = encode_der_sig(555, 666, sighash=0x01)
        s1 = self.builder.build_full_script(pin_sig, r1_sig, r2_sig)
        s2 = self.builder.build_full_script(pin_sig, r1_sig, r2_sig)
        self.assertEqual(s1, s2)

    def test_config_a_within_limits(self):
        """Config A (n=150) stays within Bitcoin Script limits"""
        import random
        random.seed(0)
        orig = os.urandom
        os.urandom = lambda n: random.randbytes(n)

        builder = QSBScriptBuilder(n=150, t1_signed=8, t1_bonus=1,
                                    t2_signed=7, t2_bonus=2)
        builder.generate_keys()
        os.urandom = orig

        pin_sig = encode_der_sig(111, 222, sighash=0x01)
        r1_sig = encode_der_sig(333, 444, sighash=0x01)
        r2_sig = encode_der_sig(555, 666, sighash=0x01)
        script = builder.build_full_script(pin_sig, r1_sig, r2_sig)

        self.assertLessEqual(len(script), 10000,
                             "Script {}B exceeds 10,000B limit".format(len(script)))


# ============================================================
# Integration: sign -> DER -> parse -> recover roundtrip
# ============================================================

class TestIntegration(unittest.TestCase):

    def test_sign_encode_parse_verify(self):
        """Full cycle: sign, DER encode, parse, verify"""
        privkey = 0x1234ABCD
        pubkey = point_mul(privkey, G)
        z = int.from_bytes(sha256(b"integration test"), 'big')

        r, s = ecdsa_sign(privkey, z)
        sig = encode_der_sig(r, s)
        self.assertTrue(is_valid_der_sig(sig))

        pr, ps = parse_der(sig)
        self.assertEqual(pr, r)
        self.assertEqual(ps, s)
        self.assertTrue(ecdsa_verify(pubkey, z, pr, ps))

    def test_sighash_single_bug_recovery(self):
        """
        SIGHASH_SINGLE bug (z=1): sign with known key,
        recover pubkey from sig with z=1.
        This is the mechanism QSB uses for dummy signatures.
        """
        privkey = 0xFEEDFACE
        pubkey = point_mul(privkey, G)
        z = 1  # SIGHASH_SINGLE bug

        r, s = ecdsa_sign(privkey, z)
        self.assertTrue(ecdsa_verify(pubkey, z, r, s))

        # Recover
        for flag in [0, 1]:
            Q = ecdsa_recover(r, s, z, recovery_flag=flag)
            if Q == pubkey:
                break
        self.assertEqual(Q, pubkey)

    def test_find_and_delete_affects_sighash(self):
        """
        Removing a dummy sig from the script changes the sighash.
        This is the core mechanism of QSB digest rounds.
        """
        tx = Transaction(version=1, locktime=42)
        tx.add_input(TxIn(b'\xAA' * 32, 0, b'', 0xfffffffe))
        tx.add_output(TxOut(10000, b'\x00' * 25))

        dummy = b'\x30\x06\x02\x01\x01\x02\x01\x01\x03'  # 9-byte sig
        script_with = push_data(b'\xff') + push_data(dummy) + push_data(b'\xee')
        script_without = find_and_delete(script_with, dummy)

        z_with = tx.sighash(0, script_with)
        z_without = tx.sighash(0, script_without)

        self.assertNotEqual(script_with, script_without)
        self.assertNotEqual(z_with, z_without)

    def test_ripemd160_der_probability(self):
        """
        Hash-to-signature puzzle: RIPEMD-160(pubkey) is almost never valid DER.
        Sanity check for search difficulty (~2^-46).
        """
        hits = 0
        trials = 10000
        for i in range(trials):
            data = os.urandom(33)
            h = ripemd160(data)
            if is_valid_der_sig(h):
                hits += 1
        # At ~2^-46 probability, we expect 0 hits in 10K trials
        self.assertEqual(hits, 0,
                         "Got {} DER hits in {} trials (expected 0)".format(hits, trials))


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    unittest.main()
