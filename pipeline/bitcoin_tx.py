"""
Bitcoin transaction construction and legacy sighash computation for QSB.
Handles: serialization, FindAndDelete, sighash, script building.
"""

import struct
import hashlib
from secp256k1 import (
    sha256d, ripemd160, hash160,
    compress_pubkey, point_mul, G, N,
    ecdsa_sign, ecdsa_recover, encode_der_sig, is_valid_der_sig,
    modinv, int_to_der_int
)
import os


# ============================================================
# Script opcodes
# ============================================================

OP_0 = 0x00
OP_PUSHDATA1 = 0x4c
OP_PUSHDATA2 = 0x4d
OP_1 = 0x51
OP_2 = 0x52
OP_16 = 0x60
OP_DUP = 0x76
OP_SWAP = 0x7c
OP_2ROLL = 0x72  # OP_2 (0x52) OP_ROLL actually = OP_2 then OP_ROLL
OP_ROLL = 0x7a
OP_OVER = 0x78
OP_MIN = 0xa3
OP_ADD = 0x93
OP_CHECKSIG = 0xac
OP_CHECKSIGVERIFY = 0xad
OP_CHECKMULTISIG = 0xae
OP_RIPEMD160_OP = 0xa6
OP_SHA256_OP = 0xa8
OP_HASH160 = 0xa9
OP_EQUALVERIFY = 0x88
OP_IF = 0x63
OP_ENDIF = 0x68


def push_data(data):
    """Create push opcode(s) for arbitrary data"""
    n = len(data)
    if n == 0:
        return bytes([OP_0])
    elif n <= 75:
        return bytes([n]) + data
    elif n <= 255:
        return bytes([OP_PUSHDATA1, n]) + data
    elif n <= 65535:
        return bytes([OP_PUSHDATA2]) + struct.pack('<H', n) + data
    else:
        raise ValueError(f"Data too large: {n}")

def push_number(n):
    """Push a small number onto the stack"""
    if n == 0:
        return bytes([OP_0])
    elif 1 <= n <= 16:
        return bytes([OP_1 + n - 1])
    else:
        # Encode as minimal push
        if n < 0:
            raise ValueError("Negative numbers not supported here")
        b = n.to_bytes((n.bit_length() + 7) // 8, 'little')
        if b[-1] & 0x80:
            b += b'\x00'
        return push_data(b)


# ============================================================
# Transaction structure
# ============================================================

class TxIn:
    def __init__(self, txid, vout, script_sig=b'', sequence=0xffffffff):
        self.txid = txid  # 32 bytes, internal byte order
        self.vout = vout
        self.script_sig = script_sig
        self.sequence = sequence

    def serialize(self):
        return (
            self.txid +
            struct.pack('<I', self.vout) +
            serialize_varint(len(self.script_sig)) +
            self.script_sig +
            struct.pack('<I', self.sequence)
        )

class TxOut:
    def __init__(self, value, script_pubkey):
        self.value = value
        self.script_pubkey = script_pubkey

    def serialize(self):
        return (
            struct.pack('<q', self.value) +
            serialize_varint(len(self.script_pubkey)) +
            self.script_pubkey
        )

class Transaction:
    def __init__(self, version=1, locktime=0):
        self.version = version
        self.inputs = []
        self.outputs = []
        self.locktime = locktime

    def add_input(self, txin):
        self.inputs.append(txin)

    def add_output(self, txout):
        self.outputs.append(txout)

    def serialize(self):
        result = struct.pack('<I', self.version)
        result += serialize_varint(len(self.inputs))
        for inp in self.inputs:
            result += inp.serialize()
        result += serialize_varint(len(self.outputs))
        for out in self.outputs:
            result += out.serialize()
        result += struct.pack('<I', self.locktime)
        return result

    def sighash(self, input_index, script_code, sighash_type=0x01):
        """
        Compute legacy sighash for input at input_index.
        script_code should already have FindAndDelete applied.
        """
        # Copy transaction
        tx_copy = Transaction(self.version, self.locktime)

        for i, inp in enumerate(self.inputs):
            if i == input_index:
                new_inp = TxIn(inp.txid, inp.vout, script_code, inp.sequence)
            else:
                new_inp = TxIn(inp.txid, inp.vout, b'', inp.sequence)

            # Handle SIGHASH_ANYONECANPAY
            if sighash_type & 0x80:
                if i != input_index:
                    continue

            # Handle SIGHASH_NONE / SIGHASH_SINGLE sequence
            base = sighash_type & 0x1f
            if base == 0x02 or base == 0x03:  # NONE or SINGLE
                if i != input_index:
                    new_inp.sequence = 0

            tx_copy.add_input(new_inp)

        # Handle outputs
        base = sighash_type & 0x1f
        if base == 0x02:  # SIGHASH_NONE
            pass  # no outputs
        elif base == 0x03:  # SIGHASH_SINGLE
            if input_index >= len(self.outputs):
                # SIGHASH_SINGLE bug. Bitcoin Core returns uint256::ONE, whose
                # raw little-endian bytes (01 00 .. 00) are fed to secp256k1's
                # msg32 and read BIG-ENDIAN — i.e. the message scalar is 2**248,
                return 1 << 248
            for i in range(input_index + 1):
                if i < input_index:
                    tx_copy.add_output(TxOut(-1, b''))
                else:
                    tx_copy.add_output(self.outputs[i])
        else:  # SIGHASH_ALL
            for out in self.outputs:
                tx_copy.add_output(out)

        serialized = tx_copy.serialize() + struct.pack('<I', sighash_type)
        return int.from_bytes(sha256d(serialized), 'big')


def serialize_varint(n):
    if n < 0xfd:
        return bytes([n])
    elif n <= 0xffff:
        return b'\xfd' + struct.pack('<H', n)
    elif n <= 0xffffffff:
        return b'\xfe' + struct.pack('<I', n)
    else:
        return b'\xff' + struct.pack('<Q', n)


# ============================================================
# FindAndDelete
# ============================================================

def find_and_delete(script, sig_data):
    """
    Remove all occurrences of push_data(sig_data) from script.
    This is the FindAndDelete operation applied before sighash computation.
    """
    pattern = push_data(sig_data)
    result = b''
    i = 0
    while i <= len(script) - len(pattern):
        if script[i:i+len(pattern)] == pattern:
            i += len(pattern)
        else:
            result += bytes([script[i]])
            i += 1
    result += script[i:]
    return result


# ============================================================
# 9-byte minimum DER signatures
# ============================================================

# secp256k1 curve field prime
_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
_B = 7

def _valid_small_r_values():
    """
    Find r values in [1, 127] that are valid x-coordinates on secp256k1.
    y^2 = x^3 + 7 mod P must have a solution.
    Returns a list of valid r values.
    """
    valid = []
    for r in range(1, 128):
        y_sq = (pow(r, 3, _P) + _B) % _P
        y = pow(y_sq, (_P + 1) // 4, _P)
        if pow(y, 2, _P) == y_sq:
            valid.append(r)
    return valid

def _encode_9byte_sig(r, s, sighash=0x03):
    """
    Encode a 9-byte minimum DER signature.
    r and s must be in [1, 127].
    Format: 30 06 02 01 <r> 02 01 <s> <sighash>
    """
    assert 1 <= r <= 127, f"r={r} out of range"
    assert 1 <= s <= 127, f"s={s} out of range"
    return bytes([0x30, 0x06, 0x02, 0x01, r, 0x02, 0x01, s, sighash])


# ============================================================
# Stack model — used to compute OP_ROLL positions from the ACTUAL modeled
# stack instead of hand-derived formulas. The hand-derived formulas in the
# original build_round_script were wrong (off-by-one: they counted sig_nonce
# above the dummy block but forgot the OP_0 CHECKMULTISIG dummy, and used a
# fixed commitment gap n+1 that must actually shrink to n+1-i per iteration).
# Threading this model through the whole script makes every position correct
# by construction and prevents cross-round drift (round 2 sits one item higher
# than round 1 because round 1 leaves a CHECKMULTISIG result on the stack).
# ============================================================

class _StackModel:
    """Models the script stack (index -1 = top). Tokens are opaque labels."""
    def __init__(self):
        self.s = []                       # bottom .. top

    def push(self, tok):
        self.s.append(tok)

    def depth(self, tok):
        """0 = top."""
        for k in range(len(self.s) - 1, -1, -1):
            if self.s[k] == tok:
                return len(self.s) - 1 - k
        raise KeyError(tok)

    def deepest_dummy(self, rnd):
        """Depth of the deepest remaining dummy of round `rnd` (for OP_MIN sanitize)."""
        best = -1
        for k in range(len(self.s)):
            t = self.s[k]
            if isinstance(t, tuple) and len(t) == 3 and t[0] == 'D' and t[1] == rnd:
                best = max(best, len(self.s) - 1 - k)
        return best

    def roll(self, d):
        """OP_ROLL semantics (the count has already been popped): move the item
        at depth d to the top."""
        i = len(self.s) - 1 - d
        self.s.append(self.s.pop(i))

    def pop(self, k=1):
        for _ in range(k):
            self.s.pop()


# ============================================================
# QSB Script Builder
# ============================================================

class QSBScriptBuilder:
    """Build the locking script for QSB"""
    
    def __init__(self, n=150, t1_signed=8, t1_bonus=0, t2_signed=8, t2_bonus=0, hash_mode='ripemd160'):
        self.n = n
        self.t1_signed = t1_signed
        self.t1_bonus = t1_bonus
        self.t2_signed = t2_signed
        self.t2_bonus = t2_bonus
        self.hash_mode = hash_mode  # 'ripemd160', 'sha256', 'sha256_double'
        
        # Generate HORS secrets and commitments
        self.hors_secrets = []  # [round][index] = 20 bytes
        self.hors_commitments = []  # [round][index] = 20 bytes

        # Generate dummy sig data
        self.dummy_sigs = []  # [round][index] = 9 bytes

        # Pinning sig (will be set)
        self.pin_sig_nonce = None
        self.pin_privkey = None

        # Round sigs
        self.round_sig_nonce = []  # [round] = sig bytes
        self.round_privkeys = []  # [round] = privkey int

    def generate_keys(self):
        """Generate all keys and commitments"""
        # Precompute valid small r values for 9-byte sigs
        small_r_values = _valid_small_r_values()

        for r in range(2):
            # HORS
            secrets_r = []
            commits_r = []
            for i in range(self.n):
                secret = os.urandom(20)
                commitment = hash160(secret)
                secrets_r.append(secret)
                commits_r.append(commitment)
            self.hors_secrets.append(secrets_r)
            self.hors_commitments.append(commits_r)

            # Dummy sigs: 9-byte minimum using SIGHASH_SINGLE bug (z=1)
            # Format: 30 06 02 01 <r> 02 01 <s> 03
            # r must be a valid x-coordinate on secp256k1 (1-127)
            # s can be any value 1-127
            # Pubkey is recovered from (r, s, z=1)
            dummy_sigs_r = []
            for i in range(self.n):
                # Enumerate unique (r_val, s_val) pairs per round
                # Offset by round to avoid collisions between rounds
                pair_idx = i + r * self.n
                r_val = small_r_values[pair_idx % len(small_r_values)]
                s_val = 1 + (pair_idx // len(small_r_values)) % 127
                sig = _encode_9byte_sig(r_val, s_val, sighash=0x03)
                # Verify uniqueness within this round
                assert sig not in dummy_sigs_r, f"Duplicate sig at round {r}, index {i}"
                dummy_sigs_r.append(sig)
            self.dummy_sigs.append(dummy_sigs_r)

        # Pinning key (not used for signing — just a placeholder)
        self.pin_privkey = int.from_bytes(os.urandom(32), 'big') % N

        # Round keys (not used for signing)
        for r in range(2):
            privkey = int.from_bytes(os.urandom(32), 'big') % N
            self.round_privkeys.append(privkey)
    
    def _puzzle_hash_ops(self):
        """Return the hash opcodes for the puzzle, based on hash_mode.
        
        'ripemd160':     OP_RIPEMD160 (1 op)
        'sha256':        OP_SHA256 (1 op)  
        'sha256_double': OP_IF OP_SHA256 OP_ENDIF OP_SHA256 (+3 ops)
            bit=0: SHA256(key), bit=1: SHA256(SHA256(key))
        """
        if self.hash_mode == 'ripemd160':
            return bytes([OP_RIPEMD160_OP])
        elif self.hash_mode == 'sha256':
            return bytes([OP_SHA256_OP])
        elif self.hash_mode == 'sha256_double':
            # Witness provides a bit; if 1, pre-hash with SHA256
            # OP_IF OP_SHA256 OP_ENDIF then OP_SHA256 always
            return bytes([OP_IF, OP_SHA256_OP, OP_ENDIF, OP_SHA256_OP])
        else:
            raise ValueError(f"Unknown hash_mode: {self.hash_mode}")
    
    def _puzzle_hash_op_count(self):
        """Number of non-push opcodes added by the puzzle hash."""
        if self.hash_mode in ('ripemd160', 'sha256'):
            return 1
        elif self.hash_mode == 'sha256_double':
            return 4  # OP_IF + OP_SHA256 + OP_ENDIF + OP_SHA256
        else:
            raise ValueError(f"Unknown hash_mode: {self.hash_mode}")


    def build_pinning_script(self, sig_nonce_bytes):
        """Build the pinning section.
        
        Single hash (sha256/ripemd160): 5 ops.
          Witness: <key_puzzle> <key_nonce>
        
        Double hash (sha256_double): 9 ops.
          Witness: <key_puzzle> <hash_choice> <key_nonce>
          Stack after OVER+CHECKSIGVERIFY: key_puzzle hash_choice key_nonce
          SWAP brings hash_choice to top for IF.
        """
        script = push_data(sig_nonce_bytes)  # hardcoded sig
        script += bytes([OP_OVER])           # 1: copy key_nonce
        script += bytes([OP_CHECKSIGVERIFY]) # 2: verify (sig_nonce, key_nonce)
        if self.hash_mode == 'sha256_double':
            # Stack: key_puzzle hash_choice key_nonce
            script += bytes([OP_SWAP])       # 3: → key_puzzle key_nonce hash_choice
            script += bytes([OP_IF])         # 4: consume hash_choice
            script += bytes([OP_SHA256_OP])  # 5: pre-hash key_nonce (only if bit=1)
            script += bytes([OP_ENDIF])      # 6
            script += bytes([OP_SHA256_OP])  # 7: always hash → sig_puzzle
            script += bytes([OP_SWAP])       # 8: get key_puzzle on top
            script += bytes([OP_CHECKSIGVERIFY])  # 9: verify (sig_puzzle, key_puzzle)
        elif self.hash_mode == 'sha256':
            script += bytes([OP_SHA256_OP])      # 3: key_nonce → sig_puzzle
            script += bytes([OP_SWAP])           # 4: get key_puzzle on top
            script += bytes([OP_CHECKSIGVERIFY]) # 5: verify (sig_puzzle, key_puzzle)
        else:  # ripemd160
            script += bytes([OP_RIPEMD160_OP])   # 3: key_nonce → sig_puzzle
            script += bytes([OP_SWAP])           # 4: get key_puzzle on top
            script += bytes([OP_CHECKSIGVERIFY]) # 5: verify (sig_puzzle, key_puzzle)
        return script

    # ---- round parameter helpers ----
    def _round_t(self, round_idx):
        t_signed = self.t1_signed if round_idx == 0 else self.t2_signed
        t_bonus = self.t1_bonus if round_idx == 0 else self.t2_bonus
        return t_signed, t_bonus, t_signed + t_bonus

    def _canonical_subset(self, round_idx):
        """Any valid t_total distinct pool indices; used to build the
        (subset-independent) locking script."""
        _, _, t_total = self._round_t(round_idx)
        return list(range(t_total))

    def _seed_witness(self, m, round_idx):
        """Push this round's witness tokens (bottom..top): kp, kn, pubs, pres, idxs.
        Matches cmd_assemble's witness order."""
        t_signed, _, t_total = self._round_t(round_idx)
        R = round_idx
        m.push(('kp', R)); m.push(('kn', R))
        for j in range(t_total - 1, -1, -1): m.push(('pub', R, j))
        for j in range(t_signed - 1, -1, -1): m.push(('pre', R, j))
        for j in range(t_total - 1, -1, -1): m.push(('idx', R, j))

    def _model_pinning(self, m):
        """Model the pinning stage's net stack effect: it consumes key_puzzle and
        key_nonce (the pinning witness) and leaves nothing."""
        m.pop(2)

    def _emit_round(self, m, round_idx, sig_nonce_bytes, subset):
        """Emit one round's script bytes, computing every OP_ROLL position from
        the live stack model `m` (which must already hold this round's witness
        block and everything below it). Returns (script_bytes, idxvals) where
        idxvals are the per-selection witness index numbers for `subset`."""
        if self.hash_mode not in ('ripemd160', 'sha256'):
            raise ValueError("_emit_round supports single-hash modes only; "
                             "hash_mode=%r is not consensus-corrected" % self.hash_mode)
        hashop = OP_RIPEMD160_OP if self.hash_mode == 'ripemd160' else OP_SHA256_OP
        n = self.n
        R = round_idx
        t_signed, t_bonus, t_total = self._round_t(round_idx)
        out = bytearray()
        # --- round data pushes ---
        for p in range(n - 1, -1, -1):
            out += push_data(self.hors_commitments[R][p]); m.push(('C', R, p))
        for p in range(n - 1, -1, -1):
            out += push_data(self.dummy_sigs[R][p]); m.push(('D', R, p))
        out += bytes([OP_0]); m.push(('zero', R))
        out += push_data(sig_nonce_bytes); m.push(('signonce', R))
        idxvals = []
        # --- signed selections (9 opcodes each) ---
        for i in range(t_signed):
            tgt = subset[i]
            A = m.depth(('idx', R, i)); m.roll(A); m.pop(1); m.push(('iv', R, i))
            san = m.deepest_dummy(R)
            m.push(('iv', R, i))                                  # OP_DUP
            m.pop(1); dc = m.depth(('C', R, tgt)); m.roll(dc)     # OP_ADD + OP_ROLL (commitment)
            D = m.depth(('pre', R, i)); m.roll(D)                 # preimage fetch
            m.pop(2)                                              # OP_HASH160 + OP_EQUALVERIFY
            m.pop(1); dd = m.depth(('D', R, tgt)); m.roll(dd)     # OP_ROLL (dummy)
            gap = dc - dd; idxvals.append(dd)
            out += push_number(A) + bytes([OP_ROLL])
            out += push_number(san) + bytes([OP_MIN])
            out += bytes([OP_DUP])
            out += push_number(gap) + bytes([OP_ADD, OP_ROLL])
            out += push_number(D) + bytes([OP_ROLL])
            out += bytes([OP_HASH160, OP_EQUALVERIFY, OP_ROLL])
        # --- bonus selections (3 opcodes each) ---
        for bi in range(t_bonus):
            j = t_signed + bi; tgt = subset[j]
            A = m.depth(('idx', R, j)); m.roll(A); m.pop(1); m.push(('iv', R, j))
            san = m.deepest_dummy(R)
            m.pop(1); dd = m.depth(('D', R, tgt)); m.roll(dd); idxvals.append(dd)
            out += push_number(A) + bytes([OP_ROLL])
            out += push_number(san) + bytes([OP_MIN])
            out += bytes([OP_ROLL])
        # --- puzzle: ROLL key_nonce, DUP, <hashop>, ROLL key_puzzle, CHECKSIGVERIFY ---
        pk = m.depth(('kn', R)); m.roll(pk); m.push(('kn', R)); m.pop(1); m.push(('sp', R))
        pp = m.depth(('kp', R)); m.roll(pp); m.pop(2)
        out += push_number(pk) + bytes([OP_ROLL, OP_DUP, hashop])
        out += push_number(pp) + bytes([OP_ROLL, OP_CHECKSIGVERIFY])
        # --- CHECKMULTISIG (t+1)-of-(t+1): pubkeys = t_total dummy pubs + key_nonce,
        #     rolled so their order matches the gathered dummy-sig order. ---
        mval = t_total + 1
        out += push_number(mval); m.push(('M', R))
        for tok in [('kn', R)] + [('pub', R, j) for j in range(t_total)]:
            d = m.depth(tok); m.roll(d); out += push_number(d) + bytes([OP_ROLL])
        out += push_number(mval); m.push(('N', R))
        out += bytes([OP_CHECKMULTISIG])
        # CHECKMULTISIG pops: N-value + N pubkeys + M-value + M sigs + dummy
        m.pop(2 * (t_total + 1) + 3); m.push(('cms', R))
        return bytes(out), idxvals

    def build_round_script(self, round_idx, sig_nonce_bytes):
        """Build a single round's script IN ISOLATION (fresh stack, nothing below
        its own witness). Used for size/opcode benchmarking. For a consensus-valid
        two-round lock use build_full_script, which threads a shared model so
        round 2's positions account for round 1's leftover."""
        m = _StackModel()
        self._seed_witness(m, round_idx)
        script, _ = self._emit_round(m, round_idx, sig_nonce_bytes,
                                     self._canonical_subset(round_idx))
        return script

    def build_full_script(self, pin_sig, round1_sig, round2_sig):
        """Build the complete locking script with model-derived positions."""
        m = _StackModel()
        # full witness (bottom..top): round-2 block, round-1 block, pin key_puzzle, key_nonce
        self._seed_witness(m, 1)
        self._seed_witness(m, 0)
        m.push(('pin_kp',)); m.push(('pin_kn',))
        script = bytearray(self.build_pinning_script(pin_sig))
        self._model_pinning(m)
        s0, _ = self._emit_round(m, 0, round1_sig, self._canonical_subset(0))
        s1, _ = self._emit_round(m, 1, round2_sig, self._canonical_subset(1))
        return bytes(script) + s0 + s1

    def compute_witness_indices(self, subsets):
        """Given the chosen subsets {0:[...], 1:[...]}, return the witness index
        numbers {0:[...], 1:[...]} the corrected selection loop expects (the
        model-derived dummy depths). Replays the same model build_full_script uses."""
        m = _StackModel()
        self._seed_witness(m, 1)
        self._seed_witness(m, 0)
        m.push(('pin_kp',)); m.push(('pin_kn',))
        self._model_pinning(m)
        _, iv0 = self._emit_round(m, 0, b'\x30\x06\x02\x01\x01\x02\x01\x01\x01', subsets[0])
        _, iv1 = self._emit_round(m, 1, b'\x30\x06\x02\x01\x01\x02\x01\x01\x01', subsets[1])
        return {0: iv0, 1: iv1}

    @staticmethod
    def count_opcodes(script):
        """Count non-push opcodes (>= 0x60) in a script, matching Bitcoin Core's rule.
        
        Returns (total_nonpush_ops, details_dict).
        """
        i = 0
        count = 0
        while i < len(script):
            op = script[i]
            if op == 0:  # OP_0
                i += 1
            elif 1 <= op <= 75:  # direct push
                i += 1 + op
            elif op == 0x4c:  # OP_PUSHDATA1
                if i + 1 < len(script):
                    sz = script[i+1]
                    i += 2 + sz
                else:
                    i += 1
            elif op == 0x4d:  # OP_PUSHDATA2
                if i + 2 < len(script):
                    sz = script[i+1] | (script[i+2] << 8)
                    i += 3 + sz
                else:
                    i += 1
            else:
                if op >= 0x60:  # OP_16 and above count
                    count += 1
                i += 1
        return count
    
    def get_round_script_code(self, round_idx, sig_nonce_bytes, selected_dummy_sigs):
        """
        Get the scriptCode for a round after FindAndDelete removes
        the selected dummy signatures.
        """
        script = self.build_round_script(round_idx, sig_nonce_bytes)
        # FindAndDelete removes each selected dummy sig
        for sig in selected_dummy_sigs:
            script = find_and_delete(script, sig)
        return script


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    print("=== Transaction test ===")

    # Create a simple transaction
    tx = Transaction(version=1, locktime=0)
    tx.add_input(TxIn(b'\x00' * 32, 0, b'', 0xffffffff))
    tx.add_output(TxOut(50000, b'\x76\xa9' + b'\x14' + b'\x00' * 20 + b'\x88\xac'))

    serialized = tx.serialize()
    print(f"Tx serialized: {len(serialized)} bytes")

    # Test sighash
    z = tx.sighash(0, b'\x00' * 25, sighash_type=0x01)
    print(f"Sighash: {z:#066x}")

    # Test script builder
    print("\n=== Script builder test (small n=5, t=2) ===")
    builder = QSBScriptBuilder(n=5, t1_signed=2, t2_signed=2)
    builder.generate_keys()

    # Create dummy sig_nonce
    pin_sig = encode_der_sig(123456789, 987654321, sighash=0x01)
    r1_sig = encode_der_sig(111111111, 222222222, sighash=0x01)
    r2_sig = encode_der_sig(333333333, 444444444, sighash=0x01)

    full_script = builder.build_full_script(pin_sig, r1_sig, r2_sig)
    print(f"Full script: {len(full_script)} bytes")

    # Test FindAndDelete
    print("\n=== FindAndDelete test ===")
    test_script = push_data(b'\xaa\xbb') + push_data(b'\xcc\xdd') + push_data(b'\xaa\xbb')
    deleted = find_and_delete(test_script, b'\xaa\xbb')
    assert deleted == push_data(b'\xcc\xdd'), "FindAndDelete failed"
    print("[OK] FindAndDelete")

    print("\n[ALL OK]")
