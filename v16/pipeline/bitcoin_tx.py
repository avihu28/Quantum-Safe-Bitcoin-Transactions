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
                # SIGHASH_SINGLE bug: return 1
                return 1
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

    def build_round_script(self, round_idx, sig_nonce_bytes):
        """Build a single round's script"""
        t_signed = self.t1_signed if round_idx == 0 else self.t2_signed
        t_bonus = self.t1_bonus if round_idx == 0 else self.t2_bonus
        t_total = t_signed + t_bonus
        
        script = b''
        
        # Push n HORS commitments (reversed order)
        for i in range(self.n - 1, -1, -1):
            script += push_data(self.hors_commitments[round_idx][i])
        
        # Push n dummy sigs (reversed order)
        for i in range(self.n - 1, -1, -1):
            script += push_data(self.dummy_sigs[round_idx][i])
        
        # OP_0 (CHECKMULTISIG dummy)
        script += bytes([OP_0])
        
        # sig_nonce (hardcoded, SIGHASH_ALL)
        script += push_data(sig_nonce_bytes)
        
        # Signed selections (9 ops each)
        for i in range(t_signed):
            idx_pos = 2 * self.n + 1 - i
            sanitize = self.n - i
            preimage_pos = 2 * self.n + 1 + t_total - 2 * i
            
            script += push_number(idx_pos)
            script += bytes([OP_ROLL])
            script += push_number(sanitize)
            script += bytes([OP_MIN])
            script += bytes([OP_DUP])
            script += push_number(self.n + 1)
            script += bytes([OP_ADD])
            script += bytes([OP_ROLL])
            script += push_number(preimage_pos)
            script += bytes([OP_ROLL])
            script += bytes([OP_HASH160])
            script += bytes([OP_EQUALVERIFY])
            script += bytes([OP_ROLL])
        
        # Bonus selections (3 ops each)
        for i in range(t_bonus):
            j = t_signed + i
            idx_pos = 2 * self.n + 1 - j
            sanitize = self.n - j
            
            script += push_number(idx_pos)
            script += bytes([OP_ROLL])
            script += push_number(sanitize)
            script += bytes([OP_MIN])
            script += bytes([OP_ROLL])
        
        # Puzzle: ROLL key_nonce + DUP + SHA256 + CHECKSIGVERIFY
        puzzle_pos = 2 * self.n + 2  # adjusted
        
        if self.hash_mode == 'sha256_double':
            # Double hash mode: hash_choice is in witness adjacent to key_nonce
            # After selection loop, hash_choice is at puzzle_pos, key_nonce at puzzle_pos+1
            # ROLL hash_choice first, then key_nonce
            script += push_number(puzzle_pos + 1)
            script += bytes([OP_ROLL])             # get hash_choice
            script += push_number(puzzle_pos + 1)
            script += bytes([OP_ROLL])             # get key_nonce
            script += bytes([OP_DUP])              # copy key_nonce for CHECKMULTISIG
            # Stack: ... hash_choice key_nonce key_nonce_copy
            # OP_ROT brings hash_choice to top
            OP_ROT = 0x7b
            script += bytes([OP_ROT])              # [kn, kn_copy, hash_choice]
            script += bytes([OP_IF])
            script += bytes([OP_SHA256_OP])         # pre-hash if bit=1
            script += bytes([OP_ENDIF])
            script += bytes([OP_SHA256_OP])         # always hash
            # Adjust key_puzzle position (+1 for hash_choice witness element)
            puzzle_key_pos = puzzle_pos + 1
        else:
            # Single hash mode (sha256 or ripemd160)
            script += push_number(puzzle_pos)
            script += bytes([OP_ROLL])
            script += bytes([OP_DUP])
            if self.hash_mode == 'ripemd160':
                script += bytes([OP_RIPEMD160_OP])
            else:
                script += bytes([OP_SHA256_OP])
            puzzle_key_pos = puzzle_pos
        
        # CHECKSIGVERIFY for sig_puzzle
        script += push_number(puzzle_key_pos)
        script += bytes([OP_ROLL])
        script += bytes([OP_CHECKSIGVERIFY])
        
        # CHECKMULTISIG (t+1)-of-(t+1)
        m = t_total + 1
        script += push_number(m)
        script += bytes([OP_2])
        script += bytes([OP_ROLL])
        
        # Roll t dummy pubkeys
        cms_roll_pos = 2 * self.n + 3 + (1 if self.hash_mode == 'sha256_double' else 0)
        for j in range(t_total):
            script += push_number(cms_roll_pos)
            script += bytes([OP_ROLL])
        
        script += push_number(m)  # N = t+1
        script += bytes([OP_CHECKMULTISIG])
        
        return script
    
    def build_full_script(self, pin_sig, round1_sig, round2_sig):
        """Build the complete locking script"""
        script = self.build_pinning_script(pin_sig)
        script += self.build_round_script(0, round1_sig)
        script += self.build_round_script(1, round2_sig)
        return script
    
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
