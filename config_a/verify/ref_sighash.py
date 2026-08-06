"""
Reference legacy sighash implementation.

Strictly follows Bitcoin Core's SignatureHash() in src/script/interpreter.cpp,
which implements the pre-segwit (BIP143 predecessor) sighash algorithm.

This is a transparent, side-by-side implementation meant to be easy to audit
against the Bitcoin Core source. It is used to cross-check our production
pipeline's sighash function.
"""

import hashlib
import struct


SIGHASH_ALL = 0x01
SIGHASH_NONE = 0x02
SIGHASH_SINGLE = 0x03
SIGHASH_ANYONECANPAY = 0x80


def dsha256(b):
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


def ser_compact(n):
    if n < 0xfd:
        return bytes([n])
    if n <= 0xffff:
        return b'\xfd' + struct.pack('<H', n)
    if n <= 0xffffffff:
        return b'\xfe' + struct.pack('<I', n)
    return b'\xff' + struct.pack('<Q', n)


def ser_bytes(b):
    return ser_compact(len(b)) + b


def ser_input(txid_le, vout, script, sequence):
    return txid_le + struct.pack('<I', vout) + ser_bytes(script) + struct.pack('<I', sequence)


def ser_output(value, script):
    # Bitcoin uses signed int64 for CAmount but SIGHASH_SINGLE produces value=-1
    # which is encoded as 0xff*8 in little-endian either way. Use Q (unsigned)
    # so we can accept both positive values and -1 represented as 0xffffffffffffffff.
    if value < 0:
        value = value & 0xffffffffffffffff
    return struct.pack('<Q', value) + ser_bytes(script)


def legacy_sighash(tx_version, tx_inputs, tx_outputs, tx_locktime,
                   input_idx, scriptCode, sighash_type):
    """
    tx_inputs: list of (txid_le: bytes32, vout: int, script: bytes, sequence: int)
    tx_outputs: list of (value: int, script: bytes)
    input_idx: index of input being signed
    scriptCode: the script used for sighash (post-FindAndDelete for our QSB case)
    sighash_type: full 32-bit type (hashType)
    """
    base_type = sighash_type & 0x1f  # 5 low bits
    anyone_can_pay = (sighash_type & SIGHASH_ANYONECANPAY) != 0

    # ---- SIGHASH_SINGLE bug ----
    # If SIGHASH_SINGLE and input_idx >= len(outputs), z = 1.
    if base_type == SIGHASH_SINGLE and input_idx >= len(tx_outputs):
        return (1).to_bytes(32, 'big')

    # Build the transaction-to-be-hashed
    # Strip scriptSigs from all inputs except input_idx (which gets scriptCode)
    out_inputs = []
    if anyone_can_pay:
        # Only include input_idx
        txid, vout, _, seq = tx_inputs[input_idx]
        out_inputs = [(txid, vout, scriptCode, seq)]
    else:
        for i, (txid, vout, _, seq) in enumerate(tx_inputs):
            if i == input_idx:
                out_inputs.append((txid, vout, scriptCode, seq))
            else:
                # For SIGHASH_NONE and SIGHASH_SINGLE, set seq=0 for other inputs
                new_seq = seq
                if base_type in (SIGHASH_NONE, SIGHASH_SINGLE):
                    new_seq = 0
                out_inputs.append((txid, vout, b'', new_seq))

    # Outputs
    if base_type == SIGHASH_NONE:
        out_outputs = []
    elif base_type == SIGHASH_SINGLE:
        # Only include output at input_idx; blank out earlier outputs
        if input_idx >= len(tx_outputs):
            # Should have been caught above, but be safe
            return (1).to_bytes(32, 'big')
        out_outputs = []
        for i in range(input_idx + 1):
            if i == input_idx:
                out_outputs.append(tx_outputs[i])
            else:
                out_outputs.append((0xffffffffffffffff, b''))  # "-1" value, empty script
    else:  # SIGHASH_ALL
        out_outputs = list(tx_outputs)

    # Serialize
    ser = b''
    ser += struct.pack('<i', tx_version)
    ser += ser_compact(len(out_inputs))
    for inp in out_inputs:
        ser += ser_input(*inp)
    ser += ser_compact(len(out_outputs))
    for val, scr in out_outputs:
        ser += ser_output(val, scr)
    ser += struct.pack('<I', tx_locktime)
    ser += struct.pack('<I', sighash_type)  # 32-bit

    # Double-SHA256
    return dsha256(ser)


if __name__ == '__main__':
    # Smoke test
    txid = bytes.fromhex("a0293e4eeeb6259f5c8d96ee0a9cbbdb1d56bd6d0e5d456f0c61f6e0b6e4e30d")[::-1]
    inputs = [(txid, 0, b'', 0xffffffff)]
    outputs = [(1000000000, bytes.fromhex("76a914b8268ce4d481413c4e848ff353cd16104291c45b88ac"))]
    scriptCode = outputs[0][1]

    z = legacy_sighash(1, inputs, outputs, 0, 0, scriptCode, 0x01)
    print(f"Test 1 SIGHASH_ALL: {z.hex()}")

    # SIGHASH_SINGLE with no outputs
    z_sb = legacy_sighash(2, inputs, [], 0, 0, scriptCode, 0x03)
    print(f"Test 2 SIGHASH_SINGLE no-outs: {z_sb.hex()} (expect 01 * 31 + 01)")
    print(f"  match z=1: {z_sb == bytes.fromhex('00' * 31 + '01')}")
