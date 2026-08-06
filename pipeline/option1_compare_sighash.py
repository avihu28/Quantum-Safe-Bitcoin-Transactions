#!/usr/bin/env python3
"""
OPTION 1: Compare bitcoin_tx.py's sighash against python-bitcoinlib.

python-bitcoinlib is a well-tested reference that matches Bitcoin Core's
sighash algorithm (and is tested against it).

This test:
1. Constructs a few test transactions with varying scriptPubKeys, sequences, locktimes
2. Computes the legacy SIGHASH_ALL sighash using bitcoin_tx.py
3. Computes the same sighash using python-bitcoinlib
4. Confirms they match byte-for-byte

If they match, bitcoin_tx.py's sighash is Bitcoin-correct.
If they differ, we have a bug in bitcoin_tx.py that the GPU also inherits.

Requirements on your Mac:
    pip3 install python-bitcoinlib

Run:
    python3 option1_compare_sighash.py
"""
import os
import sys
import hashlib
import struct

# Import bitcoin_tx.py from the current directory
sys.path.insert(0, '.')

try:
    from bitcoin.core import CMutableTransaction, CMutableTxIn, CMutableTxOut, COutPoint, lx
    from bitcoin.core.script import CScript, SignatureHash, SIGHASH_ALL
except ImportError:
    print("ERROR: python-bitcoinlib not installed.")
    print("Install with: pip3 install python-bitcoinlib")
    sys.exit(1)

try:
    from bitcoin_tx import (
        Transaction, TxIn, TxOut, find_and_delete,
        serialize_varint, sha256d, push_data
    )
except ImportError:
    print("ERROR: bitcoin_tx.py not found in current directory.")
    print("Copy bitcoin_tx.py to the same folder as this script.")
    sys.exit(1)


def h2b(h): return bytes.fromhex(h)


def compute_sighash_ours(tx_version, tx_locktime, input_index,
                          inputs, outputs, script_code, sighash_type=0x01):
    """Compute sighash using bitcoin_tx.py."""
    tx = Transaction(version=tx_version, locktime=tx_locktime)
    for (txid, vout, seq) in inputs:
        tx.add_input(TxIn(txid, vout, b'', seq))
    for (value, script_pubkey) in outputs:
        tx.add_output(TxOut(value, script_pubkey))
    return tx.sighash(input_index, script_code, sighash_type=sighash_type)


def compute_sighash_reference(tx_version, tx_locktime, input_index,
                              inputs, outputs, script_code, sighash_type=0x01):
    """Compute sighash using python-bitcoinlib (reference)."""
    tx = CMutableTransaction()
    tx.nVersion = tx_version
    tx.nLockTime = tx_locktime
    tx.vin = []
    for (txid, vout, seq) in inputs:
        outpoint = COutPoint(lx(txid[::-1].hex()), vout)
        tx.vin.append(CMutableTxIn(outpoint, CScript(), seq))
    tx.vout = []
    for (value, script_pubkey) in outputs:
        tx.vout.append(CMutableTxOut(value, CScript(script_pubkey)))
    sighash_bytes = SignatureHash(CScript(script_code), tx, input_index, sighash_type)
    return int.from_bytes(sighash_bytes, 'big')


def run_test(name, tx_version, tx_locktime, inputs, outputs, script_code, input_index=0, sighash_type=0x01):
    ours = compute_sighash_ours(tx_version, tx_locktime, input_index,
                                 inputs, outputs, script_code, sighash_type)
    ref = compute_sighash_reference(tx_version, tx_locktime, input_index,
                                     inputs, outputs, script_code, sighash_type)
    match = ours == ref
    ours_hex = ours.to_bytes(32, 'big').hex()
    ref_hex = ref.to_bytes(32, 'big').hex()
    status = "PASS" if match else "FAIL"
    print(f"[{status}] {name}")
    print(f"         ours: {ours_hex}")
    print(f"         ref:  {ref_hex}")
    if not match:
        print(f"         DIFFERENCE ABOVE")
    return match


def main():
    # Test 1: simplest P2PKH-style
    txid_1 = h2b("4fab76e9b0538a49a77443030f8e0243a5d2558155647a839acea0efaa4edc91")[::-1]
    dummy_script = bytes([0x76, 0xa9, 0x14]) + bytes(20) + bytes([0x88, 0xac])  # OP_DUP OP_HASH160 <20> OP_EQUALVERIFY OP_CHECKSIG
    all_ok = True
    all_ok &= run_test("Basic P2PKH, v1, no locktime, seq=0xffffffff, no outputs",
                        tx_version=1, tx_locktime=0,
                        inputs=[(txid_1, 0, 0xffffffff)],
                        outputs=[],
                        script_code=dummy_script)

    # Test 2: Your actual QSB scenario - version 2, locktime set, sequence with top bit
    all_ok &= run_test("QSB-like: v2, locktime=570428883, seq=0x80007498, no outputs",
                        tx_version=2, tx_locktime=570428883,
                        inputs=[(txid_1, 0, 0x80007498)],
                        outputs=[],
                        script_code=dummy_script)

    # Test 3: Longer script (closer to QSB full script length)
    big_script = bytes([0x4c, 0xff]) + bytes(255) + bytes([0xac])  # PUSHDATA1 + 255 bytes + OP_CHECKSIG
    all_ok &= run_test("Big script with OP_CHECKSIG, v2, locktime=545078097",
                        tx_version=2, tx_locktime=545078097,
                        inputs=[(txid_1, 0, 0x80007469)],
                        outputs=[],
                        script_code=big_script)

    # Test 4: with outputs
    output_script = bytes([0x76, 0xa9, 0x14]) + bytes(20) + bytes([0x88, 0xac])
    all_ok &= run_test("With output, v2, locktime, seq",
                        tx_version=2, tx_locktime=545078097,
                        inputs=[(txid_1, 0, 0x80007469)],
                        outputs=[(9900, output_script)],
                        script_code=dummy_script)

    # Test 5: script containing bytes resembling DER-sig-with-sighash_type 
    # (to exercise FindAndDelete indirectly — not actually in this test)
    # Just a long arbitrary script:
    arbitrary = bytes.fromhex("ac76a91488" * 100 + "ac")  # alternating opcodes
    all_ok &= run_test("Arbitrary long script, 0 outputs",
                        tx_version=2, tx_locktime=0,
                        inputs=[(txid_1, 0, 0xfffffffe)],
                        outputs=[],
                        script_code=arbitrary)

    # Test 7: Compute the ACTUAL QSB pinning sighash (the thing we care about)
    # and compare against python-bitcoinlib. This is the definitive test.
    import os.path
    if os.path.exists("qsb_state.json"):
        print()
        print("--- QSB real sighash test ---")
        import json
        with open("qsb_state.json") as f:
            qsb_state = json.load(f)
        full_script = h2b(qsb_state["full_script_hex"])
        pin_sig = h2b(qsb_state["pin_sig"])
        pin_sc = find_and_delete(full_script, pin_sig)
        print(f"  full_script len: {len(full_script)}")
        print(f"  pin_sc len after FaD: {len(pin_sc)}")

        txid_qsb = h2b("4fab76e9b0538a49a77443030f8e0243a5d2558155647a839acea0efaa4edc91")[::-1]
        all_ok &= run_test("REAL QSB pinning sighash (seq=2147501802, lt=702856043)",
                            tx_version=2, tx_locktime=702856043,
                            inputs=[(txid_qsb, 0, 2147501802)],
                            outputs=[],
                            script_code=pin_sc)

        # And for a round1 sighash
        sig_nonce_r1 = h2b(qsb_state["round_sigs"][0]["sig"])
        round1_base_sc = find_and_delete(full_script, sig_nonce_r1)
        # Simulate removing 9 dummy sigs (any 9 — we just test sighash math)
        for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            ds = h2b(qsb_state["dummy_sigs"][0][idx])
            round1_base_sc = find_and_delete(round1_base_sc, ds)
        print(f"  round1 sc len after FaD (9 dummies): {len(round1_base_sc)}")
        all_ok &= run_test("REAL QSB round 1 sighash (9 dummies removed)",
                            tx_version=2, tx_locktime=570428883,
                            inputs=[(txid_qsb, 0, 2147507660)],
                            outputs=[],
                            script_code=round1_base_sc)
    else:
        print()
        print("[SKIP] qsb_state.json not found in current dir — skipping real QSB sighash test.")

    print()
    if all_ok:
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("bitcoin_tx.py's sighash matches python-bitcoinlib (Bitcoin-correct).")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("SOME TESTS FAILED ✗")
        print("bitcoin_tx.py's sighash does NOT match the reference.")
        print("This is a real bug. DO NOT proceed with GPU runs until fixed.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
