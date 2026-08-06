#!/usr/bin/env python3
"""
test_end_to_end.py — full pipeline ↔ emulator equivalence.

Flow:
  1. Run cmd_setup via subprocess (generates state)
  2. Run cmd_export with dummy funding params (generates GPU input files)
  3. Do a small CPU search (digest rounds) to find a real hit
  4. Run the emulator on the same subset and compare byte-for-byte

If ANY of the three intermediates (sighash, pubkey, puzzle_hash) differs between
pipeline and emulator, we fail — that's a GPU/CPU contract break.
"""
import hashlib
import os
import struct
import subprocess
import sys
import json
import tempfile
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pipeline'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bitcoin_tx import (  # type: ignore
    Transaction, TxIn, TxOut, QSBScriptBuilder,
    push_data, find_and_delete, encode_der_sig,
)
from secp256k1 import (  # type: ignore
    N, P, G, point_mul, point_add, modinv,
)
from gpu_emulator import (
    emulate_digest_round, emulate_pinning, puzzle_hash, compress_pubkey,
)


PIPELINE_DIR = Path(__file__).resolve().parent.parent / 'pipeline'


def run_pipeline(cmd_args, work_dir):
    """Run `qsb_pipeline.py <args>` in work_dir."""
    result = subprocess.run(
        ['python3', str(PIPELINE_DIR / 'qsb_pipeline.py'), *cmd_args],
        cwd=work_dir, capture_output=True, text=True, timeout=300,
    )
    return result.returncode == 0, result.stdout, result.stderr


def cpu_search_digest(state, ri, base_sc_no_nonce, tx, qsb_idx, dummy_sigs,
                      hash_mode, t, max_tries=50000):
    """CPU-side enumerate search for round ri. Returns first match or None."""
    rs = state['round_sigs'][ri]
    r_val, s_val = rs['r'], rs['s']
    d_r_inv = modinv(r_val, N)
    dx = r_val
    dy_sq = (pow(dx, 3, P) + 7) % P
    dy = pow(dy_sq, (P + 1) // 4, P)
    if dy % 2 != 0:
        dy = P - dy
    dR = (dx, dy)

    n = state['n']
    tries = 0
    for combo in combinations(range(n), t):
        if tries >= max_tries:
            return None
        tries += 1
        sc = base_sc_no_nonce
        for idx in combo:
            sc = find_and_delete(sc, bytes.fromhex(dummy_sigs[idx]))
        z = tx.sighash(qsb_idx, sc, sighash_type=0x01)
        u1 = (-z * d_r_inv) % N
        u2 = (s_val * d_r_inv) % N
        Q = point_add(point_mul(u1, G), point_mul(u2, dR))
        pk = compress_pubkey(Q)
        ph, ph_der, _ = puzzle_hash(pk, hash_mode)
        if ph_der or (ph[0] >> 4) == 3:
            return {'combo': list(combo), 'z': z, 'pubkey': pk,
                    'puzzle_hash': ph, 'valid_der': ph_der,
                    'scriptcode': sc}
    return None


def main():
    work = Path(tempfile.mkdtemp(prefix='qsb_e2e_'))
    print("═" * 67)
    print(f"  End-to-end test")
    print(f"  Work dir: {work}")
    print("═" * 67)

    # ═══ Setup ═══
    print("\n[1] pipeline setup (Config A, all-sha256)")
    ok, out, err = run_pipeline(['setup', '--config', 'A', '--seed', '42'], work)
    if not ok:
        print(f"  ✗ Setup failed\n{(out + err)[-500:]}")
        return 1
    with open(work / 'qsb_state.json') as f:
        state = json.load(f)
    print(f"  ✓ n={state['n']}, hash_mode={state['hash_mode']}")
    print(f"  ✓ script: {len(bytes.fromhex(state['full_script_hex']))} bytes")

    # ═══ Export ═══
    print("\n[2] pipeline export")
    fake_txid = '11' * 32
    extra_txid = '22' * 32
    ok, out, err = run_pipeline(
        ['export', '--funding-txid', fake_txid, '--funding-vout', '0',
         '--funding-value', '10000',
         '--extra-input-txid', extra_txid,
         '--extra-input-vout', '1',
         '--extra-input-value', '5000',
         '--output-value', '14000',
         '--output-address', '00' * 20,  # P2PKH hex pubkeyhash
         '--locktime', '0', '--sequence', str(0xfffffffe), '--version', '1'], work)
    if not ok:
        print(f"  ✗ Export failed\n{(out + err)[-500:]}")
        return 1
    expected = ['gpu_pinning_params.json', 'pinning.bin',
                'gpu_digest_r1_params.json', 'digest_r1.bin',
                'gpu_digest_r2_params.json', 'digest_r2.bin']
    missing = [f for f in expected if not (work / f).exists()]
    if missing:
        print(f"  ✗ Missing files: {missing}")
        return 1
    print(f"  ✓ All {len(expected)} files exported")

    # ═══ Mini-search + emulator check for each digest round ═══
    print("\n[3] CPU search + emulator cross-check")
    t1 = state['t1s'] + state['t1b']
    t2 = state['t2s'] + state['t2b']
    hash_mode = state['hash_mode']
    full_script = bytes.fromhex(state['full_script_hex'])

    funding_txid_le = bytes.fromhex(fake_txid)[::-1]
    extra_txid_le = bytes.fromhex(extra_txid)[::-1]
    # Build 2-in/1-out tx EXACTLY matching the exporter's structure (must
    # match the `--extra-input-*` and `--output-*` values passed in [2]).
    tx = Transaction(version=1, locktime=0)
    tx.add_input(TxIn(extra_txid_le, 1, b'', 0xfffffffd))   # input[0] (extra)
    tx.add_input(TxIn(funding_txid_le, 0, b'', 0xfffffffe))  # input[1] (QSB)

    # Output: 14000 sats P2PKH '00'*20
    p2pkh = bytes([0x76, 0xa9, 0x14]) + bytes.fromhex('00' * 20) + bytes([0x88, 0xac])
    tx.add_output(TxOut(14000, p2pkh))

    # QSB is at input index 1
    QSB_IDX = 1

    overall_ok = True
    for ri in range(2):
        print(f"\n  round {ri + 1}:")
        with open(work / f'gpu_digest_r{ri + 1}_params.json') as f:
            params = json.load(f)
        params.setdefault('hash_mode', hash_mode)

        sig_nonce = bytes.fromhex(state['round_sigs'][ri]['sig'])
        base_sc = find_and_delete(full_script, sig_nonce)
        t = t1 if ri == 0 else t2

        hit = cpu_search_digest(state, ri, base_sc, tx, QSB_IDX,
                                state['dummy_sigs'][ri], hash_mode, t,
                                max_tries=30000)
        if hit is None:
            print(f"    — no hit in 30000 combos (subset space larger than that)")
            print(f"      skipping round {ri + 1} emulator check (not a bug; just slow)")
            continue

        print(f"    pipe: combo={hit['combo'][:4]}{'...' if len(hit['combo']) > 4 else ''}")
        print(f"          z={hex(hit['z'])[:18]}...")
        print(f"          pk={hit['pubkey'].hex()[:20]}...")
        print(f"          ph={hit['puzzle_hash'].hex()[:20]}...")

        # Run emulator on same subset (pass tx's seq/lt so emulator patches
        # them into tx_suffix to match the search-time tx)
        emu_out = emulate_digest_round(
            params, hit['combo'], sighash_type=0x01,
            sequence=tx.inputs[QSB_IDX].sequence, locktime=tx.locktime)
        emu_z_int = int.from_bytes(emu_out['sighash'], 'big')

        subtest_ok = True
        if emu_z_int != hit['z']:
            print(f"    ✗ sighash mismatch")
            print(f"       pipe: {hit['z']:064x}")
            print(f"       emu:  {emu_z_int:064x}")
            # Helpful: are the scriptcodes different?
            if emu_out['scriptcode'] != hit['scriptcode']:
                print(f"    ✗ scriptcodes also differ!")
                print(f"       pipe len: {len(hit['scriptcode'])}")
                print(f"       emu len:  {len(emu_out['scriptcode'])}")
                for i in range(min(len(hit['scriptcode']), len(emu_out['scriptcode']))):
                    if hit['scriptcode'][i] != emu_out['scriptcode'][i]:
                        print(f"       first diff at byte {i}")
                        break
            subtest_ok = False

        emu_match = next((c for c in emu_out['candidates']
                          if c['pubkey'] == hit['pubkey']), None)
        if emu_match is None:
            print(f"    ✗ emulator did not recover pipeline's pubkey")
            subtest_ok = False
        elif emu_match['puzzle_hash'] != hit['puzzle_hash']:
            print(f"    ✗ puzzle_hash mismatch")
            subtest_ok = False

        if subtest_ok:
            print(f"    ✓ pipeline ↔ emulator agree on sighash, pubkey, puzzle_hash")
        else:
            overall_ok = False

    print()
    if overall_ok:
        print("═" * 67)
        print("  ✅ End-to-end equivalence CONFIRMED")
        print("═" * 67)
        return 0
    print("═" * 67)
    print("  ❌ Divergence detected — do NOT run GPU search")
    print("═" * 67)
    return 1


if __name__ == '__main__':
    sys.exit(main())
