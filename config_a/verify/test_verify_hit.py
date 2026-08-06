#!/usr/bin/env python3
"""test_verify_hit.py — dedicated test for verify_hit.py (last-line defense).

Focuses on the REJECT paths, which are security-critical — a false "accept"
is far more dangerous than a missed "accept":

  [reject 1] wrong indices → rc=1, sighash matches but no valid DER
  [reject 2] tampered params → rc=1, sighash MISMATCH detected
  [reject 3] old-format params → rc=2, format guard trips

The ACCEPT path (rc=0 on a valid hit) is tested implicitly elsewhere:
  - test_gpu_cpu_equivalence.py exercises emulate_digest_round's full logic
    including its is_valid_der → hit mapping.
  - test_end_to_end.py confirms pipeline ↔ emulator byte-agreement on real
    Config A params (sighash, pubkey, puzzle_hash).

A direct accept test here would require either (a) ~2^22 CPU tries to find
a real strict-DER hit in Config A's parameter space, or (b) monkey-patching
the emulator's validity check, which fights Python's import ordering when
verify_hit.py is invoked as a subprocess.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run_verify_hit(work_dir, *args):
    r = subprocess.run(
        ['python3', str(REPO / 'verify' / 'verify_hit.py'),
         '--work-dir', str(work_dir), *args],
        capture_output=True, text=True, timeout=60,
    )
    return r.returncode, r.stdout + r.stderr


def setup_pipeline(work_dir, seed='42'):
    r = subprocess.run(
        ['python3', str(REPO / 'pipeline' / 'qsb_pipeline.py'),
         'setup', '--config', 'A', '--seed', seed],
        cwd=work_dir, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, f"setup failed: {r.stderr}"
    fake_txid = '11' * 32
    extra_txid = '22' * 32
    r = subprocess.run(
        ['python3', str(REPO / 'pipeline' / 'qsb_pipeline.py'),
         'export',
         '--funding-txid', fake_txid, '--funding-vout', '0',
         '--funding-value', '10000',
         '--extra-input-txid', extra_txid,
         '--extra-input-vout', '1',
         '--extra-input-value', '5000',
         '--output-value', '14000',
         '--output-address', '00' * 20,
         '--locktime', '0', '--sequence', '4294967294', '--version', '1'],
        cwd=work_dir, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, f"export failed: {r.stderr}"
    return fake_txid


def main():
    print("═" * 70)
    print("  verify_hit.py driver tests (reject paths)")
    print("═" * 70)

    work = Path(tempfile.mkdtemp(prefix='qsb_vh_'))
    print(f"  work dir: {work}")

    fake_txid = setup_pipeline(work)
    with open(work / 'qsb_state.json') as f:
        state = json.load(f)
    t = state['t1s'] + state['t1b']
    wrong_indices = ','.join(str(i) for i in range(t))

    results = {}

    # [reject 1] wrong indices
    print("\n[reject 1] wrong indices → rc=1, no valid DER")
    rc, out = run_verify_hit(
        work, 'digest', '--round', '1', '--indices', wrong_indices,
        '--locktime', '0', '--sequence', '0xfffffffe',
        '--funding-txid', fake_txid,
    )
    results['reject_wrong_indices'] = (
        rc == 1 and 'sighashes match' in out
        and 'No candidate produced a valid DER' in out)
    print(f"  {'✓' if results['reject_wrong_indices'] else '✗'} rc={rc}")

    # [reject 2] tampered params (sighash mismatch)
    print("\n[reject 2] tampered gpu_digest_r1_params.json → rc=1, sighash mismatch")
    params_file = work / 'gpu_digest_r1_params.json'
    with open(params_file) as f:
        p = json.load(f)
    orig_tx_prefix = p['tx_prefix']
    bad = bytearray(bytes.fromhex(orig_tx_prefix))
    bad[5] ^= 0xFF
    p['tx_prefix'] = bad.hex()
    with open(params_file, 'w') as f:
        json.dump(p, f)
    rc, out = run_verify_hit(
        work, 'digest', '--round', '1', '--indices', wrong_indices,
        '--locktime', '0', '--sequence', '0xfffffffe',
        '--funding-txid', fake_txid,
    )
    results['reject_tampered'] = (rc == 1 and 'SIGHASH MISMATCH' in out)
    print(f"  {'✓' if results['reject_tampered'] else '✗'} rc={rc}")
    p['tx_prefix'] = orig_tx_prefix
    with open(params_file, 'w') as f:
        json.dump(p, f)

    # [reject 3] old-format params
    print("\n[reject 3] old-format params (missing pre_hors_section) → rc=2")
    with open(params_file) as f:
        p = json.load(f)
    p.pop('pre_hors_section', None)
    with open(params_file, 'w') as f:
        json.dump(p, f)
    rc, out = run_verify_hit(
        work, 'digest', '--round', '1', '--indices', wrong_indices,
        '--locktime', '0', '--sequence', '0xfffffffe',
        '--funding-txid', fake_txid,
    )
    results['reject_old_format'] = (rc == 2 and 'OLD format' in out)
    print(f"  {'✓' if results['reject_old_format'] else '✗'} rc={rc}")

    print()
    print("═" * 70)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'}  {name}")
    all_ok = all(results.values())
    print("═" * 70)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
