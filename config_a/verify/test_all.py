#!/usr/bin/env python3
"""
test_all.py — the single test that must pass before spending any GPU money.

Runs every check we have:
  [A] Pre-flight verifier      (Config A script budget, size, F&D reference match)
  [B] Equivalence test suite   (32 cross-checks: DER, sighash, F&D, pubkey, ECDSA, puzzle_hash)
  [C] End-to-end integration   (run real `setup` + `export` + CPU search + emulator)
  [D] Config sanity            (A, Ar, S all build + fit; D correctly flagged as over)

If every phase exits 0, the pipeline is safe to use. Otherwise STOP.

Usage:
    python3 test_all.py
"""
import subprocess
import sys
import time
from pathlib import Path


VERIFY = Path(__file__).resolve().parent


def run(label, cmd, timeout=600):
    print(f"\n{'=' * 70}")
    print(f" {label}")
    print('=' * 70)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=False, text=True, timeout=timeout)
    elapsed = time.time() - t0
    print(f"\n  {label}: exit={r.returncode}, {elapsed:.1f}s")
    return r.returncode == 0


def config_sanity():
    """Verify each config builds correctly and the over-budget config is flagged."""
    print("\n" + "=" * 70)
    print(" [D] Config sanity")
    print("=" * 70)
    results = {}
    for name, should_pass in [('A', True), ('Ar', True), ('S', True), ('D', False)]:
        r = subprocess.run(
            ['python3', str(VERIFY / 'verify_script_budget.py'), '--config', name],
            capture_output=True, text=True, timeout=60,
        )
        ok = (r.returncode == 0) == should_pass
        marker = "✓" if ok else "✗"
        expected = "should fit" if should_pass else "should be over-budget"
        got = "fits" if r.returncode == 0 else "over-budget"
        print(f"  {marker} Config {name} ({expected}): {got}")
        results[name] = ok
    all_ok = all(results.values())
    print(f"\n  Result: {'all configs as expected' if all_ok else 'UNEXPECTED CONFIG BEHAVIOR'}")
    return all_ok


def main():
    print("=" * 70)
    print("  QSB — Master Test Suite")
    print("=" * 70)
    print("  This must ALL pass before spending real GPU compute.")

    results = []

    # [A] Pre-flight
    ok = run(
        "[A] Pre-flight verifier (Config A)",
        ['python3', str(VERIFY / 'verify_script_budget.py'), '--config', 'A'],
    )
    results.append(('preflight', ok))

    # [B] Equivalence suite
    ok = run(
        "[B] Equivalence test suite",
        ['python3', str(VERIFY / 'test_gpu_cpu_equivalence.py')],
    )
    results.append(('equivalence', ok))

    # [C] End-to-end
    ok = run(
        "[C] End-to-end pipeline ↔ emulator",
        ['python3', str(VERIFY / 'test_end_to_end.py')],
    )
    results.append(('e2e', ok))

    # [C2] verify_hit reject paths
    ok = run(
        "[C2] verify_hit.py reject paths",
        ['python3', str(VERIFY / 'test_verify_hit.py')],
    )
    results.append(('verify_hit_rejects', ok))

    # [C3] GPU kernel static analysis (g++ with CUDA stubs)
    ok = run(
        "[C3] GPU kernel static analysis",
        ['python3', str(VERIFY / 'test_kernel_static.py')],
    )
    results.append(('kernel_static', ok))

    # [C4] Tile partitioning self-test (small cases verified, large case checked for coverage)
    ok = run(
        "[C4] Tile partition (LPT) self-test",
        ['python3', str(VERIFY.parent / 'pipeline' / 'tile_partition.py')],
    )
    results.append(('tile_partition', ok))

    # [D] Config sanity
    ok = config_sanity()
    results.append(('config_sanity', ok))

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, ok in results:
        mark = "✅" if ok else "❌"
        print(f"  {mark}  {name}")

    all_ok = all(ok for _, ok in results)
    print()
    if all_ok:
        print("🎯 ALL TESTS PASSED — safe to proceed with GPU search.")
        return 0
    print("🛑 TEST FAILURE — DO NOT run GPU search. Fix issues first.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
