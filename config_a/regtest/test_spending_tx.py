#!/usr/bin/env python3
"""
test_spending_tx.py — Regtest Phase 3 test.

Takes the output of a real GPU search (a hits.json file) plus the regtest
funding tx info (regtest_funding.json from Phase 1), runs the pipeline's
assemble step to produce a spending tx (Tx2), submits it to regtest, and
confirms it is accepted into a block.

This is the FINAL VALIDATION: if Tx2 is accepted by Bitcoin Core's consensus
in regtest, the same assembled tx will be consensus-valid on mainnet.

Prerequisites:
  - Regtest node running (./setup_regtest.sh start)
  - Phase 1 passed (regtest_funding.json exists in cwd)
  - GPU search complete with all three hits available:
      * pinning hit: locktime + sequence
      * round 1 hit: subset indices
      * round 2 hit: subset indices

Hits should be provided either via command-line or via a hits.json file with
the structure:
  {
    "pin_locktime": 12345,
    "pin_sequence": "0xfffffffe",
    "round1_indices": [0, 5, 12, 23, 41, 67, 89, 121, 143],
    "round2_indices": [1, 8, 17, 39, 52, 78, 91, 102, 119]
  }

Usage:
  python3 test_spending_tx.py --hits gpu_hits.json
  python3 test_spending_tx.py --pin-locktime 12345 --r1-indices 0,5,12,... --r2-indices ...

Exit code:
  0 = SPENDING TX ACCEPTED — the pipeline is consensus-correct.
  1 = REJECTED — read the diagnostic to learn what's wrong.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PIPELINE = REPO / 'pipeline'
sys.path.insert(0, str(PIPELINE))

REGTEST_SCRIPT = Path(__file__).resolve().parent / 'setup_regtest.sh'
WALLET = "qsb_test"


def cli(*args, check=True, parse_json=False):
    full_args = ['bash', str(REGTEST_SCRIPT), 'cli',
                 f'-rpcwallet={WALLET}', *map(str, args)]
    r = subprocess.run(full_args, capture_output=True, text=True, timeout=60)
    if check and r.returncode != 0:
        print(f"ERROR running bitcoin-cli {' '.join(map(str, args))}:")
        print(r.stderr)
        sys.exit(1)
    if parse_json:
        try:
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            return r.stdout.strip()
    return r.stdout.strip()


def parse_hits(args):
    """Read hit info either from --hits JSON file or from CLI flags."""
    if args.hits:
        with open(args.hits) as f:
            data = json.load(f)
        return data
    # Build from CLI flags
    if not (args.pin_locktime is not None and args.r1_indices and args.r2_indices):
        print("ERROR: provide --hits or all of --pin-locktime --r1-indices --r2-indices")
        sys.exit(1)
    return {
        'pin_locktime': args.pin_locktime,
        'pin_sequence': args.pin_sequence,
        'round1_indices': [int(x) for x in args.r1_indices.split(',')],
        'round2_indices': [int(x) for x in args.r2_indices.split(',')],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hits', help="JSON file with hit info")
    ap.add_argument('--pin-locktime', type=int)
    ap.add_argument('--pin-sequence', default='0xfffffffe',
                    help="hex sequence used by pinning (default: 0xfffffffe)")
    ap.add_argument('--r1-indices', help="comma-separated round 1 subset")
    ap.add_argument('--r2-indices', help="comma-separated round 2 subset")
    ap.add_argument('--funding', default='regtest_funding.json',
                    help="funding info from Phase 1 (default: cwd)")
    ap.add_argument('--state', default='qsb_state.json')
    args = ap.parse_args()

    hits = parse_hits(args)
    print("═" * 67)
    print("  QSB regtest Phase 3: spending-tx consensus test")
    print("═" * 67)

    funding_path = Path(args.funding)
    if not funding_path.exists():
        print(f"ERROR: {funding_path} not found — run test_funding_tx.py first")
        sys.exit(1)
    with open(funding_path) as f:
        funding = json.load(f)
    print(f"  Funding tx:    {funding['funding_txid']}")
    print(f"  Funding vout:  {funding['funding_vout']}")
    print(f"  Funding value: {funding['funding_value']} sats")

    print(f"\n  Hits:")
    print(f"    pin lt:     {hits['pin_locktime']}")
    print(f"    pin seq:    {hits.get('pin_sequence', '0xfffffffe')}")
    print(f"    r1 indices: {hits['round1_indices']}")
    print(f"    r2 indices: {hits['round2_indices']}")

    # Sanity: regtest is up and the funding tx is there
    try:
        cli('getblockchaininfo', parse_json=True)
    except SystemExit:
        print("ERROR: regtest node not running. ./setup_regtest.sh start")
        sys.exit(1)

    funding_info = cli('gettxout', funding['funding_txid'],
                       str(funding['funding_vout']), parse_json=True)
    if not funding_info:
        print(f"ERROR: funding output not found in regtest UTXO set")
        print("  Did the funding tx get reorged? Try Phase 1 again.")
        sys.exit(1)
    print(f"  Confirmed: funding output exists in UTXO set ({funding_info.get('confirmations')} conf)")

    # Run cmd_assemble to produce the spending tx
    pin_seq = hits.get('pin_sequence', '0xfffffffe')
    if isinstance(pin_seq, str):
        pin_seq = int(pin_seq, 0)
    print(f"\n  Running pipeline assemble...")
    r = subprocess.run(
        ['python3', str(PIPELINE / 'qsb_pipeline.py'), 'assemble',
         '--locktime', str(hits['pin_locktime']),
         '--sequence', str(pin_seq),
         '--round1', ','.join(str(i) for i in hits['round1_indices']),
         '--round2', ','.join(str(i) for i in hits['round2_indices']),
         '--funding-txid', funding['funding_txid'],
         '--funding-vout', str(funding['funding_vout']),
         '--funding-value', str(funding['funding_value'])],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        print(f"  ✗ pipeline assemble failed:")
        print(r.stdout[-1500:])
        print(r.stderr[-500:])
        sys.exit(1)
    print(r.stdout[-800:])

    raw_tx_path = Path('qsb_raw_tx.hex')
    if not raw_tx_path.exists():
        print(f"  ✗ assemble did not produce qsb_raw_tx.hex")
        sys.exit(1)
    raw_tx = raw_tx_path.read_text().strip()
    print(f"\n  Spending tx: {len(raw_tx)//2} bytes")

    # Test acceptance via testmempoolaccept (no broadcast yet)
    print(f"\n  testmempoolaccept (dry-run consensus check)...")
    accept_check = cli('testmempoolaccept', json.dumps([raw_tx]),
                       parse_json=True)
    result = accept_check[0]
    if not result.get('allowed'):
        reason = result.get('reject-reason', '?')
        print(f"  ✗ REJECTED: {reason}")
        print()
        print("  Common rejection reasons and what they mean:")
        print("  - 'mandatory-script-verify-flag-failed': script execution failed")
        print("    (look at the parenthetical for specifics: e.g., 'Operation not")
        print("     valid with the current stack size' = byte-layout problem)")
        print("  - 'non-mandatory-script-verify-flag': standardness, not consensus")
        print("    (should not happen on regtest with acceptnonstdtxn=1)")
        print("  - 'bad-txns-vin-empty/oversize/...': basic structure problem")
        print()
        print("  Full result:")
        print(json.dumps(result, indent=2))
        sys.exit(1)
    print(f"  ✓ accepted by testmempoolaccept")
    print(f"    fee: {result.get('fees', {}).get('base', '?')} BTC")
    print(f"    vsize: {result.get('vsize', '?')}")

    # Broadcast it
    print(f"\n  sendrawtransaction (broadcast + accept into mempool)...")
    txid = cli('sendrawtransaction', raw_tx)
    print(f"  ✓ broadcast: {txid}")

    # Mine to confirm
    print(f"\n  mining 1 block...")
    miner_addr = cli('getnewaddress')
    block_hashes = cli('generatetoaddress', '1', miner_addr, parse_json=True)
    block = cli('getblock', block_hashes[0], parse_json=True)
    if txid in block['tx']:
        print(f"  ✓ tx confirmed in block {block_hashes[0]}")
    else:
        print(f"  ✗ tx NOT confirmed (rejected at block-validation time?)")
        print(f"  Block contents: {block['tx']}")
        sys.exit(1)

    print()
    print("═" * 67)
    print("  ✅ PHASE 3 PASSED — SPENDING TX IS CONSENSUS-VALID")
    print()
    print("  Bitcoin Core's full consensus engine has validated:")
    print("    - Script structure (op count at runtime, byte layout, F&D)")
    print("    - All ECDSA signatures (pinning, sig_puzzle, both digest rounds)")
    print("    - Sighash computation")
    print("    - Transaction structure (size, sigops, etc.)")
    print()
    print("  This same tx, with the same script and signatures, would be")
    print("  consensus-valid on mainnet. Mempool relay still depends on")
    print("  miner policy (use Slipstream or similar direct submission).")
    print("═" * 67)


if __name__ == '__main__':
    main()
