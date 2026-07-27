#!/usr/bin/env python3
"""
test_funding_tx.py — Regtest Phase 1 test.

Builds a funding tx (Tx1) that creates a QSB scriptPubKey output, submits it
to a regtest bitcoind, and confirms it is accepted into a block.

This validates:
  - The QSB scriptPubKey is well-formed (parses as a valid script of any size)
  - The transaction containing it is structurally valid
  - The transaction relays + mines on Bitcoin Core's consensus engine

What it does NOT validate:
  - The QSB script actually executes correctly (that requires Tx2; see test_spending_tx.py)

Prerequisites:
  - bitcoind installed and on PATH
  - regtest node started:  ./setup_regtest.sh start
  - QSB state generated:   python3 ../pipeline/qsb_pipeline.py setup --config A

Usage:
  python3 test_funding_tx.py
  python3 test_funding_tx.py --qsb-value 10000   # default QSB output value (sats)

Exit code 0 = funding tx accepted.
Non-zero = bitcoind rejected. Read the diagnostic to find out why.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'pipeline'))

from bitcoin_tx import Transaction, TxIn, TxOut  # type: ignore
from secp256k1 import sha256d  # type: ignore


REGTEST_SCRIPT = Path(__file__).resolve().parent / 'setup_regtest.sh'
WALLET = "qsb_test"


def cli(*args, check=True, parse_json=False):
    """Run bitcoin-cli via setup_regtest.sh wrapper."""
    full_args = ['bash', str(REGTEST_SCRIPT), 'cli',
                 f'-rpcwallet={WALLET}', *map(str, args)]
    r = subprocess.run(full_args, capture_output=True, text=True, timeout=60)
    if check and r.returncode != 0:
        print(f"ERROR running bitcoin-cli {' '.join(map(str, args))}:")
        print(r.stderr)
        sys.exit(1)
    if parse_json:
        return json.loads(r.stdout)
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', default='qsb_state.json',
                    help="path to qsb_state.json (default: cwd)")
    ap.add_argument('--qsb-value', type=int, default=10000,
                    help="value in sats to lock into the QSB output (default: 10000)")
    ap.add_argument('--fee', type=int, default=2000,
                    help="fee in sats (default: 2000)")
    args = ap.parse_args()

    state_path = Path(args.state)
    if not state_path.exists():
        print(f"ERROR: state file not found: {state_path}")
        print("Run `qsb_pipeline.py setup --config A` first.")
        sys.exit(1)

    print("═" * 67)
    print("  QSB regtest Phase 1: funding-tx structure test")
    print("═" * 67)

    # Load state to get the QSB scriptPubKey
    with open(state_path) as f:
        state = json.load(f)
    qsb_script = bytes.fromhex(state['full_script_hex'])
    print(f"  QSB scriptPubKey: {len(qsb_script)} bytes (config={state.get('config', '?')})")
    print(f"  QSB output value: {args.qsb_value} sats")

    # Sanity-check bitcoind is up
    try:
        info = cli('getblockchaininfo', parse_json=True)
        print(f"  Regtest blocks: {info['blocks']}")
    except Exception as e:
        print(f"ERROR: bitcoin-cli failed — is the regtest node running?")
        print(f"  Try: ./setup_regtest.sh start")
        sys.exit(1)

    # Pick a UTXO to fund from
    utxos = cli('listunspent', '1', '999999', parse_json=True)
    if not utxos:
        print("ERROR: no UTXOs in wallet — has the regtest been mined?")
        sys.exit(1)
    # Pick one with enough value
    utxo = None
    needed_sats = args.qsb_value + args.fee
    for u in utxos:
        if int(u['amount'] * 1e8) >= needed_sats + 1000:  # leave room for change
            utxo = u
            break
    if utxo is None:
        print(f"ERROR: no UTXO with enough value ({needed_sats} sats needed)")
        sys.exit(1)
    print(f"  Funding from {utxo['txid']}:{utxo['vout']} "
          f"({int(utxo['amount']*1e8)} sats)")

    # Get a change address
    change_addr = cli('getnewaddress')
    print(f"  Change address: {change_addr}")

    # Build the unsigned tx using Bitcoin Core's createrawtransaction
    inputs_json = json.dumps([{'txid': utxo['txid'], 'vout': utxo['vout']}])
    # Outputs: QSB output (raw script via "data" trick won't work; we need to use
    # a manual approach since createrawtransaction only supports addresses or 'data')
    # 
    # We construct the tx manually with our Transaction class, then sign it via
    # signrawtransactionwithwallet.
    input_amount_sats = int(utxo['amount'] * 1e8)
    change_amount = input_amount_sats - args.qsb_value - args.fee
    if change_amount < 546:
        print(f"WARN: change amount {change_amount} below dust; absorbing into fee")
        change_amount = 0

    # Get the change scriptPubKey
    change_info = cli('getaddressinfo', change_addr, parse_json=True)
    change_script = bytes.fromhex(change_info['scriptPubKey'])

    # Build tx
    tx = Transaction(version=2, locktime=0)
    txid_le = bytes.fromhex(utxo['txid'])[::-1]
    tx.add_input(TxIn(txid_le, utxo['vout'], b'', 0xfffffffd))  # RBF-disabled
    tx.add_output(TxOut(args.qsb_value, qsb_script))
    if change_amount > 0:
        tx.add_output(TxOut(change_amount, change_script))

    raw_unsigned = tx.serialize().hex()
    print(f"  Unsigned tx: {len(raw_unsigned)//2} bytes")

    # Sign via wallet
    signed = cli('signrawtransactionwithwallet', raw_unsigned, parse_json=True)
    if not signed.get('complete'):
        print(f"ERROR: signing incomplete:")
        print(json.dumps(signed.get('errors', []), indent=2))
        sys.exit(1)
    raw_signed = signed['hex']
    print(f"  Signed tx: {len(raw_signed)//2} bytes")

    # Test acceptance (dry run) before broadcasting
    print("\n  testmempoolaccept (dry-run check)...")
    accept_check = cli('testmempoolaccept', json.dumps([raw_signed]),
                       parse_json=True)
    if not accept_check[0].get('allowed'):
        reason = accept_check[0].get('reject-reason', '?')
        print(f"  ✗ rejected: {reason}")
        # Note: regtest with acceptnonstdtxn=1 should NOT reject for non-standardness.
        # If we get here, it's likely a real consensus violation.
        sys.exit(1)
    print("  ✓ accepted by testmempoolaccept")

    # Broadcast
    print("\n  sendrawtransaction (broadcast + accept into mempool)...")
    txid = cli('sendrawtransaction', raw_signed)
    print(f"  ✓ accepted, txid: {txid}")

    # Mine a block to confirm it
    print("\n  mining 1 block...")
    miner_addr = cli('getnewaddress')
    block_hashes = cli('generatetoaddress', '1', miner_addr, parse_json=True)
    print(f"  ✓ mined block {block_hashes[0]}")

    # Verify the tx is actually in the block
    block = cli('getblock', block_hashes[0], parse_json=True)
    if txid in block['tx']:
        print(f"  ✓ tx found in block at index {block['tx'].index(txid)}")
    else:
        print(f"  ✗ tx NOT in block (this should not happen)")
        sys.exit(1)

    # Save the funding tx info for the spending test
    funding_info = {
        'funding_txid': txid,
        'funding_vout': 0,  # QSB output is always vout 0 in our construction
        'funding_value': args.qsb_value,
        'funding_script_hex': qsb_script.hex(),
        'block_hash': block_hashes[0],
        'block_height': info['blocks'] + 1,
    }
    out_file = Path('regtest_funding.json')
    with open(out_file, 'w') as f:
        json.dump(funding_info, f, indent=2)
    print(f"\n  Saved funding info to {out_file}")

    # Bonus: decode the QSB script via Bitcoin Core's decodescript RPC.
    # This at least confirms it parses cleanly at the opcode-stream level.
    print(f"\n  decodescript (parser-level validation)...")
    decoded = cli('decodescript', qsb_script.hex(), parse_json=True)
    asm_preview = decoded.get('asm', '')[:120]
    print(f"  ✓ decoded as type='{decoded.get('type', '?')}'")
    print(f"    first opcodes: {asm_preview}{'...' if len(decoded.get('asm', '')) > 120 else ''}")

    print()
    print("═" * 67)
    print("  ✅ PHASE 1 PASSED")
    print("  - Bitcoin Core accepted the QSB scriptPubKey output")
    print("  - tx confirmed in regtest block")
    print()
    print("  Next step: run the GPU search using this funding txid, then")
    print("  test the spending tx with test_spending_tx.py")
    print("═" * 67)


if __name__ == '__main__':
    main()
