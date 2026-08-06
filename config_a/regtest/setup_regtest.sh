#!/bin/bash
# setup_regtest.sh — start bitcoind in regtest mode for QSB testing.
#
# Creates an isolated regtest node in /tmp/qsb_regtest (or $QSB_REGTEST_DIR).
# Default settings: acceptnonstdtxn=1, txconfirmtarget=1, fallbackfee=0.0001
#
# Usage:
#   ./setup_regtest.sh start    # start the node
#   ./setup_regtest.sh stop     # stop and DESTROY the data dir
#   ./setup_regtest.sh cli ...  # forward args to bitcoin-cli
#
# After start, the node is ready and the wallet "qsb_test" is created and
# funded with 50 BTC of mature coins.
set -e

REGTEST_DIR="${QSB_REGTEST_DIR:-/tmp/qsb_regtest}"
RPC_PORT="${QSB_REGTEST_RPC_PORT:-18443}"
WALLET_NAME="qsb_test"

CLI="bitcoin-cli -regtest -datadir=${REGTEST_DIR} -rpcport=${RPC_PORT}"

cmd_start() {
    if [ -d "$REGTEST_DIR" ]; then
        echo "Existing regtest dir at $REGTEST_DIR — stopping any existing node"
        cmd_stop || true
        sleep 1
        rm -rf "$REGTEST_DIR"
    fi

    mkdir -p "$REGTEST_DIR"
    cat > "$REGTEST_DIR/bitcoin.conf" <<EOF
regtest=1
server=1
txindex=1
acceptnonstdtxn=1
fallbackfee=0.0001
maxtxfee=1
[regtest]
rpcport=${RPC_PORT}
rpcuser=qsb
rpcpassword=qsb_test_pw
EOF

    echo "Starting bitcoind in $REGTEST_DIR (rpc port ${RPC_PORT})..."
    bitcoind -datadir="${REGTEST_DIR}" -daemon

    # Wait for RPC to come up
    for i in $(seq 1 20); do
        if $CLI getblockchaininfo >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    if ! $CLI getblockchaininfo >/dev/null 2>&1; then
        echo "ERROR: bitcoind did not become ready"
        return 1
    fi

    echo "Creating wallet $WALLET_NAME..."
    $CLI createwallet "$WALLET_NAME" >/dev/null

    echo "Mining 101 blocks (gives 50 BTC of mature coins)..."
    ADDR=$($CLI -rpcwallet="$WALLET_NAME" getnewaddress)
    $CLI generatetoaddress 101 "$ADDR" >/dev/null

    BALANCE=$($CLI -rpcwallet="$WALLET_NAME" getbalance)
    echo "Wallet balance: $BALANCE BTC"
    echo ""
    echo "Ready. To use:"
    echo "  $CLI -rpcwallet=$WALLET_NAME <command>"
    echo "Or via this script:"
    echo "  ./setup_regtest.sh cli -rpcwallet=$WALLET_NAME <command>"
}

cmd_resume() {
    # Start bitcoind on existing $REGTEST_DIR WITHOUT wiping it.
    # Use this when the data dir is intact but the daemon isn't running.
    if [ ! -d "$REGTEST_DIR" ]; then
        echo "ERROR: $REGTEST_DIR does not exist. Use 'start' for a fresh node."
        return 1
    fi
    if [ ! -f "$REGTEST_DIR/bitcoin.conf" ]; then
        echo "  bitcoin.conf missing — regenerating (chain data is what matters)..."
        cat > "$REGTEST_DIR/bitcoin.conf" <<EOF
regtest=1
server=1
txindex=1
acceptnonstdtxn=1
fallbackfee=0.0001
maxtxfee=1
[regtest]
rpcport=${RPC_PORT}
rpcuser=qsb
rpcpassword=qsb_test_pw
EOF
        echo "  bitcoin.conf regenerated."
    fi
    if pgrep -f "bitcoind.*${REGTEST_DIR}" >/dev/null; then
        echo "bitcoind already running on $REGTEST_DIR"
        if $CLI getblockchaininfo >/dev/null 2>&1; then
            echo "  RPC is responding."
            $CLI getblockchaininfo | grep -E "blocks|chain" | head
            return 0
        else
            echo "  but RPC isn't responding. Try ./setup_regtest.sh stop && ./setup_regtest.sh resume."
            return 1
        fi
    fi

    echo "Resuming bitcoind in $REGTEST_DIR (preserving existing chain state)..."
    bitcoind -datadir="${REGTEST_DIR}" -daemon

    for i in $(seq 1 20); do
        if $CLI getblockchaininfo >/dev/null 2>&1; then break; fi
        sleep 1
    done
    if ! $CLI getblockchaininfo >/dev/null 2>&1; then
        echo "ERROR: bitcoind did not become ready"
        return 1
    fi
    info=$($CLI getblockchaininfo)
    blocks=$(echo "$info" | grep -E '"blocks"' | head -1 | awk '{print $2}' | tr -d ',')
    echo "  ready. blocks=$blocks"
    if $CLI -rpcwallet="$WALLET_NAME" getbalance >/dev/null 2>&1; then
        bal=$($CLI -rpcwallet="$WALLET_NAME" getbalance)
        echo "  wallet $WALLET_NAME balance: $bal BTC"
    else
        # Wallet might need to be loaded
        $CLI loadwallet "$WALLET_NAME" >/dev/null 2>&1 || true
        if $CLI -rpcwallet="$WALLET_NAME" getbalance >/dev/null 2>&1; then
            bal=$($CLI -rpcwallet="$WALLET_NAME" getbalance)
            echo "  wallet $WALLET_NAME loaded; balance: $bal BTC"
        else
            echo "  (wallet $WALLET_NAME not found — chain state preserved but no wallet)"
        fi
    fi
}


cmd_stop() {
    if pgrep -f "bitcoind.*${REGTEST_DIR}" >/dev/null; then
        echo "Stopping bitcoind..."
        $CLI stop 2>/dev/null || true
        # Wait up to 10 seconds for clean shutdown
        for i in $(seq 1 10); do
            if ! pgrep -f "bitcoind.*${REGTEST_DIR}" >/dev/null; then
                break
            fi
            sleep 1
        done
    fi
    if [ -d "$REGTEST_DIR" ]; then
        echo "Removing $REGTEST_DIR"
        rm -rf "$REGTEST_DIR"
    fi
}

cmd_cli() {
    $CLI "$@"
}

case "${1:-}" in
    start)  cmd_start ;;
    resume) cmd_resume ;;
    stop)   cmd_stop ;;
    cli)    shift; cmd_cli "$@" ;;
    *)
        echo "usage: $0 {start|resume|stop|cli ...}"
        echo "  start   — fresh node (DESTROYS existing data dir)"
        echo "  resume  — restart daemon on existing data dir (preserves chain state)"
        echo "  stop    — stop daemon and DESTROY data dir"
        echo "  cli ... — forward to bitcoin-cli"
        exit 1
        ;;
esac
