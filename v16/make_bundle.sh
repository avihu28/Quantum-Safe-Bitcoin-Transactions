#!/usr/bin/env bash
# Build qsb_v16.zip — the payload the orchestrator uploads to each rented machine.
#
# The orchestrator reads ./qsb_v16.zip, uploads it, and verifies the remote
# SHA-256 against its local copy. The zip is generated, not tracked: the
# authoritative sources live in bundle/ so they are readable and diffable.
#
# Usage:  ./make_bundle.sh
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d bundle ]; then
    echo "ERROR: bundle/ not found — run from the v16/ directory." >&2
    exit 1
fi

rm -f qsb_v16.zip
# -X strips extra file attributes so the archive is reproducible across machines.
( cd bundle && zip -qrX ../qsb_v16.zip . -x '.*' )

echo "built qsb_v16.zip ($(wc -c < qsb_v16.zip | tr -d ' ') bytes)"
# List the packed entries. Portable across BSD/GNU (no `head -n -N`).
unzip -Z1 qsb_v16.zip | sed 's/^/  /'
