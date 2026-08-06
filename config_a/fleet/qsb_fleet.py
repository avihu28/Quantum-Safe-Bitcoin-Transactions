#!/usr/bin/env python3
"""
qsb_fleet.py — one-click vast.ai fleet launcher for the QSB GPU search.

Flow:
    launch  — Search for offers, rent N instances, upload kernels+state,
              start an independent search on each with a unique (seq, lt) slice.
              Machines run fully in parallel with NO cross-machine sync.
    status  — Show each rented instance's current status + any hits reported.
    hits    — Pull *.log and qsb_hits.jsonl from all instances to a local dir.
    stop    — Stop (not destroy) all fleet instances.
    destroy — Destroy all fleet instances and stop incurring cost.

Setup (one-time):
    pip install vastai paramiko scp
    export VAST_API_KEY=<your key from https://cloud.vast.ai/manage-keys/>
    # or: mkdir -p ~/.config/vastai && echo "<your key>" > ~/.config/vastai/vast_api_key

Required files in the working directory (produced by the pipeline):
    qsb_state.json, qsb_scriptpubkey.hex
    pinning.bin, digest_r1.bin, digest_r2.bin
    gpu_pinning_params.json, gpu_digest_r1_params.json, gpu_digest_r2_params.json

The launcher tracks the fleet in a local state file (qsb_fleet.json), so all
subsequent commands (status, hits, stop, destroy) know which instances are ours.

Usage example:
    python3 qsb_fleet.py launch --count 10 --gpu RTX_4090 --max-dph 0.50
    python3 qsb_fleet.py status
    python3 qsb_fleet.py hits --out ./hits/
    python3 qsb_fleet.py destroy
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


FLEET_FILE = "qsb_fleet.json"
TARBALL = "qsb_search_bundle.tar.gz"

# Files that MUST exist in the working dir before launching
REQUIRED_FILES = [
    "qsb_state.json", "qsb_scriptpubkey.hex",
    "pinning.bin", "digest_r1.bin", "digest_r2.bin",
    "gpu_pinning_params.json",
    "gpu_digest_r1_params.json", "gpu_digest_r2_params.json",
]

# Files to include in the tarball uploaded to each instance
BUNDLE_FILES = REQUIRED_FILES + ["run_all.sh"]
# GPU kernel source files (uploaded separately from pipeline bundle)
KERNEL_SOURCES = ["qsb_digest_search.cu", "qsb_real_search.cu",
                  "GPUHash.h", "GPUMath.h"]

DEFAULT_IMAGE = "nvidia/cuda:12.3.0-devel-ubuntu22.04"


# =====================================================================
# Fleet state management
# =====================================================================

@dataclass
class Machine:
    instance_id: int
    offer_id: int
    gpu_name: str
    num_gpus: int
    dph: float
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    ssh_user: str = "root"
    # Search partition assigned to this machine
    seq_hex: Optional[str] = None
    lt_start: Optional[int] = None
    lt_range: Optional[int] = None
    global_gpu_offset: Optional[int] = None
    # Timestamps
    created_at: float = field(default_factory=time.time)
    last_seen: Optional[float] = None


def load_fleet() -> list[Machine]:
    if not Path(FLEET_FILE).exists():
        return []
    with open(FLEET_FILE) as f:
        data = json.load(f)
    return [Machine(**m) for m in data.get("machines", [])]


def save_fleet(machines: list[Machine]) -> None:
    with open(FLEET_FILE, "w") as f:
        json.dump({"machines": [asdict(m) for m in machines]}, f, indent=2)


# =====================================================================
# vast.ai CLI wrappers
# =====================================================================

def vastai(*args, check=True, capture=True) -> subprocess.CompletedProcess:
    """Run `vastai` CLI with args. Returns the CompletedProcess.
    Reads API key from VAST_API_KEY env or ~/.config/vastai/vast_api_key."""
    cmd = ["vastai", *map(str, args)]
    if capture:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    return subprocess.run(cmd, check=check, text=True)


def vastai_json(*args) -> list | dict:
    """Run `vastai <args> --raw` and return parsed JSON."""
    r = vastai(*args, "--raw", check=False)
    if r.returncode != 0:
        raise RuntimeError(f"vastai {args} failed:\n{r.stderr}")
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"could not parse vastai output as JSON: {e}\n{r.stdout[:500]}")


def search_offers(gpu_name, max_dph: float, min_ram: float = 8.0,
                  min_disk: float = 20.0, min_reliability: float = 0.98,
                  datacenter_only: bool = False, min_gpus: int = 1,
                  prefer_multi_gpu: bool = False) -> list[dict]:
    """Search for offers matching the criteria. Returns list of offer dicts.

    `gpu_name` can be a single GPU model (str) or a comma-separated list. For a
    list, results are merged across all GPU types and re-sorted by $/GPU/h.

    `max_dph` is the per-machine $/hour ceiling. For multi-GPU machines, this is
    the TOTAL price (e.g., an 8-GPU machine at $2.40/hr = $0.30/GPU/hr). Set
    `max_dph` accordingly when filtering for multi-GPU offers.
    """
    if isinstance(gpu_name, str):
        gpu_names = [g.strip() for g in gpu_name.split(',') if g.strip()]
    else:
        gpu_names = list(gpu_name)

    all_offers: list[dict] = []
    for gn in gpu_names:
        q = (f"gpu_name={gn} num_gpus>={min_gpus} dph<{max_dph} "
             f"reliability>{min_reliability} cpu_ram>={min_ram} "
             f"disk_space>={min_disk} rented=False")
        if datacenter_only:
            q += " datacenter=true"
        offers = vastai_json("search", "offers", q, "-o", "dph_total")
        if not isinstance(offers, list):
            raise RuntimeError(f"Unexpected offers response for {gn}: {offers}")
        all_offers.extend(offers)

    # Re-sort merged list. Default: by $/GPU/h ascending (cheapest-per-GPU
    # first). With prefer_multi_gpu: by num_gpus descending (16 > 14 > … > 1)
    # then by $/GPU/h ascending — so we prefer fewer-bigger machines and let
    # price break ties within each GPU-count bucket.
    if prefer_multi_gpu:
        all_offers.sort(key=lambda o: (
            -o.get('num_gpus', 1),
            o.get('dph_total', 1e9) / max(1, o.get('num_gpus', 1)),
        ))
    else:
        all_offers.sort(key=lambda o: (o.get('dph_total', 1e9) /
                                        max(1, o.get('num_gpus', 1))))
    return all_offers


def create_instance(offer_id: int, image: str = DEFAULT_IMAGE,
                    disk: int = 20, label: str = "qsb") -> int:
    """Rent the offer and return the new instance_id."""
    # NOTE: we use onstart-cmd to delay kernel start; we'll upload files via SSH
    # before kicking off the search.
    r = vastai_json(
        "create", "instance", offer_id,
        "--image", image,
        "--disk", disk,
        "--ssh", "--direct",
        "--label", label,
        "--onstart-cmd", "sleep infinity",  # keep container running; we drive it via SSH
    )
    if not r.get("success"):
        raise RuntimeError(f"create instance failed: {r}")
    # API returns new_contract as the instance id
    return int(r["new_contract"])


def show_instance(instance_id: int) -> dict:
    return vastai_json("show", "instance", instance_id)


def destroy_instance(instance_id: int) -> None:
    vastai("destroy", "instance", instance_id, check=False)


def stop_instance(instance_id: int) -> None:
    vastai("stop", "instance", instance_id, check=False)


# =====================================================================
# Bundling + SSH
# =====================================================================

def check_prereqs(work_dir: Path) -> None:
    missing = [f for f in REQUIRED_FILES if not (work_dir / f).exists()]
    if missing:
        print(f"ERROR: missing required files in {work_dir}: {missing}")
        print("Run `qsb_pipeline.py setup` and `qsb_pipeline.py export` first.")
        sys.exit(1)


def build_bundle(work_dir: Path, gpu_dir: Path) -> Path:
    """Create a tarball containing state, binary inputs, kernel sources, and run scripts.

    Optionally includes libssl-dev .deb files for offline fallback install. If the
    user pre-stages .deb files at <work_dir>/deb_fallback/*.deb, they are bundled
    and used by the bootstrap if apt-get fails. See bootstrap_libssl_install() in
    BOOTSTRAP_COMPILE for how this is consumed.
    """
    out = work_dir / TARBALL
    with tarfile.open(out, "w:gz") as tf:
        for f in BUNDLE_FILES:
            p = work_dir / f
            if p.exists():
                tf.add(p, arcname=f)
        for f in KERNEL_SOURCES:
            p = gpu_dir / f
            if p.exists():
                tf.add(p, arcname=f)
            else:
                print(f"WARN: kernel source {p} not found — bundle may be incomplete")
        # Include all the run_*.sh helpers from gpu_dir
        for f in ("run_all.sh", "run_pin.sh", "run_digest.sh"):
            p = gpu_dir / f
            if p.exists():
                tf.add(p, arcname=f)
        # Include any .deb fallback files (libssl-dev for offline install)
        deb_dir = work_dir / "deb_fallback"
        if deb_dir.is_dir():
            deb_count = 0
            for p in sorted(deb_dir.glob("*.deb")):
                tf.add(p, arcname=f"deb_fallback/{p.name}")
                deb_count += 1
            if deb_count > 0:
                print(f"  bundle: included {deb_count} .deb fallback files from "
                      f"{deb_dir}")
    return out


SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=10",
    "-o", "TCPKeepAlive=yes",
]


def wait_for_ssh(m: Machine, timeout: float = 720.0) -> bool:
    """Poll the instance until SSH responds. Up to 12 minutes by default —
    vast.ai images can be slow on first boot (image pull, container start,
    SSH key install). Most healthy machines come up within 3-7 minutes.
    Beyond 12 minutes, the instance is almost certainly stuck or dead.

    Returns True as soon as a single 'echo ready' round-trip succeeds. Updates
    m.ssh_host / m.ssh_port as soon as vast.ai reports them.

    Fails fast if the instance enters a terminal state (exited, errored,
    stopped, or offline-for-5+min).
    """
    t0 = time.time()
    last_status = None
    last_log_time = 0.0
    probe_count = 0
    while time.time() - t0 < timeout:
        try:
            info = show_instance(m.instance_id)
        except Exception as e:
            time.sleep(15)
            continue

        ssh_host = info.get("ssh_host")
        ssh_port = info.get("ssh_port")
        status = info.get("actual_status", "?")
        elapsed = int(time.time() - t0)

        # Fail fast on terminal / unrecoverable states. "offline" sometimes
        # recovers, but if it's been offline for >5 min we treat it as dead.
        # "exited" / "stopped" / "errored" are immediately terminal.
        terminal_now = {"exited", "errored", "stopped", "kicked"}
        terminal_eventually = {"offline"}
        if status in terminal_now:
            print(f"    [inst {m.instance_id}] ✗ status={status} (terminal) — giving up after {elapsed}s",
                  flush=True)
            return False
        if status in terminal_eventually and elapsed > 300:
            print(f"    [inst {m.instance_id}] ✗ status={status} for >5 min — giving up",
                  flush=True)
            return False

        # Periodic status print every 60s if status changed
        if elapsed - last_log_time > 60:
            print(f"    [inst {m.instance_id}] status={status} ssh={ssh_host}:{ssh_port} ({elapsed}s elapsed, probes={probe_count})",
                  flush=True)
            last_log_time = elapsed
            last_status = status

        if ssh_host and ssh_port:
            m.ssh_host = ssh_host
            m.ssh_port = int(ssh_port)
            probe_count += 1
            r = subprocess.run(
                ["ssh", *SSH_OPTS,
                 "-p", str(m.ssh_port),
                 f"{m.ssh_user}@{m.ssh_host}",
                 "echo ready"],
                capture_output=True, text=True, timeout=45,
            )
            if r.returncode == 0 and "ready" in r.stdout:
                print(f"    [inst {m.instance_id}] ✓ SSH ready ({elapsed}s, {probe_count} probes)",
                      flush=True)
                return True
        time.sleep(15)
    return False


def ssh_exec(m: Machine, cmd: str, timeout: float = 120.0,
             retries: int = 3) -> tuple[int, str, str]:
    """Run a command on the remote machine via ssh, with retries on transient errors.

    Retries on SSH-level failures (connection reset, timeout, kex failures).
    Does NOT retry on non-zero exit codes from the remote command itself —
    those are surfaced as-is.
    """
    last_rc, last_out, last_err = -1, "", "no attempts made"
    for attempt in range(retries):
        try:
            # Pass the cmd via stdin to a remote `bash -s` rather than
            # as an SSH-side argument. This avoids:
            #   - shell quoting issues with embedded $, {, ", \, etc.
            #   - default-shell variability (sshd might exec /bin/sh = dash)
            #   - command-line length limits
            # The remote sees a clean bash script over stdin.
            args = ["ssh", *SSH_OPTS,
                    "-p", str(m.ssh_port),
                    f"{m.ssh_user}@{m.ssh_host}",
                    "bash -s"]
            r = subprocess.run(args, input=cmd,
                                capture_output=True, text=True, timeout=timeout)
            # If SSH itself failed (255 = ssh error code), retry
            if r.returncode == 255 and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                last_rc, last_out, last_err = r.returncode, r.stdout, r.stderr
                continue
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired as e:
            last_rc = -1
            last_out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            last_err = f"ssh timeout after {timeout}s"
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
        except Exception as e:
            last_rc = -1
            last_err = f"ssh exception: {e}"
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
    return last_rc, last_out, last_err


def scp_upload(m: Machine, local: Path, remote: str, retries: int = 3) -> bool:
    """SCP a file with retries on transient network errors."""
    for attempt in range(retries):
        r = subprocess.run(
            ["scp", *SSH_OPTS, "-P", str(m.ssh_port),
             str(local), f"{m.ssh_user}@{m.ssh_host}:{remote}"],
            capture_output=True, text=True, timeout=900,
        )
        if r.returncode == 0:
            return True
        if attempt < retries - 1:
            time.sleep(5 * (attempt + 1))
    return False


def scp_download(m: Machine, remote: str, local_dir: Path,
                 retries: int = 3) -> bool:
    local_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        r = subprocess.run(
            ["scp", "-r", *SSH_OPTS, "-P", str(m.ssh_port),
             f"{m.ssh_user}@{m.ssh_host}:{remote}", str(local_dir) + "/"],
            capture_output=True, text=True, timeout=900,
        )
        if r.returncode == 0:
            return True
        if attempt < retries - 1:
            time.sleep(5 * (attempt + 1))
    return False


def scp_download_file(m: Machine, remote: str, local_path: Path,
                      retries: int = 3) -> bool:
    """Download a single remote file to a specific local path."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        r = subprocess.run(
            ["scp", *SSH_OPTS, "-P", str(m.ssh_port),
             f"{m.ssh_user}@{m.ssh_host}:{remote}", str(local_path)],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode == 0:
            return True
        if attempt < retries - 1:
            time.sleep(3)
    return False


# =====================================================================
# Search-space partitioning
# =====================================================================

def partition_search(num_gpus_per_machine: list[int],
                     seq_base: int = 0xfffffffe, seq_count: int = 1,
                     lt_start: int = 500_000_000, lt_total: int = 100_000_000):
    """Assign each machine a (sequence, locktime-range) slice based on its GPU count.

    `num_gpus_per_machine` is a list of GPU counts, one per machine. Machines
    with more GPUs get proportionally larger lt slices, so each individual GPU
    in the fleet does roughly the same amount of work.

    Returns a list of slice dicts (one per machine), in the same order as input.
    """
    slices = []
    total_gpus = sum(num_gpus_per_machine)
    if total_gpus == 0:
        return slices

    # Each GPU gets a share of total_lt roughly proportional to lt_total / total_gpus.
    # Each machine gets num_gpus_on_that_machine × per_gpu_share.
    per_gpu_lt = lt_total // total_gpus

    cum_offset_lt = lt_start
    cum_gpu_offset = 0
    for m_idx, ngpu in enumerate(num_gpus_per_machine):
        machine_lt = ngpu * per_gpu_lt
        seq = seq_base - (m_idx // len(num_gpus_per_machine))  # cycle if needed
        slices.append({
            "seq_hex": f"0x{seq & 0xffffffff:08x}",
            "lt_start": cum_offset_lt,
            "lt_range": machine_lt,
            "global_gpu_offset": cum_gpu_offset,
        })
        cum_offset_lt += machine_lt
        cum_gpu_offset += ngpu
    return slices


# =====================================================================
# Commands
# =====================================================================

def cmd_launch(args):
    work_dir = Path.cwd()
    gpu_dir = Path(args.gpu_dir).resolve()
    check_prereqs(work_dir)
    if not gpu_dir.exists():
        print(f"ERROR: --gpu-dir {gpu_dir} not found")
        sys.exit(1)

    print("─── searching for offers ───")
    try:
        offers = search_offers(
            gpu_name=args.gpu, max_dph=args.max_dph,
            min_reliability=args.min_reliability,
            datacenter_only=args.datacenter_only,
            min_gpus=args.min_gpus,
            prefer_multi_gpu=getattr(args, 'prefer_multi_gpu', False),
        )
    except Exception as e:
        print(f"vastai search failed: {e}")
        sys.exit(1)
    if len(offers) < args.count:
        print(f"WARN: found {len(offers)} offers but you asked for {args.count}")
        if len(offers) == 0:
            sys.exit(1)

    chosen = offers[: args.count]
    print(f"Chosen offers:")
    total_gpus_in_fleet = 0
    for o in chosen:
        ngpu = o.get('num_gpus', 1)
        total_gpus_in_fleet += ngpu
        per_gpu_dph = o.get('dph_total', 0.0) / max(ngpu, 1)
        print(f"  offer {o['id']}: {o.get('gpu_name')} × {ngpu}, "
              f"${o.get('dph_total'):.3f}/h (${per_gpu_dph:.3f}/GPU/h), "
              f"rel={o.get('reliability2', 0):.3f}, loc={o.get('geolocation', '?')}")

    total_dph = sum(o['dph_total'] for o in chosen)
    print(f"\nFleet: {total_gpus_in_fleet} GPUs across {len(chosen)} machines")
    print(f"Total max cost: ${total_dph:.2f}/hour "
          f"(${total_dph/max(total_gpus_in_fleet,1):.3f}/GPU/h average)")
    if not args.yes:
        resp = input(f"\nRent {len(chosen)} instances? [y/N] ")
        if resp.strip().lower() != "y":
            print("Aborted.")
            return

    print("\n─── building upload bundle ───")
    bundle = build_bundle(work_dir, gpu_dir)
    print(f"  bundle: {bundle} ({bundle.stat().st_size / 1024:.0f} KB)")

    # Partition the search space — each machine gets a slice proportional to its GPU count
    num_gpus_list = [o.get('num_gpus', 1) for o in chosen]
    slices = partition_search(
        num_gpus_per_machine=num_gpus_list,
        seq_base=args.seq_base, seq_count=1,
        lt_start=args.lt_start, lt_total=args.lt_total,
    )

    # ── Phase A: create all instances (fast: ~1s per vastai API call) ──
    print("\n─── creating instances ───")
    machines = []
    for i, (offer, slc) in enumerate(zip(chosen, slices)):
        ngpu = offer.get('num_gpus', 1)
        print(f"  [{i + 1}/{len(chosen)}] renting offer {offer['id']} "
              f"({ngpu} × {offer.get('gpu_name')})...", end='', flush=True)
        try:
            iid = create_instance(offer["id"], image=args.image, disk=args.disk,
                                   label=f"qsb-{i}")
        except Exception as e:
            print(f" ✗ {e}")
            continue
        m = Machine(
            instance_id=iid, offer_id=offer["id"],
            gpu_name=offer.get("gpu_name", "?"),
            num_gpus=ngpu,
            dph=offer.get("dph_total", 0.0),
            seq_hex=slc["seq_hex"], lt_start=slc["lt_start"],
            lt_range=slc["lt_range"], global_gpu_offset=slc["global_gpu_offset"],
        )
        machines.append(m)
        # Save incrementally so `destroy` works even if we crash before bootstrap
        save_fleet(machines)
        print(f" ✓ instance {iid}")

    if not machines:
        print("\nNo instances created. Aborting.")
        return

    # ── Phase B: SSH/upload/bootstrap all machines in parallel ──
    print(f"\n─── provisioning {len(machines)} instances in parallel ───")
    print(f"  (SSH boot ~1-3 min, kernel compile ~30-60 sec per machine)")
    import concurrent.futures
    import threading
    print_lock = threading.Lock()

    def provision_one(idx, m):
        def log(msg):
            with print_lock:
                print(f"  [inst {m.instance_id}] {msg}", flush=True)

        log("waiting for SSH...")
        if not wait_for_ssh(m, timeout=args.ssh_timeout):
            log("✗ SSH did not come up in time (keeping for manual inspection)")
            return ('ssh_timeout', m)
        log(f"SSH up at {m.ssh_host}:{m.ssh_port} — uploading bundle")
        if not scp_upload(m, bundle, "/root/bundle.tar.gz"):
            log("✗ scp upload failed")
            return ('scp_failed', m)
        log("running bootstrap (apt-get + nvcc compile)...")
        rc, out, err = ssh_exec(m, BOOTSTRAP_SCRIPT.format(
            seq=m.seq_hex, lt_start=m.lt_start,
            lt_range=m.lt_range, global_offset=m.global_gpu_offset,
            total_gpus=total_gpus_in_fleet,
            extra_flags="single_hash",
        ), timeout=600)
        if rc != 0:
            log(f"✗ bootstrap failed:\n{err[-500:]}")
            return ('bootstrap_failed', m)
        log("✓ bootstrap done — binaries compiled. Run `qsb_fleet.py search --reuse` to start the search.")
        return ('ok', m)

    results = {}
    # Up to 10 threads in parallel — vast.ai API isn't a bottleneck here, network
    # bandwidth and per-machine boot time are.
    max_workers = min(len(machines), 16)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(provision_one, i, m): m for i, m in enumerate(machines)}
        for fut in concurrent.futures.as_completed(futures):
            try:
                status, m = fut.result()
                results[m.instance_id] = status
            except Exception as e:
                m = futures[fut]
                with print_lock:
                    print(f"  [inst {m.instance_id}] ✗ exception: {e}", flush=True)
                results[m.instance_id] = 'exception'

    # Save updated fleet state (now with ssh_host/port from successful provisions)
    save_fleet(machines)

    # Summary
    ok_count = sum(1 for s in results.values() if s == 'ok')
    print(f"\n  Provisioned: {ok_count}/{len(machines)} machines successfully")
    failed = [iid for iid, s in results.items() if s != 'ok']
    if failed:
        print(f"  Failed: {failed}")
        print(f"  (these are still rented and billing — `destroy` removes everything)")

    # Write final fleet state
    save_fleet(machines)
    print(f"\n{len(machines)} machines in fleet (saved to {FLEET_FILE})")
    print("\nNext: `python3 qsb_fleet.py status` to monitor progress")


BOOTSTRAP_COMPILE = r"""
set +e
cd /root
mkdir -p qsb_run
cd qsb_run
rm -f bootstrap.ok bootstrap.err qsb_real.log qsb_digest.log qsb_real qsb_digest
tar -xzf /root/bundle.tar.gz

echo "[bootstrap] apt-get install build-essential libssl-dev..."
apt-get update -qq > /tmp/apt.log 2>&1
apt-get install -y -qq build-essential libssl-dev wget ca-certificates >> /tmp/apt.log 2>&1
cp /tmp/apt.log apt.log 2>/dev/null

if [ ! -f /usr/include/openssl/sha.h ]; then
    echo "FAILED: libssl-dev install failed" > bootstrap.err
    echo "BINARIES_MISSING (libssl-dev install failed)"
    exit 1
fi

echo "[bootstrap] which nvcc:"
which nvcc

if ! command -v nvcc >/dev/null 2>&1; then
    echo "FAILED: nvcc not found" > bootstrap.err
    echo "BINARIES_MISSING (nvcc missing)"
    exit 1
fi

echo "[bootstrap] compiling qsb_real..."
nvcc -O3 -o qsb_real qsb_real_search.cu -lcrypto -lm > qsb_real.log 2>&1
RC1=$?
echo "[bootstrap] qsb_real nvcc rc=$RC1"

echo "[bootstrap] compiling qsb_digest..."
nvcc -O3 -o qsb_digest qsb_digest_search.cu -lcrypto -lm > qsb_digest.log 2>&1
RC2=$?
echo "[bootstrap] qsb_digest nvcc rc=$RC2"

chmod +x run_all.sh run_pin.sh run_digest.sh 2>/dev/null

if [ -x qsb_real ] && [ -x qsb_digest ]; then
    touch bootstrap.ok
    echo "BINARIES_OK"
    exit 0
else
    [ -x qsb_real ] || echo "qsb_real_failed (rc=$RC1)" >> bootstrap.err
    [ -x qsb_digest ] || echo "qsb_digest_failed (rc=$RC2)" >> bootstrap.err
    echo "BINARIES_MISSING"
    exit 1
fi
""".strip()

START_PIN_SCRIPT = r"""
# Idempotent pin start — no `set -e`
cd /root/qsb_run || {{ echo "ERROR: /root/qsb_run missing" >&2; exit 1; }}
echo "[start_pin] killing any prior search..."
# IMPORTANT: -x (exact basename) NOT -f. -f would kill our own bash because the
# script body contains 'qsb_real' / 'qsb_digest'.
pkill -x qsb_real 2>/dev/null
pkill -x qsb_digest 2>/dev/null
sleep 1
rm -f pin_hit.json pin_status digest_r*_status digest_r*_hit.json
mkdir -p results
rm -f results/*.txt
if [ ! -x ./qsb_real ]; then
    echo "ERROR: qsb_real binary missing" >&2; exit 1
fi
chmod +x run_pin.sh 2>/dev/null
export QSB_TOTAL_GPUS="{total_gpus}"
export QSB_GLOBAL_OFFSET="{global_offset}"
nohup ./run_pin.sh {extra_flags} >/root/qsb_run/pin.log 2>&1 &
PID=$!
sleep 2
if kill -0 $PID 2>/dev/null; then
    echo "PIN_RUNNING pid=$PID"
else
    echo "ERROR: pin process died within 2s; tail of log:" >&2
    tail -20 /root/qsb_run/pin.log >&2
    exit 1
fi
""".strip()

START_DIGEST_SCRIPT = r"""
# Idempotent digest start — no `set -e` so we don't fail on missing files etc.
cd /root/qsb_run || {{ echo "ERROR: /root/qsb_run missing" >&2; exit 1; }}
echo "[start_digest] cwd=$(pwd)"
echo "[start_digest] killing any prior search..."
# IMPORTANT: use -x (exact basename) NOT -f (whole cmdline). The -f match would
# kill our OWN bash process, since the script body contains 'qsb_digest'.
pkill -x qsb_real 2>/dev/null
pkill -x qsb_digest 2>/dev/null
sleep 1
echo "[start_digest] cleaning old status files..."
rm -f digest_r{round}_status digest_r{round}_hit.json
mkdir -p results
rm -f results/digest_r{round}_*.txt
echo "[start_digest] checking qsb_digest binary..."
if [ ! -x ./qsb_digest ]; then
    echo "ERROR: qsb_digest binary missing or not executable" >&2
    ls -la qsb_digest 2>&1
    exit 1
fi
echo "[start_digest] checking run_digest.sh..."
if [ ! -x ./run_digest.sh ]; then
    chmod +x run_digest.sh 2>/dev/null
    if [ ! -x ./run_digest.sh ]; then
        echo "ERROR: run_digest.sh not executable" >&2
        exit 1
    fi
fi
export QSB_TOTAL_GPUS="{total_gpus}"
export QSB_GLOBAL_OFFSET="{global_offset}"
echo "[start_digest] launching nohup..."
nohup ./run_digest.sh {round} {sequence} {locktime} {extra_flags} \
    >/root/qsb_run/digest_r{round}.log 2>&1 &
PID=$!
echo "[start_digest] launched PID=$PID; sleeping 2s to confirm it stays alive..."
sleep 2
if kill -0 $PID 2>/dev/null; then
    echo "DIGEST_R{round}_RUNNING pid=$PID"
else
    echo "ERROR: digest process died within 2s; tail of log:" >&2
    tail -20 /root/qsb_run/digest_r{round}.log >&2
    exit 1
fi
""".strip()

# Backward-compat alias for callsites that still expect BOOTSTRAP_SCRIPT
# (the old monolithic compile+pin+digest sequence). Prefer the split scripts above.
BOOTSTRAP_SCRIPT = BOOTSTRAP_COMPILE


def cmd_refresh_ssh(args):
    """Re-query vast.ai for each instance's current ssh_host/ssh_port and
    update qsb_fleet.json. Vast.ai sometimes rotates SSH endpoints after
    boot (e.g., switching from a control-plane proxy to a direct port).
    Run this when SSH suddenly fails for the whole fleet."""
    machines = load_fleet()
    if not machines:
        print("No fleet on record.")
        return

    print(f"\n═══ REFRESH SSH ENDPOINTS ═══")
    print(f"  querying vast.ai for {len(machines)} instances...\n")

    updated = 0
    for m in machines:
        try:
            info = show_instance(m.instance_id)
        except Exception as e:
            print(f"  inst {m.instance_id}: ✗ vast API error: {e}")
            continue
        new_host = info.get("ssh_host")
        new_port = info.get("ssh_port")
        status = info.get("actual_status", "?")
        if not new_host or not new_port:
            print(f"  inst {m.instance_id}: status={status}, no ssh_host/port in API response")
            continue
        old = f"{m.ssh_host}:{m.ssh_port}"
        new = f"{new_host}:{new_port}"
        if old != new:
            print(f"  inst {m.instance_id}: status={status}, {old} -> {new} (UPDATED)")
            m.ssh_host = new_host
            m.ssh_port = int(new_port)
            updated += 1
        else:
            print(f"  inst {m.instance_id}: status={status}, {new} (unchanged)")

    save_fleet(machines)
    print(f"\n  Updated {updated} of {len(machines)} machines. Try `check-results` again.")


def cmd_status(args):
    machines = load_fleet()
    if not machines:
        print("No fleet on record.")
        return
    print(f"{'id':>10}  {'gpu':<12}  {'dph':>6}  {'status':<10}  {'progress':<30}")
    print("-" * 80)
    total_cost = 0.0
    for m in machines:
        try:
            info = show_instance(m.instance_id)
            status = info.get("actual_status", "?")
        except Exception:
            status = "err"
        # Tail the remote log to see last line of progress
        progress = ""
        if status == "running" and m.ssh_host and m.ssh_port:
            rc, out, err = ssh_exec(m, "tail -n 1 /root/qsb_run/qsb_search.log 2>/dev/null | head -c 80",
                                    timeout=10)
            if rc == 0:
                progress = out.strip()[:30]
        print(f"{m.instance_id:>10}  {m.gpu_name:<12}  {m.dph:>5.2f}  {status:<10}  {progress:<30}")
        if status == "running":
            total_cost += m.dph
    print(f"\nTotal burn rate: ${total_cost:.2f}/hour")


def cmd_hits(args):
    machines = load_fleet()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    found_any = False
    for m in machines:
        if not m.ssh_host or not m.ssh_port:
            continue
        sub = out_dir / f"instance_{m.instance_id}"
        sub.mkdir(parents=True, exist_ok=True)
        # Check if qsb_hits.jsonl has content
        rc, out, err = ssh_exec(m, "wc -l /root/qsb_run/qsb_hits.jsonl 2>/dev/null | awk '{print $1}'",
                                timeout=10)
        lines = 0
        if rc == 0:
            try:
                lines = int(out.strip())
            except ValueError:
                lines = 0
        if lines > 0:
            print(f"instance {m.instance_id}: {lines} hit(s) — downloading")
            scp_download(m, "/root/qsb_run/qsb_hits.jsonl", sub)
            scp_download(m, "/root/qsb_run/qsb_search.log", sub)
            scp_download(m, "/root/qsb_run/pin_gpu*.log", sub)
            scp_download(m, "/root/qsb_run/digest_r*_gpu*.log", sub)
            found_any = True
    if not found_any:
        print("No hits reported yet.")


def cmd_stop(args):
    """STOP (pause) all fleet instances. This powers down each vast.ai
    instance — SSH stops working until you run `start`. Storage is preserved
    (binaries, tile files, summary files all stay on disk). Billing is
    reduced to storage-only.

    To just kill running kernels WITHOUT pausing instances, use
    `kill-kernels` instead.
    """
    machines = load_fleet()
    print(f"Stopping (pausing) {len(machines)} instances. Storage is preserved.")
    print("Use `start` to resume, or `kill-kernels` if you only wanted to halt search.")
    for m in machines:
        print(f"stopping instance {m.instance_id}...")
        stop_instance(m.instance_id)


def cmd_start(args):
    """START (resume) all paused fleet instances and wait for SSH to respond.

    Instances powered down via `stop` keep their storage but lose network
    connectivity until resumed. This calls vastai start on each, then polls
    until SSH responds. Use after `stop` and before `restart-digest`.
    """
    machines = load_fleet()
    if not machines:
        print("No fleet on record.")
        return
    print(f"Starting (resuming) {len(machines)} instances...")
    for m in machines:
        print(f"  starting inst {m.instance_id}...")
        vastai("start", "instance", m.instance_id, check=False)

    print(f"\nWaiting for SSH on each instance (up to 12 min per machine)...")
    import concurrent.futures
    def wait_one(m):
        ok = wait_for_ssh(m, timeout=720)
        return (m, ok)
    ready, failed = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(machines), 16)) as ex:
        for m, ok in ex.map(wait_one, machines):
            if ok:
                ready.append(m)
                print(f"  ✓ inst {m.instance_id}: SSH ready")
            else:
                failed.append(m)
                print(f"  ✗ inst {m.instance_id}: SSH never came up")

    print()
    print(f"  {len(ready)}/{len(machines)} machines back up.")
    if failed:
        print(f"  {len(failed)} failed to resume:")
        for m in failed:
            print(f"      inst {m.instance_id}")
        print(f"  Consider destroying failed instances:")
        for m in failed:
            print(f"      vastai destroy instance {m.instance_id}")

    if ready:
        print(f"\nReady to run search/digest. Examples:")
        print(f"  python3 ... restart-digest <seq> <lt> --round 1")
        print(f"  python3 ... restart-digest <seq> <lt> --round 2")


def cmd_kill_kernels(args):
    """Kill any running qsb_real / qsb_digest processes on every fleet
    machine, WITHOUT pausing the instances themselves. Use this when you
    want to halt the search and immediately start a different one (e.g.,
    going from R1 to R2)."""
    machines = load_fleet()
    if not machines:
        print("No fleet on record.")
        return
    for m in machines:
        rc, out, err = ssh_exec(
            m,
            "pkill -x qsb_digest 2>/dev/null; pkill -x qsb_real 2>/dev/null; "
            "echo OK_KILLED",
            timeout=15)
        if rc == 0 and "OK_KILLED" in (out or ""):
            print(f"  ✓ inst {m.instance_id}: kernels killed")
        else:
            print(f"  ✗ inst {m.instance_id}: failed (rc={rc})")


def cmd_destroy(args):
    machines = load_fleet()
    # If --ids specified, destroy those specific instance IDs via vast.ai API
    # whether or not they appear in qsb_fleet.json. The launcher prunes failed
    # instances from the local fleet record but does NOT auto-destroy them on
    # vast.ai (they keep billing) — this lets us clean them up by ID anyway.
    if args.ids:
        wanted = sorted({int(x.strip()) for x in args.ids.split(',') if x.strip()})
        in_fleet = [m for m in machines if m.instance_id in wanted]
        not_in_fleet = sorted(set(wanted) - {m.instance_id for m in machines})
        if not args.yes:
            print(f"Will destroy {len(wanted)} instance(s) on vast.ai:")
            for m in in_fleet:
                print(f"  inst {m.instance_id}: {m.gpu_name} × {m.num_gpus} @ ${m.dph}/h  (in fleet)")
            for iid in not_in_fleet:
                print(f"  inst {iid}: ?  (not in qsb_fleet.json — destroying via API anyway)")
            resp = input(f"Proceed? [y/N] ")
            if resp.strip().lower() != "y":
                return
        # Destroy ALL wanted IDs (in-fleet + orphan) via the API
        for iid in wanted:
            print(f"destroying instance {iid}...")
            try:
                destroy_instance(iid)
            except Exception as e:
                print(f"  ✗ destroy {iid} failed: {e}")
        # Prune from local fleet record too (idempotent)
        remaining = [m for m in machines if m.instance_id not in wanted]
        save_fleet(remaining)
        print(f"\nDestroyed {len(wanted)} instance(s). {len(remaining)} remaining in fleet.")
        return
    # No --ids: destroy ALL (preserves prior behavior)
    if not args.yes:
        resp = input(f"Destroy {len(machines)} instances? Money spent is lost. [y/N] ")
        if resp.strip().lower() != "y":
            return
    for m in machines:
        print(f"destroying instance {m.instance_id}...")
        destroy_instance(m.instance_id)
    # Clear local state
    if Path(FLEET_FILE).exists():
        Path(FLEET_FILE).rename(Path(FLEET_FILE + ".bak"))
    print(f"\nAll instances destroyed. Fleet file renamed to {FLEET_FILE}.bak")


def _is_machine_searching(m: 'Machine') -> tuple[bool, str]:
    """Return (is_searching, reason). Heuristic: search is healthy if
    /root/qsb_run/qsb_search.log exists and contains a progress line.

    Used by reprovision --auto to decide which machines to re-bootstrap.
    """
    if not m.ssh_host or not m.ssh_port:
        return False, "no SSH info recorded"
    rc, out, err = ssh_exec(
        m,
        # Check three things in one ssh call:
        # 1. qsb_run dir exists
        # 2. qsb_search.log exists and has content
        # 3. a search binary is currently running
        "test -d /root/qsb_run && "
        "wc -l /root/qsb_run/qsb_search.log 2>/dev/null | awk '{print $1}' && "
        "pgrep -f 'qsb_real|qsb_digest' >/dev/null && echo SEARCHING || echo IDLE",
        timeout=15,
    )
    if rc != 0:
        return False, f"ssh failed (rc={rc})"
    lines = out.strip().split("\n")
    if not lines:
        return False, "empty ssh response"
    # First line is wc -l output, second is SEARCHING/IDLE
    log_lines = 0
    try:
        log_lines = int(lines[0])
    except (ValueError, IndexError):
        log_lines = 0
    is_running = lines[-1].strip() == "SEARCHING"
    if is_running and log_lines > 0:
        return True, f"running ({log_lines} log lines)"
    if is_running and log_lines == 0:
        return True, "running but log empty (just started?)"
    if not is_running and log_lines == 0:
        return False, "no search process, log empty"
    if not is_running:
        return False, f"no search process (log has {log_lines} lines — crashed?)"
    return False, "indeterminate"


def cmd_reprovision(args):
    """Re-run the bootstrap (apt + compile + start search) on broken instances.

    Useful when some machines failed to provision during launch (SSH timeout,
    bootstrap script error, etc.) but are still rented.

    Modes:
      --ids "12345,67890"   re-provision specific instance IDs
      --auto                detect machines that aren't searching, re-provision them
      --all                 re-provision EVERY machine in the fleet (nuclear option)
    """
    machines = load_fleet()
    if not machines:
        print("No fleet on record. Run `launch` first.")
        return

    bundle_path = Path(TARBALL).resolve()
    if not bundle_path.exists():
        # If we have --gpu-dir, rebuild the bundle from scratch
        if args.gpu_dir:
            gpu_dir = Path(args.gpu_dir).resolve()
            if not gpu_dir.exists():
                print(f"ERROR: --gpu-dir {gpu_dir} not found")
                sys.exit(1)
            check_prereqs(Path.cwd())
            print(f"Rebuilding bundle (no {TARBALL} in cwd)...")
            bundle_path = build_bundle(Path.cwd(), gpu_dir)
        else:
            print(f"ERROR: {TARBALL} not found in cwd, and --gpu-dir not provided.")
            print(f"  Either run from the same dir as your previous `launch` (which")
            print(f"  left {TARBALL} behind), or pass --gpu-dir to rebuild.")
            sys.exit(1)
    print(f"  bundle: {bundle_path} ({bundle_path.stat().st_size / 1024:.0f} KB)")

    total_gpus_in_fleet = sum(m.num_gpus for m in machines)

    # Decide which machines to reprovision
    if args.all:
        targets = list(machines)
        print(f"Reprovisioning ALL {len(targets)} machines.")
    elif args.ids:
        wanted = {int(x.strip()) for x in args.ids.split(",")}
        targets = [m for m in machines if m.instance_id in wanted]
        missing = wanted - {m.instance_id for m in targets}
        if missing:
            print(f"WARN: instance IDs not in fleet: {sorted(missing)}")
        if not targets:
            print("No matching instances.")
            return
        print(f"Reprovisioning {len(targets)} specific machines: "
              f"{[m.instance_id for m in targets]}")
    elif args.auto:
        print(f"Probing all {len(machines)} machines to find broken ones...")
        targets = []
        # Probe in parallel — much faster than serial
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            futures = {ex.submit(_is_machine_searching, m): m for m in machines}
            for fut in concurrent.futures.as_completed(futures):
                m = futures[fut]
                try:
                    healthy, reason = fut.result()
                except Exception as e:
                    healthy, reason = False, f"probe exception: {e}"
                tag = "✓ healthy" if healthy else "✗ broken"
                print(f"  inst {m.instance_id}: {tag}  ({reason})")
                if not healthy:
                    targets.append(m)
        if not targets:
            print("\nAll machines look healthy. Nothing to reprovision.")
            return
        print(f"\nFound {len(targets)} broken machines.")
    else:
        print("ERROR: must specify one of --ids, --auto, or --all")
        sys.exit(1)

    if not args.yes:
        resp = input(f"\nReprovision {len(targets)} instances? [y/N] ")
        if resp.strip().lower() != "y":
            print("Aborted.")
            return

    # Reprovision in parallel — same logic as Phase B of cmd_launch
    print(f"\n─── reprovisioning {len(targets)} instances in parallel ───")
    import concurrent.futures
    import threading
    print_lock = threading.Lock()

    def reprovision_one(m: 'Machine'):
        def log(msg):
            with print_lock:
                print(f"  [inst {m.instance_id}] {msg}", flush=True)

        # Make sure SSH info is fresh (the machine may have re-IP'd since launch)
        log("waiting for SSH...")
        if not wait_for_ssh(m, timeout=args.ssh_timeout):
            log("✗ SSH did not come up")
            return ('ssh_timeout', m)

        log(f"SSH at {m.ssh_host}:{m.ssh_port} — killing any stale search procs")
        ssh_exec(m, "pkill -x qsb_real 2>/dev/null; pkill -x qsb_digest 2>/dev/null; "
                    "sleep 1; rm -rf /root/qsb_run", timeout=30)

        log("uploading bundle...")
        if not scp_upload(m, bundle_path, "/root/bundle.tar.gz"):
            log("✗ scp failed")
            return ('scp_failed', m)

        log("running bootstrap (apt + nvcc compile, ~2-3 min)...")
        rc, out, err = ssh_exec(m, BOOTSTRAP_SCRIPT.format(
            seq=m.seq_hex, lt_start=m.lt_start, lt_range=m.lt_range,
            global_offset=m.global_gpu_offset,
            total_gpus=total_gpus_in_fleet,
            extra_flags="single_hash",
        ), timeout=600)
        if rc != 0:
            log(f"✗ bootstrap failed (rc={rc}):")
            log(f"  stderr: {err[-300:]}")
            log(f"  stdout: {out[-300:]}")
            return ('bootstrap_failed', m)
        log("✓ search started")
        return ('ok', m)

    results = {}
    max_workers = min(len(targets), 16)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(reprovision_one, m): m for m in targets}
        for fut in concurrent.futures.as_completed(futures):
            try:
                status, m = fut.result()
                results[m.instance_id] = status
            except Exception as e:
                m = futures[fut]
                with print_lock:
                    print(f"  [inst {m.instance_id}] ✗ exception: {e}", flush=True)
                results[m.instance_id] = 'exception'

    save_fleet(machines)

    ok = sum(1 for s in results.values() if s == 'ok')
    print(f"\n  Reprovisioned: {ok}/{len(targets)} successfully")
    failed = [(iid, s) for iid, s in results.items() if s != 'ok']
    if failed:
        print(f"  Still broken:")
        for iid, s in failed:
            print(f"    inst {iid}: {s}")
        print(f"  These are still rented. Consider destroying them via the vast.ai")
        print(f"  GUI individually if reprovisioning keeps failing.")


# =====================================================================
# Main
# =====================================================================

# =====================================================================
# Coordinated search (pin → digest pivot)
# =====================================================================

def _ssh_run_remote_script(m: 'Machine', script_text: str, timeout: float = 600.0):
    """Run a multiline shell script on the remote machine."""
    return ssh_exec(m, script_text, timeout=timeout)


def _read_remote_file(m: 'Machine', remote_path: str) -> Optional[str]:
    """cat a remote file. Returns content as string, or None if missing."""
    rc, out, err = ssh_exec(m, f"cat {remote_path} 2>/dev/null", timeout=15)
    if rc != 0 or not out.strip():
        return None
    return out


def _bootstrap_compile(m: 'Machine', bundle_path: Path,
                       lock: 'threading.Lock') -> tuple[str, 'Machine']:
    """Upload bundle and compile kernels on one machine. After the attempt:
       - download build logs locally (whether it succeeded or not)
       - check whether qsb_real and qsb_digest binaries exist on remote
       - return ('ok', m) if both binaries exist, regardless of nvcc warnings/exit code
    """
    def log(msg):
        with lock:
            print(f"  [inst {m.instance_id}] {msg}", flush=True)

    log("waiting for SSH (up to 12 min)...")
    if not wait_for_ssh(m, timeout=720):
        log("✗ SSH never came up after 12 minutes (instance probably dead)")
        return ('ssh_timeout', m)

    log(f"SSH ready at {m.ssh_host}:{m.ssh_port}; uploading bundle (~80 KB)...")
    if not scp_upload(m, bundle_path, "/root/bundle.tar.gz", retries=5):
        log("✗ scp upload failed after 5 retries")
        return ('scp_failed', m)

    # Run bootstrap. We DON'T trust its exit code for success detection;
    # we'll directly check whether the binaries exist afterward.
    log("running bootstrap (apt + nvcc, ~3-5 min)...")
    rc, out, err = ssh_exec(m, BOOTSTRAP_COMPILE, timeout=900, retries=2)

    # Pull diagnostic files locally for inspection.
    diag_dir = Path("fleet_diag") / f"inst_{m.instance_id}"
    diag_dir.mkdir(parents=True, exist_ok=True)
    for remote, local_name in [
        ("/root/qsb_run/qsb_real.log", "qsb_real.log"),
        ("/root/qsb_run/qsb_digest.log", "qsb_digest.log"),
        ("/root/qsb_run/bootstrap.ok", "bootstrap.ok"),
        ("/root/qsb_run/bootstrap.err", "bootstrap.err"),
        ("/root/qsb_run/apt.log", "apt.log"),
    ]:
        scp_download_file(m, remote, diag_dir / local_name, retries=2)
    # Save bootstrap stdout/stderr too
    (diag_dir / "bootstrap_stdout.txt").write_text(out or "")
    (diag_dir / "bootstrap_stderr.txt").write_text(err or "")

    # GROUND TRUTH: do the binaries exist on the remote?
    rc_check, out_check, _ = ssh_exec(
        m,
        "ls -la /root/qsb_run/qsb_real /root/qsb_run/qsb_digest 2>/dev/null && "
        "echo BINARIES_OK || echo BINARIES_MISSING",
        timeout=30,
    )

    binaries_ok = "BINARIES_OK" in out_check
    if binaries_ok:
        log(f"✓ binaries built (bootstrap rc={rc}, but kernels exist)")
        return ('ok', m)

    # Compile failed for real. Print actionable diagnostics.
    log(f"✗ binaries NOT built. Bootstrap rc={rc}")
    real_log = (diag_dir / "qsb_real.log").read_text() if (diag_dir / "qsb_real.log").exists() else ""
    digest_log = (diag_dir / "qsb_digest.log").read_text() if (diag_dir / "qsb_digest.log").exists() else ""
    if real_log:
        log(f"  qsb_real.log (last 2000 chars):")
        for line in real_log[-2000:].split("\n"):
            log(f"    {line}")
    elif digest_log:
        log(f"  qsb_digest.log (last 2000 chars):")
        for line in digest_log[-2000:].split("\n"):
            log(f"    {line}")
    else:
        log(f"  (no compile log retrievable; bootstrap stdout last 1000 chars:)")
        for line in (out or "")[-1000:].split("\n"):
            log(f"    {line}")
    log(f"  full diagnostics saved to: {diag_dir}/")
    return ('compile_failed', m)


def _start_pin(m: 'Machine', total_gpus: int, extra_flags: str = "single_hash") -> bool:
    """Kick off pinning search on one machine (background, non-blocking).
    Verifies success by checking that a qsb_real BINARY (not just any process
    matching 'qsb_real') is running.
    """
    # NOTE: retries=1 (no retry). The start script kills any prior search before
    # spawning new ones. If we retry, the second attempt kills the FIRST attempt's
    # workers and deletes its log files. Bad.
    rc, out, err = ssh_exec(m, START_PIN_SCRIPT.format(
        total_gpus=total_gpus, global_offset=m.global_gpu_offset,
        extra_flags=extra_flags,
    ), timeout=120, retries=1)
    print(f"  [inst {m.instance_id}] start_pin script output:")
    for line in (out or "").rstrip().split("\n"):
        print(f"    {line}")
    if err and err.strip():
        print(f"  [inst {m.instance_id}] start_pin stderr:")
        for line in err.rstrip().split("\n"):
            print(f"    {line}")
    # Wait for binaries to spawn
    time.sleep(5)
    # Use pgrep -x for exact basename match (NOT -f which matches the whole cmdline
    # and gets false positives from SSH commands containing the string 'qsb_real')
    rc2, out2, _ = ssh_exec(m,
        "n=$(pgrep -x qsb_real | wc -l); echo PIN_PROCS=$n",
        timeout=15, retries=2)
    nproc_match = re.search(r"PIN_PROCS=(\d+)", out2 or "")
    nproc = int(nproc_match.group(1)) if nproc_match else 0
    if nproc > 0:
        print(f"  [inst {m.instance_id}] ✓ {nproc} qsb_real processes running")
        return True
    print(f"  [inst {m.instance_id}] ✗ no qsb_real processes running after start. {out2!r}")
    return False


def _start_digest(m: 'Machine', round_num: int, sequence: str, locktime: int,
                  total_gpus: int) -> bool:
    """Kick off digest round N on one machine (background, non-blocking).
    Verifies success by checking that qsb_digest BINARIES (one per GPU) are running.
    """
    # NOTE: retries=1 (no retry) — start scripts are NOT safely retryable since
    # they kill any prior search before spawning new ones.
    rc, out, err = ssh_exec(m, START_DIGEST_SCRIPT.format(
        round=round_num, sequence=sequence, locktime=locktime,
        total_gpus=total_gpus, global_offset=m.global_gpu_offset,
        extra_flags="single_hash",
    ), timeout=120, retries=1)
    print(f"  [inst {m.instance_id}] start_digest r{round_num} script output:")
    for line in (out or "").rstrip().split("\n"):
        print(f"    {line}")
    if err and err.strip():
        print(f"  [inst {m.instance_id}] start_digest r{round_num} stderr:")
        for line in err.rstrip().split("\n"):
            print(f"    {line}")
    # Wait longer than _start_pin: run_digest.sh launches 1 qsb_digest per GPU, that 
    # can take 10-15s on machines with many GPUs to all spawn (CUDA init time).
    time.sleep(8)
    # Use pgrep -x to match the basename only — avoids matching SSH commands or 
    # source filenames that contain 'qsb_digest'.
    rc2, out2, _ = ssh_exec(m,
        "n=$(pgrep -x qsb_digest | wc -l); echo DIGEST_PROCS=$n",
        timeout=15, retries=2)
    nproc_match = re.search(r"DIGEST_PROCS=(\d+)", out2 or "")
    nproc = int(nproc_match.group(1)) if nproc_match else 0
    if nproc > 0:
        print(f"  [inst {m.instance_id}] ✓ {nproc} qsb_digest processes running")
        return True
    # Failure — print everything we know
    print(f"  [inst {m.instance_id}] ✗ no qsb_digest processes running. {out2!r}")
    log = _read_remote_file(m, f"/root/qsb_run/digest_r{round_num}.log")
    if log:
        print(f"  [inst {m.instance_id}] digest_r{round_num}.log (last 800 chars):")
        for line in log[-800:].split("\n"):
            print(f"    {line}")
    else:
        print(f"  [inst {m.instance_id}] /root/qsb_run/digest_r{round_num}.log doesn't exist")
    # Also try to grab GPU 0 log
    gpu0_log = _read_remote_file(m, f"/root/qsb_run/results/digest_r{round_num}_gpu0.log")
    if gpu0_log:
        print(f"  [inst {m.instance_id}] gpu0.log (last 500 chars):")
        for line in gpu0_log[-500:].split("\n"):
            print(f"    {line}")
    return False


def _check_pin_hit(m: 'Machine') -> Optional[dict]:
    """Return parsed pin_hit.json if the machine has found one, else None."""
    raw = _read_remote_file(m, "/root/qsb_run/pin_hit.json")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _resolve_funding_info(args) -> tuple[Optional[str], int]:
    """Resolve funding_txid + vout. Returns (txid, vout) or (None, 0) if unknown."""
    if getattr(args, 'funding_txid', None):
        return (args.funding_txid, getattr(args, 'funding_vout', 0))
    # Try regtest_funding.json in cwd
    p = Path('regtest_funding.json')
    if p.exists():
        try:
            with open(p) as f:
                rf = json.load(f)
            txid = rf.get('funding_txid') or rf.get('txid')
            vout = rf.get('vout', 0)
            if txid:
                return (txid, vout)
        except Exception:
            pass
    return (None, 0)


def _verify_pin_on_cpu(seq_hex: str, locktime: int, funding_txid: str,
                       funding_vout: int = 0,
                       work_dir: str = '.',
                       gpu_hit_file: Optional[str] = None) -> tuple[bool, str]:
    """Run verify_hit.py locally to confirm a pin claim reproduces on CPU.

    This is the CRITICAL GATE. We ran into a case where the GPU reported a
    pin hit that, on CPU re-derivation, produced no valid DER signature for
    any (recid, r_try) combination. Running digest against a phantom pin
    burns money and finds nothing. Catch it here.

    If gpu_hit_file is supplied, verify_hit will also compare the GPU-reported
    pubkey, SHA256(pk), and sighash against CPU-computed values, helping
    localize any divergence.

    Returns (ok, message). ok=True only if verify_hit returns rc=0.
    """
    verify_script = Path(__file__).resolve().parent.parent / 'verify' / 'verify_hit.py'
    if not verify_script.exists():
        return (False, f"verify_hit.py not found at {verify_script}")

    # Normalize seq to 0x-prefixed hex for the CLI
    if isinstance(seq_hex, int):
        seq_arg = f"0x{seq_hex:08x}"
    elif seq_hex.lower().startswith('0x'):
        seq_arg = seq_hex
    elif all(c in '0123456789abcdefABCDEF' for c in seq_hex):
        seq_arg = '0x' + seq_hex
    else:
        seq_arg = hex(int(seq_hex))

    cmd = [
        sys.executable, str(verify_script),
        '--work-dir', work_dir,
        'pin',
        '--locktime', str(locktime),
        '--sequence', seq_arg,
        '--funding-txid', funding_txid,
        '--funding-vout', str(funding_vout),
    ]
    if gpu_hit_file:
        cmd += ['--gpu-hit-file', gpu_hit_file]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return (False, "verify_hit.py timed out (>60s)")
    except Exception as e:
        return (False, f"verify_hit.py failed to launch: {e}")

    output = (r.stdout or '') + (r.stderr or '')
    if r.returncode == 0:
        return (True, output)
    return (False, output)


def _verify_digest_on_cpu(round_num: int, indices_csv: str,
                           seq_hex: str, locktime: int,
                           gpu_sighash: Optional[str] = None,
                           gpu_pubhash: Optional[str] = None,
                           work_dir: str = '.') -> bool:
    """Run verify_digest_against_kernel.py to confirm a digest hit reproduces
    on CPU. Catches kernel false positives (e.g., kernel computes sighash
    against a corrupted preimage and finds a hit that won't satisfy the
    actual script).

    Returns True iff one of the two index interpretations (state or storage)
    produces a sighash that matches gpu_sighash. Without gpu_sighash, this
    returns False (we can't be sure).
    """
    if not gpu_sighash:
        # Without the GPU's claimed sighash, we can only check whether the
        # indices SOMEWHERE in either interpretation produce a valid DER.
        # That's a much weaker check. For now, refuse to verify silently —
        # caller should treat False as "needs manual inspection".
        print("    (no gpu_sighash on hit; skipping sighash equality check)")
        return False

    verify_script = (Path(__file__).resolve().parent.parent
                     / 'verify' / 'verify_digest_against_kernel.py')
    if not verify_script.exists():
        print(f"    verify_digest_against_kernel.py not found at {verify_script}")
        return False

    # Normalize seq
    if isinstance(seq_hex, int):
        seq_arg = f"0x{seq_hex:08x}"
    elif seq_hex.lower().startswith('0x'):
        seq_arg = seq_hex
    else:
        seq_arg = hex(int(seq_hex))

    cmd = [
        sys.executable, str(verify_script),
        '--round', str(round_num),
        '--indices', str(indices_csv),
        '--sequence', seq_arg,
        '--locktime', str(locktime),
        '--gpu-sighash', gpu_sighash,
    ]
    if gpu_pubhash:
        cmd += ['--gpu-pubhash', gpu_pubhash]

    print(f"\n  ─── Verifying R{round_num} hit on CPU ───")
    print(f"    indices={indices_csv} seq={seq_arg} lt={locktime}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                           cwd=work_dir)
    except subprocess.TimeoutExpired:
        print("    verify_digest_against_kernel.py timed out (>60s)")
        return False
    except Exception as e:
        print(f"    failed to launch verifier: {e}")
        return False

    out = r.stdout or ''
    # The verifier prints "✓ MATCH" on either Interpretation A or B if the
    # sighash matches. If neither matches, it prints "NEITHER interpretation
    # matches".
    if '✓ MATCH' in out:
        print("    ✓ R{} hit verified on CPU.".format(round_num))
        # Print which interpretation matched + state indices for assembly
        for line in out.splitlines():
            if 'MATCH' in line or 'state indices' in line.lower() or 'STATE INDICES' in line:
                print(f"    {line.strip()}")
        return True
    print("    ✗ Neither interpretation matched the GPU's sighash.")
    print(out[-1500:])
    return False


def _check_digest_hit(m: 'Machine', round_num: int) -> Optional[dict]:
    raw = _read_remote_file(m, f"/root/qsb_run/digest_r{round_num}_hit.json")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _fetch_digest_hit_details(m: 'Machine', round_num: int) -> dict:
    """Pull the kernel's per-GPU summary files from `m` and extract sighash,
    pubhash, indices, etc., for any GPU that reported a HIT. Returns a dict
    suitable for passing to _verify_digest_on_cpu. Empty dict if nothing found.

    This is the BACKUP path used when the digest_rN_hit.json written by
    run_digest.sh doesn't include the sighash field (e.g. because an older
    run_digest.sh version is on the remote). The summary files are written
    by the kernel directly and always contain full hit details.
    """
    # List per-GPU summary files
    rc, out, _ = ssh_exec(m, "ls /root/qsb_run/results/digest_summary_gpu*.txt 2>/dev/null", timeout=15)
    if rc != 0 or not out.strip():
        return {}
    summary_files = [f.strip() for f in out.strip().split('\n') if f.strip()]
    for sf in summary_files:
        body = _read_remote_file(m, sf)
        if not body:
            continue
        # Parse FOUND blocks: lines like
        #   STATUS=FOUND ...
        #   indices=10,20,...
        #   sighash=...
        #   pubhash=...
        #   recid=...
        #   hash_choice=...
        if 'STATUS=FOUND' not in body and 'status=FOUND' not in body and 'hits=1' not in body.lower():
            continue
        out_dict: dict = {}
        for line in body.split('\n'):
            line = line.strip()
            for key in ('indices', 'sighash', 'pubhash', 'recid', 'hash_choice', 'combo_idx'):
                if line.startswith(key + '='):
                    val = line.split('=', 1)[1].strip()
                    out_dict[key] = val
                    break
        # Need at least indices + sighash to be useful
        if out_dict.get('indices') and out_dict.get('sighash'):
            return out_dict
    return {}


def _kill_kernels_on_fleet(machines: list, label: str = "kernels") -> None:
    """Kill any running qsb_real / qsb_digest processes across the fleet."""
    print(f"  killing {label} on {len(machines)} machines...")
    import concurrent.futures as _cf
    def kill_one(m):
        ssh_exec(m, "pkill -x qsb_real 2>/dev/null; pkill -x qsb_digest 2>/dev/null; true",
                 timeout=15)
    with _cf.ThreadPoolExecutor(max_workers=min(len(machines), 16)) as ex:
        list(ex.map(kill_one, machines))
    print(f"  ✓ killed {label} on all {len(machines)} machines")


def _check_status_file(m: 'Machine', filename: str) -> str:
    """Return 'running'/'found'/'exhausted'/'unknown' for a remote status file."""
    raw = _read_remote_file(m, f"/root/qsb_run/{filename}")
    if not raw:
        return "unknown"
    return raw.strip()


def _poll_for_first_hit(machines: list['Machine'], hit_checker, status_file: str,
                        poll_interval: float = 30.0,
                        timeout: float = 4 * 3600.0) -> tuple[Optional[dict], 'Machine']:
    """Poll all machines in parallel until one returns a hit (via hit_checker), or
    all machines finish their slice (status='exhausted'), or timeout.

    hit_checker(machine) → dict|None
    Returns (hit_dict, machine_that_found_it) or (None, None) if exhausted/timeout.
    """
    import concurrent.futures
    import time
    t0 = time.time()
    last_status_print = 0.0
    while time.time() - t0 < timeout:
        # Probe all machines in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            futures = {ex.submit(hit_checker, m): m for m in machines}
            statuses = {}
            for fut in concurrent.futures.as_completed(futures):
                m = futures[fut]
                try:
                    hit = fut.result()
                except Exception:
                    hit = None
                if hit is not None:
                    return hit, m
                statuses[m.instance_id] = _check_status_file(m, status_file)

        # Are all machines done without a hit?
        done_count = sum(1 for s in statuses.values() if s in ('found', 'exhausted'))
        running_count = sum(1 for s in statuses.values() if s == 'running')
        unknown_count = len(statuses) - done_count - running_count

        if done_count == len(machines):
            return None, None  # all exhausted, no hit

        # Periodic status print
        elapsed = time.time() - t0
        if elapsed - last_status_print > 60:
            print(f"  [{int(elapsed)}s] running={running_count} "
                  f"done={done_count} unknown={unknown_count}", flush=True)
            last_status_print = elapsed

        time.sleep(poll_interval)
    return None, None


def cmd_debug_digest(args):
    """Run digest debug-mode dump on remote, run CPU-side equivalent locally,
    diff side-by-side. Same approach as cmd_debug_pin.

    Usage:
        qsb_fleet.py debug-digest <round> <subset_csv> <seq_hex> <locktime>
    """
    machines = load_fleet()
    if not machines:
        print("No fleet on record.")
        return

    # Pick first machine that has the digest binary
    target = None
    for m in machines:
        rc, out, err = ssh_exec(m, "ls /root/qsb_run/qsb_digest 2>/dev/null", timeout=10)
        if rc == 0 and "qsb_digest" in (out or ""):
            target = m; break
    if target is None:
        print("✗ No machine has qsb_digest binary built.")
        return
    print(f"Using machine inst {target.instance_id} for debug dump.")

    seq_hex = args.seq if args.seq.startswith("0x") else f"0x{args.seq}"
    cmd = (f"cd /root/qsb_run && ./qsb_digest digest_r{args.round}.bin 0 "
           f"{seq_hex} {args.lt} 1 0 single_hash debug {args.subset} 2>&1")
    print(f"\n─── Running GPU debug kernel on inst {target.instance_id} ───")
    print(f"  cmd: {cmd}")
    rc, out, err = ssh_exec(target, cmd, timeout=120)
    gpu_lines = [ln for ln in (out or "").splitlines() if ln.startswith("DBG:")]
    print(f"  → {len(gpu_lines)} DBG lines from GPU")
    if not gpu_lines:
        print(f"  ✗ No DBG output. Full stdout:\n{out}\n  stderr:\n{err}")
        return

    print(f"\n─── Running CPU debug locally ───")
    debug_script = (Path(__file__).resolve().parent.parent / "verify" /
                    "debug_digest_intermediate.py")
    cmd2 = [
        sys.executable, str(debug_script),
        str(args.round), args.subset,
        "--work-dir", ".",
    ]
    r = subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
    cpu_lines = [ln for ln in (r.stdout or "").splitlines() if ln.startswith("DBG:")]
    print(f"  → {len(cpu_lines)} DBG lines from CPU")
    if not cpu_lines:
        print(f"  ✗ CPU dump failed. stdout: {r.stdout}\n  stderr: {r.stderr}")
        return

    def to_map(lines):
        m = {}
        for ln in lines:
            if "=" in ln:
                k, _, v = ln[len("DBG:"):].partition("=")
                m[k.strip()] = v.strip()
        return m

    gpu_m = to_map(gpu_lines); cpu_m = to_map(cpu_lines)
    keys, seen = [], set()
    for ln in gpu_lines:
        if "=" in ln:
            k = ln[len("DBG:"):].partition("=")[0].strip()
            if k not in seen: keys.append(k); seen.add(k)
    for ln in cpu_lines:
        if "=" in ln:
            k = ln[len("DBG:"):].partition("=")[0].strip()
            if k not in seen: keys.append(k); seen.add(k)

    print(f"\n─── Side-by-side comparison ({len(keys)} fields) ───\n")
    width = max(len(k) for k in keys) if keys else 0
    first_div = None
    for k in keys:
        gv = gpu_m.get(k, "<missing>"); cv = cpu_m.get(k, "<missing>")
        match = (gv == cv)
        flag = "  " if match else "✗ "
        if not match and first_div is None: first_div = k
        print(f"  {flag}{k:<{width}}")
        print(f"      GPU: {gv}")
        print(f"      CPU: {cv}")
    print()
    if first_div:
        print(f"  ⚠ FIRST DIVERGENCE: {first_div}")
        print(f"    Bug is in or before computation of: {first_div}")
    else:
        print(f"  ✓ All fields match. The bug is NOT in the dumped pipeline.")


def cmd_debug_pin(args):
    """Run the kernel's debug-mode dump for a specific (seq, lt) on one of the
    running machines, then run the equivalent CPU computation locally, and
    print both side-by-side so you can localize divergence.

    Usage:
        qsb_fleet.py debug-pin 0x80006137 1317906633 --funding-txid <hex>
    """
    machines = load_fleet()
    if not machines:
        print("No fleet on record.")
        return

    # Pick first machine that has qsb_real binary built
    target = None
    for m in machines:
        rc, out, err = ssh_exec(m, "ls /root/qsb_run/qsb_real 2>/dev/null", timeout=10)
        if rc == 0 and "qsb_real" in (out or ""):
            target = m
            break
    if target is None:
        print("✗ No machine has qsb_real binary built. Run search first.")
        return
    print(f"Using machine inst {target.instance_id} for debug dump.")

    # Run GPU debug kernel on remote
    seq_hex = args.seq if args.seq.startswith("0x") else f"0x{args.seq}"
    cmd = (f"cd /root/qsb_run && ./qsb_real pinning.bin 0 single_hash "
           f"debug {seq_hex} {args.lt} 2>&1")
    print(f"\n─── Running GPU debug kernel on inst {target.instance_id} ───")
    print(f"  cmd: {cmd}")
    rc, out, err = ssh_exec(target, cmd, timeout=60)
    gpu_lines = [ln for ln in (out or "").splitlines() if ln.startswith("DBG:")]
    print(f"  → {len(gpu_lines)} DBG lines from GPU")
    if not gpu_lines:
        print(f"  ✗ No DBG output. Full stdout:\n{out}\n  stderr:\n{err}")
        return

    # Run CPU debug locally
    print(f"\n─── Running CPU debug locally ───")
    fund_txid, fund_vout = _resolve_funding_info(args)
    if not fund_txid:
        print("  ✗ No funding txid (need --funding-txid or regtest_funding.json)")
        return
    debug_script = (Path(__file__).resolve().parent.parent / "verify" /
                    "debug_pin_intermediate.py")
    cmd2 = [
        sys.executable, str(debug_script),
        seq_hex, str(args.lt),
        "--work-dir", ".",
        "--funding-txid", fund_txid,
        "--funding-vout", str(fund_vout),
    ]
    r = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
    cpu_lines = [ln for ln in (r.stdout or "").splitlines() if ln.startswith("DBG:")]
    print(f"  → {len(cpu_lines)} DBG lines from CPU")
    if not cpu_lines:
        print(f"  ✗ CPU dump failed. stdout: {r.stdout}\n  stderr: {r.stderr}")
        return

    # Build name → value maps and compare
    def to_map(lines):
        m = {}
        for ln in lines:
            if "=" in ln:
                k, _, v = ln[len("DBG:"):].partition("=")
                m[k.strip()] = v.strip()
        return m

    gpu_m = to_map(gpu_lines)
    cpu_m = to_map(cpu_lines)
    keys = []
    seen = set()
    # Order: union, GPU order first, then any CPU-only
    for ln in gpu_lines:
        if "=" in ln:
            k = ln[len("DBG:"):].partition("=")[0].strip()
            if k not in seen:
                keys.append(k); seen.add(k)
    for ln in cpu_lines:
        if "=" in ln:
            k = ln[len("DBG:"):].partition("=")[0].strip()
            if k not in seen:
                keys.append(k); seen.add(k)

    print(f"\n─── Side-by-side comparison ({len(keys)} fields) ───\n")
    width = max(len(k) for k in keys)
    first_div = None
    for k in keys:
        gv = gpu_m.get(k, "<missing>")
        cv = cpu_m.get(k, "<missing>")
        match = (gv == cv)
        flag = "  " if match else "✗ "
        if not match and first_div is None:
            first_div = k
        print(f"  {flag}{k:<{width}}")
        print(f"      GPU: {gv}")
        print(f"      CPU: {cv}")
    print()
    if first_div:
        print(f"  ⚠ FIRST DIVERGENCE: {first_div}")
        print(f"    Bug is in or before computation of: {first_div}")
    else:
        print(f"  ✓ All fields match. The bug is NOT in the dumped pipeline.")
        print(f"    (Maybe in the kernel's path that's not exercised here.)")


def cmd_diag(args):
    """Pull diagnostic info from every machine in the current fleet.

    For each machine, downloads:
      /root/qsb_run/qsb_real.log
      /root/qsb_run/qsb_digest.log
      /root/qsb_run/bootstrap.ok / bootstrap.err
      /root/qsb_run/pin.log / pin_hit.json
      /root/qsb_run/digest_r1.log / digest_r1_hit.json
      /root/qsb_run/digest_r2.log / digest_r2_hit.json
      /root/qsb_run/results/*  (per-GPU search logs, if present)
      output of: nvcc --version, nvidia-smi, ls /root/qsb_run/

    Saves all of it to fleet_diag/inst_<id>/ locally.
    """
    machines = load_fleet()
    if not machines:
        print("No fleet on record.")
        return

    diag_root = Path("fleet_diag")
    diag_root.mkdir(exist_ok=True)
    print(f"Pulling diagnostics from {len(machines)} machines into {diag_root}/...")

    import concurrent.futures
    import threading
    plock = threading.Lock()

    def diag_one(m: 'Machine'):
        d = diag_root / f"inst_{m.instance_id}"
        d.mkdir(parents=True, exist_ok=True)

        # Refresh ssh_host/port if needed
        if not m.ssh_host or not m.ssh_port:
            try:
                info = show_instance(m.instance_id)
                m.ssh_host = info.get("ssh_host") or m.ssh_host
                m.ssh_port = int(info.get("ssh_port", 0)) or m.ssh_port
            except Exception:
                pass
        if not m.ssh_host:
            with plock:
                print(f"  [inst {m.instance_id}] no SSH info; skipping")
            return

        # System info
        rc, out, err = ssh_exec(
            m,
            "echo === nvcc; nvcc --version 2>/dev/null; "
            "echo === nvidia-smi; nvidia-smi 2>/dev/null | head -20; "
            "echo === ls qsb_run; ls -la /root/qsb_run/ 2>/dev/null; "
            "echo === ls results; ls -la /root/qsb_run/results/ 2>/dev/null; "
            "echo === ps; ps auxw | grep -E 'qsb_real|qsb_digest' | grep -v grep; "
            "echo === gpu utilization; nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory --format=csv 2>/dev/null",
            timeout=60, retries=2,
        )
        (d / "system_info.txt").write_text(f"=== rc={rc} ===\n{out}\n=== STDERR ===\n{err}\n")

        # Files to pull
        for remote, local in [
            ("/root/qsb_run/qsb_real.log",      "qsb_real.log"),
            ("/root/qsb_run/qsb_digest.log",    "qsb_digest.log"),
            ("/root/qsb_run/bootstrap.ok",      "bootstrap.ok"),
            ("/root/qsb_run/bootstrap.err",     "bootstrap.err"),
            ("/root/qsb_run/pin.log",           "pin.log"),
            ("/root/qsb_run/pin_hit.json",      "pin_hit.json"),
            ("/root/qsb_run/pin_status",        "pin_status"),
            ("/root/qsb_run/digest_r1.log",     "digest_r1.log"),
            ("/root/qsb_run/digest_r1_hit.json","digest_r1_hit.json"),
            ("/root/qsb_run/digest_r1_status",  "digest_r1_status"),
            ("/root/qsb_run/digest_r2.log",     "digest_r2.log"),
            ("/root/qsb_run/digest_r2_hit.json","digest_r2_hit.json"),
            ("/root/qsb_run/digest_r2_status",  "digest_r2_status"),
        ]:
            scp_download_file(m, remote, d / local, retries=1)
        # Pull results dir as a whole
        scp_download(m, "/root/qsb_run/results", d, retries=1)

        with plock:
            files = list(d.iterdir())
            sizes = {p.name: p.stat().st_size for p in files if p.is_file()}
            interesting = []
            if sizes.get("qsb_real.log", 0) > 0:
                interesting.append(f"qsb_real.log={sizes['qsb_real.log']}b")
            if sizes.get("qsb_digest.log", 0) > 0:
                interesting.append(f"qsb_digest.log={sizes['qsb_digest.log']}b")
            if sizes.get("bootstrap.err", 0) > 0:
                interesting.append("BOOTSTRAP.ERR PRESENT")
            if (d / "results").exists():
                rfiles = list((d / "results").iterdir())
                if rfiles:
                    interesting.append(f"results/={len(rfiles)} files")
            print(f"  [inst {m.instance_id}] {' '.join(interesting) or '(empty)'}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(machines), 16)) as ex:
        list(ex.map(diag_one, machines))

    print(f"\nDone. Inspect fleet_diag/inst_<id>/ for per-machine logs.")
    print(f"Quick survey:")
    for m in machines:
        d = diag_root / f"inst_{m.instance_id}"
        bootstrap_ok = (d / "bootstrap.ok").exists() and (d / "bootstrap.ok").stat().st_size >= 0
        bootstrap_err = (d / "bootstrap.err").exists() and (d / "bootstrap.err").stat().st_size > 0
        real_log = (d / "qsb_real.log").exists() and (d / "qsb_real.log").stat().st_size > 0
        marker = ("✓ ok" if bootstrap_ok and not bootstrap_err else
                  ("✗ err" if bootstrap_err else "?"))
        print(f"  inst {m.instance_id}: {marker}  (real_log={real_log})")


def cmd_search(args):
    """End-to-end coordinated search:
       1. Provision fleet (or use existing if --reuse)
       2. Start pinning on all machines
       3. Wait for ANY pin hit
       4. Stop pinning everywhere, start digest R1 with the pin's (seq, lt)
       5. Wait for ANY R1 hit
       6. Start digest R2 with the same (seq, lt)
       7. Wait for ANY R2 hit
       8. Pull all hit JSON files locally and write a consolidated gpu_hits.json
    """
    work_dir = Path.cwd()
    gpu_dir = Path(args.gpu_dir).resolve()
    check_prereqs(work_dir)
    if not gpu_dir.exists():
        print(f"ERROR: --gpu-dir {gpu_dir} not found")
        sys.exit(1)

    # Either reuse existing fleet or provision new one
    if args.reuse:
        machines = load_fleet()
        if not machines:
            print("ERROR: --reuse but no fleet on record")
            sys.exit(1)
        print(f"Reusing existing fleet of {len(machines)} machines")
    else:
        # Search + create instances
        print("─── searching for offers ───")
        try:
            offers = search_offers(
                gpu_name=args.gpu, max_dph=args.max_dph,
                min_reliability=args.min_reliability,
                datacenter_only=args.datacenter_only,
                min_gpus=args.min_gpus,
                prefer_multi_gpu=getattr(args, 'prefer_multi_gpu', False),
            )
        except Exception as e:
            print(f"vastai search failed: {e}")
            sys.exit(1)
        if not offers:
            print("No offers matched. Loosen --max-dph or --min-gpus.")
            sys.exit(1)

        # Pick offers — two modes:
        #   --target-gpus N (preferred): greedy by $/GPU until total GPUs >= N
        #                                 (with optional over-provisioning %)
        #   --count N (legacy):          pick top N machines by $/h
        if args.target_gpus is not None:
            # Sort by per-GPU price ascending (best first)
            ranked = sorted(offers,
                            key=lambda o: o.get('dph_total', 0) /
                                          max(o.get('num_gpus', 1), 1))
            target = int(args.target_gpus *
                         (1.0 + args.over_provision_pct / 100.0))
            chosen = []
            total = 0
            for o in ranked:
                if total >= target:
                    break
                chosen.append(o)
                total += o.get('num_gpus', 1)
            if total < args.target_gpus:
                print(f"WARN: only found {total} GPUs (asked {args.target_gpus}). "
                      f"Loosen --max-dph or --min-gpus to find more.")
        else:
            chosen = offers[:args.count]

        total_gpus = sum(o.get('num_gpus', 1) for o in chosen)
        total_dph = sum(o.get('dph_total', 0.0) for o in chosen)
        avg_dph_per_gpu = total_dph / max(total_gpus, 1)

        print(f"\nFleet: {total_gpus} GPUs across {len(chosen)} machines, "
              f"${total_dph:.2f}/h total (${avg_dph_per_gpu:.3f}/GPU/h)")
        for o in chosen:
            ngpu = o.get('num_gpus', 1)
            per_gpu = o.get('dph_total', 0) / max(ngpu, 1)
            print(f"  offer {o['id']}: {o.get('gpu_name')} × {ngpu:>2}, "
                  f"${o.get('dph_total'):.3f}/h (${per_gpu:.3f}/GPU/h), "
                  f"rel={o.get('reliability2', 0):.3f}, "
                  f"loc={o.get('geolocation', '?')}")
        if not args.yes:
            resp = input(f"\nRent {len(chosen)} instances ({total_gpus} GPUs)? [y/N] ")
            if resp.strip().lower() != "y":
                print("Aborted.")
                return

        print("\n─── building bundle ───")
        bundle = build_bundle(work_dir, gpu_dir)
        print(f"  bundle: {bundle.stat().st_size / 1024:.0f} KB")

        # Phase A: create instances (sequential, fast)
        print("\n─── creating instances ───")
        machines = []
        cum_gpu_offset = 0
        for i, offer in enumerate(chosen):
            ngpu = offer.get('num_gpus', 1)
            print(f"  [{i+1}/{len(chosen)}] renting offer {offer['id']} "
                  f"({ngpu} × {offer.get('gpu_name')})...", end='', flush=True)
            try:
                iid = create_instance(offer["id"], image=args.image, disk=args.disk,
                                       label=f"qsb-{i}")
            except Exception as e:
                print(f" ✗ {e}")
                continue
            m = Machine(
                instance_id=iid, offer_id=offer["id"],
                gpu_name=offer.get("gpu_name", "?"),
                num_gpus=ngpu,
                dph=offer.get("dph_total", 0.0),
                # seq_hex/lt_start/lt_range no longer used — pin kernel iterates internally
                # We still set them so that the dataclass is valid; values don't matter
                seq_hex="0x80000000", lt_start=0, lt_range=0,
                global_gpu_offset=cum_gpu_offset,
            )
            cum_gpu_offset += ngpu
            machines.append(m)
            save_fleet(machines)
            print(f" ✓ instance {iid} (gpu_offset={m.global_gpu_offset})")

        if not machines:
            print("\nNo instances created. Aborting.")
            return

        total_gpus_in_fleet = sum(m.num_gpus for m in machines)

        # ──────────────────────────────────────────────────────────────────
        # PIPELINE MODE (only when --use-pin-locktime; full pin+digest flow
        # still uses the sequential path below).
        # 
        # Each machine runs its own thread: bootstrap → upload tiles → start
        # digest. The main thread polls for hits as machines come online.
        # Slow/dead machines do NOT block fast ones.
        # ──────────────────────────────────────────────────────────────────
        # PIPELINE MODE only supports R1→R2 chain. When user passes
        # --start-round 2, we want to skip R1 entirely and go straight to R2,
        # which is handled correctly by the sequential code path below.
        # Otherwise, pipeline mode would still launch R1 because its hit
        # detection / round transitions are R1-centric.
        if args.use_pin_locktime is not None and args.start_round >= 2 and not args.no_pipeline:
            print(f"\n  --start-round={args.start_round} is incompatible with pipeline mode.")
            print(f"  Forcing --no-pipeline (sequential mode) for R{args.start_round}-only search.")
            args.no_pipeline = True

        if args.use_pin_locktime is not None and not args.no_pipeline:
            print(f"\n─── PIPELINE MODE: per-machine bootstrap+digest in parallel ───")
            print(f"    (faster wall time, but accepts coverage gaps from failed machines)")
            print(f"    (use --no-pipeline to force sequential mode that re-partitions)")
            seq_hex = args.use_pin_sequence
            locktime = args.use_pin_locktime

            # ── PIN VERIFICATION GATE ──
            # Run verify_hit.py on CPU to confirm the supplied pin actually
            # produces a valid DER on the current state. Without this, a stale
            # or wrong pin sends the entire fleet searching impossible R1.
            if not args.skip_pin_verify:
                fund_txid, fund_vout = _resolve_funding_info(args)
                if not fund_txid:
                    print("  ✗ Cannot verify pin: no --funding-txid supplied "
                          "and no regtest_funding.json found in cwd.")
                    print("    Pass --funding-txid <hex> or --skip-pin-verify "
                          "(NOT recommended).")
                    return
                print(f"\n─── Verifying supplied pin on CPU ───")
                print(f"  seq={seq_hex} lt={locktime} funding={fund_txid[:16]}...")
                ok, output = _verify_pin_on_cpu(
                    seq_hex, locktime, fund_txid, fund_vout, work_dir='.')
                if not ok:
                    print("\n  ✗ PIN VERIFICATION FAILED — pin does NOT reproduce on CPU.")
                    print("    The fleet would search an impossible R1 space.")
                    print("    Aborting. Last 10 lines of verify_hit.py output:")
                    for line in (output or '').rstrip().split('\n')[-10:]:
                        print(f"    {line}")
                    print("\n  Likely causes: stale pin from a different state, "
                          "wrong (seq, lt), or a buggy kernel binary.")
                    print(f"  To bypass (NOT recommended): pass --skip-pin-verify")
                    return
                print(f"  ✓ Pin verified on CPU — proceeding to digest.")
            else:
                print(f"\n  ⚠ --skip-pin-verify set; CPU pin verification BYPASSED.")
            
            # Pre-compute tile partitions for the ORIGINAL fleet GPU count.
            # If some machines fail, their tiles go un-searched (gap), which is
            # fine for find-one-hit semantics.
            print(f"  computing LPT tile partitions for {total_gpus_in_fleet} GPUs...")
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pipeline'))
            from tile_partition import partition_for_fleet, write_tile_file
            
            state_path = Path('qsb_state.json')
            if not state_path.exists():
                print("  ✗ qsb_state.json not found in cwd — aborting")
                return
            with open(state_path) as f:
                state = json.load(f)
            n_pool = state['n']
            t_sel_r1 = state['t1s'] + state['t1b']
            t_sel_r2 = state['t2s'] + state['t2b']
            
            tiles_dir = Path('digest_tiles')
            tiles_dir.mkdir(exist_ok=True)
            
            for round_num, t_sel in [(1, t_sel_r1), (2, t_sel_r2)]:
                assignment, stats = partition_for_fleet(n_pool, t_sel, total_gpus_in_fleet)
                print(f"  round {round_num}: C({n_pool},{t_sel}) → "
                      f"{stats['num_tiles']} tiles, "
                      f"imbalance {stats['imbalance_ratio']:.4f}x")
                for gpu_id, tlist in assignment.items():
                    p = tiles_dir / f"digest_r{round_num}_tiles_gpu_{gpu_id}.bin"
                    write_tile_file(str(p), tlist)
            
            print(f"  ✓ tile files generated in {tiles_dir}/")
            print(f"  starting per-machine pipeline ({len(machines)} threads)...")
            
            import concurrent.futures
            import threading
            plock = threading.Lock()
            
            # Shared state across threads:
            ready_machines: list[Machine] = []      # successfully started digest
            ready_lock = threading.Lock()
            
            def pipeline_one_machine(m: 'Machine') -> tuple[str, 'Machine']:
                """Bootstrap, upload tiles, start digest — all on a single machine."""
                def log(msg):
                    with plock:
                        print(f"  [inst {m.instance_id}] {msg}", flush=True)
                
                # Step 1: bootstrap (SSH wait + bundle upload + compile)
                status, _ = _bootstrap_compile(m, bundle, plock)
                if status != 'ok':
                    log(f"✗ bootstrap failed: {status}")
                    return (status, m)
                
                # Step 2: upload tile files for both rounds (this machine's GPUs only)
                log(f"uploading tile files for both rounds...")
                for round_num in (1, 2):
                    for g in range(m.num_gpus):
                        gid = m.global_gpu_offset + g
                        src = tiles_dir / f"digest_r{round_num}_tiles_gpu_{gid}.bin"
                        if not src.exists():
                            log(f"✗ missing local tile file {src}")
                            return ('tile_missing', m)
                        if not scp_upload(m, src,
                                f"/root/qsb_run/digest_r{round_num}_tiles_gpu_{gid}.bin",
                                retries=3):
                            log(f"✗ tile upload failed: {src.name}")
                            return ('tile_upload_failed', m)
                log(f"✓ tile files uploaded")
                
                # Step 3: start digest R1 immediately
                log(f"starting digest R{args.start_round}...")
                ok = _start_digest(m, args.start_round, seq_hex, locktime, total_gpus_in_fleet)
                if not ok:
                    log(f"✗ digest R{args.start_round} start failed")
                    return ('digest_start_failed', m)
                log(f"✓ digest R{args.start_round} running")
                with ready_lock:
                    ready_machines.append(m)
                return ('ok', m)
            
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(len(machines), 32)) as ex:
                futures = {ex.submit(pipeline_one_machine, m): m for m in machines}
                
                # Main thread: poll for first hit while machines come online
                pipeline_results = {}
                t0 = time.time()
                last_status_time = t0
                r1_hit = None
                r1_machine = None
                
                # Process completions and poll in interleaved fashion
                while futures:
                    done, _pending = concurrent.futures.wait(
                        list(futures.keys()), timeout=15,
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in done:
                        try:
                            status, m = fut.result()
                        except Exception as e:
                            m = futures[fut]
                            status = f'exception: {e}'
                        pipeline_results[m.instance_id] = status
                        del futures[fut]
                    
                    # Periodic status print
                    elapsed = time.time() - t0
                    if elapsed - (last_status_time - t0) >= 30:
                        with ready_lock:
                            ready_count = len(ready_machines)
                        with plock:
                            print(f"  [{int(elapsed)}s] {ready_count}/{len(machines)} "
                                  f"machines searching, "
                                  f"{len(pipeline_results)} done bootstrapping",
                                  flush=True)
                        last_status_time = time.time()
                    
                    # Check for hit on any ready machine
                    with ready_lock:
                        snapshot = list(ready_machines)
                    for m in snapshot:
                        hit = _check_digest_hit(m, args.start_round)
                        if hit is not None:
                            r1_hit = hit
                            r1_machine = m
                            break
                    if r1_hit:
                        with plock:
                            print(f"\n  ✓ R1 HIT from inst {r1_machine.instance_id}",
                                  flush=True)
                        break
                
                # Cancel remaining futures (they'll continue but we don't care)
                for fut in futures:
                    fut.cancel()
            
            save_fleet(machines)
            
            # Report bootstrap status
            ok_machines = [m for m in machines
                          if pipeline_results.get(m.instance_id) == 'ok']
            failed_iids = [iid for iid, s in pipeline_results.items() if s != 'ok']
            print(f"\n  Pipeline result: {len(ok_machines)}/{len(machines)} machines started search")
            if failed_iids:
                print(f"  Failed (still billed!): {failed_iids}")
                print(f"  Diagnostics: fleet_diag/inst_<id>/")
            
            # If no hit yet, keep polling
            if r1_hit is None:
                if not ok_machines:
                    print("\n  ✗ ZERO machines started search successfully.")
                    print("  Run: python3 ... destroy -y")
                    return
                with ready_lock:
                    snapshot = list(ready_machines)
                print(f"\n  All bootstraps done; polling {len(snapshot)} active machines for R1 hit...")
                r1_hit, r1_machine = _poll_for_first_hit(
                    snapshot, lambda m: _check_digest_hit(m, 1),
                    "digest_r1_status",
                    poll_interval=args.poll_interval, timeout=args.digest_timeout,
                )
            
            if r1_hit is None:
                print("\n  ✗ NO R1 HIT — all active machines exhausted. Likely kernel bug.")
                return
            print(f"  R1 hit details:\n{json.dumps(r1_hit, indent=2)}")

            # CPU-verify the R1 hit before committing the fleet to R2.
            # Mirror of the pin-verify gate. If kernel reports a bogus hit
            # (e.g., from a stale tx_suffix layout assumption), R2 will
            # search against a phantom and find nothing.
            if not getattr(args, 'skip_digest_verify', False):
                r1_indices = r1_hit.get('combo') or r1_hit.get('indices') or ''
                r1_sighash = r1_hit.get('sighash')
                r1_pubhash = r1_hit.get('pubhash')
                # Fallback: pull kernel summary if JSON lacks sighash
                if not r1_sighash:
                    print(f"\n  hit JSON lacks sighash; pulling kernel summary from inst {r1_machine.instance_id}...")
                    details = _fetch_digest_hit_details(r1_machine, round_num=1)
                    if details:
                        r1_sighash = details.get('sighash')
                        r1_pubhash = details.get('pubhash')
                        if not r1_indices and details.get('indices'):
                            r1_indices = details['indices']
                        print(f"  ✓ recovered: sighash={r1_sighash[:16] if r1_sighash else 'NONE'}...")
                ok = _verify_digest_on_cpu(round_num=1, indices_csv=r1_indices,
                                            seq_hex=seq_hex, locktime=locktime,
                                            gpu_sighash=r1_sighash,
                                            gpu_pubhash=r1_pubhash)
                if not ok:
                    print("\n  ✗ R1 VERIFICATION FAILED — hit does NOT reproduce on CPU.")
                    print("    Refusing to start R2 on a phantom hit.")
                    print("    (Use --skip-digest-verify to override, NOT recommended.)")
                    # Stop still-running R1 kernels before returning
                    with ready_lock:
                        snapshot_for_kill = list(ready_machines)
                    _kill_kernels_on_fleet(snapshot_for_kill, label="R1 (post-failed-verify)")
                    return

            # Stop R1 everywhere
            print(f"\n  stopping R1 on all machines...")
            with ready_lock:
                snapshot = list(ready_machines)
            for m in snapshot:
                ssh_exec(m, "pkill -x qsb_digest 2>/dev/null || true", timeout=15)
            
            # Start R2 on all ready machines (parallel)
            print(f"\n═══ DIGEST R2 STAGE — starting on {len(snapshot)} machines ═══")
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(len(snapshot), 32)) as ex:
                r2_futures = {
                    ex.submit(_start_digest, m, 2, seq_hex, locktime, total_gpus_in_fleet): m
                    for m in snapshot
                }
                r2_started = []
                for fut in concurrent.futures.as_completed(r2_futures):
                    m = r2_futures[fut]
                    try:
                        if fut.result():
                            r2_started.append(m)
                    except Exception as e:
                        print(f"  WARN: R2 start failed on {m.instance_id}: {e}")
            print(f"  R2 running on {len(r2_started)}/{len(snapshot)} machines")
            
            r2_hit, r2_machine = _poll_for_first_hit(
                r2_started, lambda m: _check_digest_hit(m, 2),
                "digest_r2_status",
                poll_interval=args.poll_interval, timeout=args.digest_timeout,
            )
            if r2_hit is None:
                print("\n  ✗ NO R2 HIT.")
                return
            print(f"\n  ✓ R2 HIT from inst {r2_machine.instance_id}")
            print(json.dumps(r2_hit, indent=2))

            # CPU-verify R2 too — catches the same class of kernel bug.
            if not getattr(args, 'skip_digest_verify', False):
                r2_indices = r2_hit.get('combo') or r2_hit.get('indices') or ''
                r2_sighash = r2_hit.get('sighash')
                r2_pubhash = r2_hit.get('pubhash')
                if not r2_sighash:
                    print(f"\n  hit JSON lacks sighash; pulling kernel summary from inst {r2_machine.instance_id}...")
                    details = _fetch_digest_hit_details(r2_machine, round_num=2)
                    if details:
                        r2_sighash = details.get('sighash')
                        r2_pubhash = details.get('pubhash')
                        if not r2_indices and details.get('indices'):
                            r2_indices = details['indices']
                ok = _verify_digest_on_cpu(round_num=2, indices_csv=r2_indices,
                                            seq_hex=seq_hex, locktime=locktime,
                                            gpu_sighash=r2_sighash,
                                            gpu_pubhash=r2_pubhash)
                if not ok:
                    print("\n  ✗ R2 VERIFICATION FAILED — hit does NOT reproduce on CPU.")
                    print("    Saved hits anyway for manual inspection. DO NOT")
                    print("    assemble a tx with these indices — it will fail on broadcast.")
                    # Still save the hit files so user can inspect
                else:
                    print("    Proceeding to consolidate hits.")
            
            # Save the consolidated hits
            print(f"\n═══ SEARCH COMPLETE — pulling hit logs ═══")
            for m in snapshot:
                d = Path('hits') / f"instance_{m.instance_id}"
                d.mkdir(parents=True, exist_ok=True)
                for fname in ['digest_r1_hit.json', 'digest_r2_hit.json',
                              'pin_hit.json']:
                    scp_download_file(m, f"/root/qsb_run/{fname}", d / fname,
                                     retries=2)
            print(f"  ✓ hits saved to hits/")
            print(f"\nDon't forget to: python3 ... destroy -y")
            return

        # ──────────────────────────────────────────────────────────────────
        # SEQUENTIAL MODE (full pin → digest pivot, used when no pre-known pin hit)
        # ──────────────────────────────────────────────────────────────────
        # Phase B: bootstrap+compile in parallel
        print(f"\n─── compiling kernels on {len(machines)} machines (parallel) ───")
        import concurrent.futures
        import threading
        plock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(machines), 16)) as ex:
            futures = {ex.submit(_bootstrap_compile, m, bundle, plock): m
                       for m in machines}
            results = {}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    status, m = fut.result()
                except Exception as e:
                    m = futures[fut]
                    status = f'exception: {e}'
                results[m.instance_id] = status
        save_fleet(machines)

        ok_machines = [m for m in machines if results.get(m.instance_id) == 'ok']
        print(f"\n  Compiled: {len(ok_machines)}/{len(machines)} machines")
        if len(ok_machines) < len(machines):
            failed = [iid for iid, s in results.items() if s != 'ok']
            print(f"  Failed (still billed!): {failed}")
            print(f"  Diagnostics for each failed machine saved under fleet_diag/inst_<id>/")
        if not ok_machines:
            print("\n  ✗ ZERO machines compiled successfully.")
            print("  All machines are still rented. Inspect fleet_diag/ for actual errors.")
            print("  When ready, run: python3 ... destroy -y")
            return
        if len(ok_machines) < len(machines) and not args.yes:
            resp = input("Continue with healthy machines only? [y/N] ")
            if resp.strip().lower() != "y":
                print("Aborted. Use `destroy` to stop billing.")
                return
        # Recompute total_gpus from healthy machines only — partitioning must be tight
        machines = ok_machines
        # Re-pack global_gpu_offset to keep it contiguous
        cum = 0
        for m in machines:
            m.global_gpu_offset = cum
            cum += m.num_gpus
        save_fleet(machines)

    total_gpus = sum(m.num_gpus for m in machines)
    if total_gpus == 0:
        print("\n  ✗ No usable GPUs in fleet. Aborting before search.")
        return
    print(f"\nUsing {total_gpus} GPUs across {len(machines)} machines for search.")

    # ── Pin stage (skipped if --use-pin-locktime supplied) ──
    if args.use_pin_locktime is not None:
        # User has a pin hit from a previous run; skip the pin search entirely
        seq_hex = args.use_pin_sequence
        if not seq_hex.startswith('0x'):
            seq_hex = '0x' + seq_hex if all(c in '0123456789abcdefABCDEF' for c in seq_hex) else hex(int(seq_hex))
        locktime = args.use_pin_locktime
        pin_hit = {
            'sequence': int(seq_hex, 16), 'sequence_hex': seq_hex,
            'locktime': locktime,
            'hash_choice': args.use_pin_hash_choice,
            'recid': args.use_pin_recid,
            'machine_offset': -1, '_supplied': True,
        }
        pin_machine = None  # not from this run
        print(f"\n═══ PIN STAGE — SKIPPED (using supplied hit) ═══")
        print(f"  sequence:    {seq_hex}")
        print(f"  locktime:    {locktime}")
        print(f"  hash_choice: {args.use_pin_hash_choice}")
        print(f"  recid:       {args.use_pin_recid}")
    else:
        print(f"\n═══ PIN STAGE — searching for any (seq, lt) pinning hit ═══")
        if args.pin_seq_start:
            print(f"  pin seq_start override: {args.pin_seq_start}")
        print(f"  starting pinning on all {len(machines)} machines...")
        # Compose extra flags including optional seq_start
        pin_flags = "single_hash"
        if args.pin_seq_start:
            pin_flags = f"single_hash seq_start={args.pin_seq_start}"
        for m in machines:
            ok = _start_pin(m, total_gpus, extra_flags=pin_flags)
            if not ok:
                print(f"  WARN: failed to start pin on inst {m.instance_id}")
        print(f"  polling every {args.poll_interval}s for first pin hit...")
        pin_hit, pin_machine = _poll_for_first_hit(
            machines, _check_pin_hit, "pin_status",
            poll_interval=args.poll_interval, timeout=args.pin_timeout,
        )
        if pin_hit is None:
            print("\n  ✗ NO PIN HIT FOUND — fleet exhausted its iteration without a hit.")
            print("    Either bad luck (very unlikely with 2^32 seq space) or kernel bug.")
            print("    Check /root/qsb_run/pin.log on each machine for details.")
            return

        print(f"\n  ✓ PIN HIT from inst {pin_machine.instance_id}:")
        print(json.dumps(pin_hit, indent=2))
        seq_hex = pin_hit.get('sequence_hex') or hex(pin_hit['sequence'])
        locktime = pin_hit['locktime']

        # Stop pinning on all machines (some may still be searching)
        print(f"\n  stopping pin on all machines...")
        for m in machines:
            ssh_exec(m, "pkill -x qsb_real 2>/dev/null || true", timeout=15)

    # ── PIN VERIFICATION GATE (applies to both supplied and freshly-found pins) ──
    # Without this gate, a buggy or stale pin sends the fleet into impossible R1
    # space — exactly what burned hours of fleet time on April 28-29.
    if not args.skip_pin_verify:
        fund_txid, fund_vout = _resolve_funding_info(args)
        if not fund_txid:
            print("\n  ✗ Cannot verify pin: no --funding-txid supplied and no "
                  "regtest_funding.json found in cwd.")
            print("    Pass --funding-txid <hex> or --skip-pin-verify (NOT recommended).")
            return

        # Fetch the GPU's full hit file (includes diagnostic pubkey/sha/sighash
        # the kernel claimed). With this we can localize WHERE the GPU diverges
        # from CPU if validation fails.
        local_hit_file = None
        if pin_machine is not None:
            try:
                hit_dir = Path('hits') / f"instance_{pin_machine.instance_id}"
                hit_dir.mkdir(parents=True, exist_ok=True)
                # List the remote results dir to find the actual hit file
                # (we don't know which local GPU index produced it).
                rc, out, err = ssh_exec(
                    pin_machine,
                    "ls /root/qsb_run/results/pinning_hit_*.txt 2>/dev/null | head -5",
                    timeout=15,
                )
                hit_files = [ln.strip() for ln in (out or '').splitlines() if ln.strip()]
                if hit_files:
                    print(f"  found {len(hit_files)} hit file(s) on remote: "
                          f"{[Path(f).name for f in hit_files]}")
                    for remote in hit_files:
                        local = hit_dir / Path(remote).name
                        if scp_download_file(pin_machine, remote, local, retries=2):
                            local_hit_file = str(local)
                            print(f"  pulled GPU hit file: {local_hit_file}")
                            break
                else:
                    print(f"  WARN: no pinning_hit_*.txt found on remote — "
                          f"diagnostic data unavailable")
            except Exception as e:
                print(f"  (could not fetch GPU hit file: {e})")

        print(f"\n─── Verifying pin on CPU before R1 ───")
        print(f"  seq={seq_hex} lt={locktime} funding={fund_txid[:16]}...")
        ok, output = _verify_pin_on_cpu(
            seq_hex, locktime, fund_txid, fund_vout, work_dir='.',
            gpu_hit_file=local_hit_file)
        if not ok:
            print("\n  ✗ PIN VERIFICATION FAILED — pin does NOT reproduce on CPU.")
            print("    The fleet would search an impossible R1 space. Aborting.")
            print("    Full verify_hit.py output (with GPU↔CPU diff if available):")
            for line in (output or '').rstrip().split('\n'):
                print(f"    {line}")
            print("\n  Read the comparison above to localize the bug:")
            print("    - If GPU sighash differs from CPU: bug in suffix patching or SHA-256.")
            print("    - If GPU pubkey differs but is one of CPU's: bug in DER check.")
            print("    - If GPU pubkey doesn't match any CPU pubkey: bug in EC math.")
            print(f"\n  To bypass (NOT recommended): pass --skip-pin-verify")
            return
        print(f"  ✓ Pin verified on CPU — proceeding to digest.")
    else:
        print(f"\n  ⚠ --skip-pin-verify set; CPU pin verification BYPASSED.")

    # ── Digest stage (R1 unless --start-round 2 → skip to R2) ──
    if args.start_round >= 2:
        print(f"\n═══ DIGEST R2 STAGE — searching at (seq={seq_hex}, lt={locktime}) ═══")
        print(f"  (skipping R1 — --start-round=2; assumes R1 already solved with this pin)")
    else:
        print(f"\n═══ DIGEST R1 STAGE — searching at (seq={seq_hex}, lt={locktime}) ═══")

    # Generate balanced tile partition for the fleet
    print(f"  computing LPT tile partition for {total_gpus} GPUs...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pipeline'))
    try:
        from tile_partition import partition_for_fleet, write_tile_file  # type: ignore
    except ImportError:
        print("  WARN: tile_partition module not found; falling back to mod-N")
        partition_for_fleet = None

    # Determine n_pool, t_sel from state
    state_path = Path('qsb_state.json')
    if not state_path.exists():
        print(f"  WARN: qsb_state.json not found in cwd, can't compute tiles; using mod-N")
        partition_for_fleet = None

    tiles_uploaded = False
    if partition_for_fleet is not None:
        with open(state_path) as f:
            state = json.load(f)
        n_pool = state['n']
        # round 1 has t1s + t1b indices
        t_sel_r1 = state['t1s'] + state['t1b']
        t_sel_r2 = state['t2s'] + state['t2b']
        # Generate tiles for each round, write per-GPU files locally, upload to each machine
        tiles_dir = Path('digest_tiles')
        tiles_dir.mkdir(exist_ok=True)

        def gen_and_upload(round_num, t_sel):
            nonlocal tiles_uploaded
            print(f"  round {round_num}: partitioning C({n_pool},{t_sel}) across {total_gpus} GPUs...")
            assignment, stats = partition_for_fleet(n_pool, t_sel, total_gpus)
            print(f"    → {stats['num_tiles']} tiles, imbalance "
                  f"{stats['imbalance_ratio']:.4f}x, "
                  f"covered={stats['covered']}")
            if not stats['covered']:
                print(f"    ✗ partition does NOT fully cover the search space")
                return False
            # Write per-GPU tile files
            for gpu_id, tlist in assignment.items():
                p = tiles_dir / f"digest_r{round_num}_tiles_gpu_{gpu_id}.bin"
                write_tile_file(str(p), tlist)
            # Upload to each machine: each machine takes its local GPU's tile files
            print(f"    uploading tile files to {len(machines)} machines (parallel)...")
            import concurrent.futures
            def upload_to_machine(m: 'Machine'):
                # This machine's GPUs are global ids [m.global_gpu_offset, m.global_gpu_offset + m.num_gpus)
                ok = True
                for g in range(m.num_gpus):
                    gid = m.global_gpu_offset + g
                    src = tiles_dir / f"digest_r{round_num}_tiles_gpu_{gid}.bin"
                    if not scp_upload(m, src, f"/root/qsb_run/digest_r{round_num}_tiles_gpu_{gid}.bin"):
                        ok = False
                return ok
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(len(machines), 16)) as ex:
                results = list(ex.map(upload_to_machine, machines))
            success_count = sum(results)
            print(f"    ✓ uploaded to {success_count}/{len(machines)} machines")
            return success_count > 0

        # Pre-generate + upload BOTH rounds' tile files in parallel before starting R1
        # (doing both up front means R2 doesn't add latency between rounds).
        # If --start-round 2, skip R1 tiles entirely (they're never used).
        if args.start_round >= 2:
            print(f"  (skipping R1 tile generation — --start-round=2)")
            if gen_and_upload(2, t_sel_r2):
                tiles_uploaded = True
        else:
            if gen_and_upload(1, t_sel_r1):
                tiles_uploaded = True
            gen_and_upload(2, t_sel_r2)

    if tiles_uploaded:
        print(f"  ✓ tile files uploaded; kernel will use balanced LPT partitioning")
    else:
        print(f"  ⚠ falling back to mod-N partitioning (less balanced)")

    if args.start_round >= 2:
        print(f"\n═══ DIGEST R1 STAGE — SKIPPED (--start-round=2) ═══")
        print(f"  Assuming R1 was already solved with these (seq, lt). Going straight to R2.")
    else:
        print(f"  starting digest r1 on all {len(machines)} machines (full-fleet partitioning)...")
        for m in machines:
            ok = _start_digest(m, 1, seq_hex, locktime, total_gpus)
            if not ok:
                print(f"  WARN: failed to start digest r1 on inst {m.instance_id}")

        r1_hit, r1_machine = _poll_for_first_hit(
            machines, lambda m: _check_digest_hit(m, 1), "digest_r1_status",
            poll_interval=args.poll_interval, timeout=args.digest_timeout,
        )
        if r1_hit is None:
            print("\n  ✗ NO R1 HIT — fleet exhausted C(150,9) without a hit. Bug.")
            return
        print(f"\n  ✓ R1 HIT from inst {r1_machine.instance_id}:")
        print(json.dumps(r1_hit, indent=2))

        # CPU-verify the R1 hit before committing the fleet to R2.
        # Mirror of the pin-verify gate. If the kernel reports a hit whose
        # sighash doesn't reproduce on CPU, the indices won't satisfy the
        # actual script and R2 would burn fleet hours against a phantom.
        if not getattr(args, 'skip_digest_verify', False):
            r1_indices = r1_hit.get('combo') or r1_hit.get('indices') or ''
            r1_sighash = r1_hit.get('sighash')
            r1_pubhash = r1_hit.get('pubhash')
            # Fallback: if the JSON doesn't have sighash (older run_digest.sh
            # didn't capture it), try to fetch from the kernel's per-GPU
            # summary files which always contain full details.
            if not r1_sighash:
                print(f"\n  hit JSON lacks sighash; pulling kernel summary from inst {r1_machine.instance_id}...")
                details = _fetch_digest_hit_details(r1_machine, round_num=1)
                if details:
                    r1_sighash = details.get('sighash')
                    r1_pubhash = details.get('pubhash')
                    if not r1_indices and details.get('indices'):
                        r1_indices = details['indices']
                    print(f"  ✓ recovered: sighash={r1_sighash[:16] if r1_sighash else 'NONE'}...")
            ok = _verify_digest_on_cpu(round_num=1, indices_csv=r1_indices,
                                        seq_hex=seq_hex, locktime=locktime,
                                        gpu_sighash=r1_sighash,
                                        gpu_pubhash=r1_pubhash)
            if not ok:
                print("\n  ✗ R1 VERIFICATION FAILED — hit does NOT reproduce on CPU.")
                print("    The indices do not satisfy the actual R1 script.")
                print("    Refusing to start R2 on a phantom hit. Investigate kernel bug.")
                print("    (Use --skip-digest-verify to override, NOT recommended.)")
                # Stop the still-running R1 kernels before returning so we
                # don't leave the fleet burning.
                _kill_kernels_on_fleet(machines, label="R1 (post-failed-verify)")
                return

        # Stop r1 everywhere
        for m in machines:
            ssh_exec(m, "pkill -x qsb_digest 2>/dev/null || true", timeout=15)

    # ── Digest R2 ──
    print(f"\n═══ DIGEST R2 STAGE — searching at (seq={seq_hex}, lt={locktime}) ═══")
    print(f"  starting digest r2 on all {len(machines)} machines...")
    for m in machines:
        ok = _start_digest(m, 2, seq_hex, locktime, total_gpus)
        if not ok:
            print(f"  WARN: failed to start digest r2 on inst {m.instance_id}")

    r2_hit, r2_machine = _poll_for_first_hit(
        machines, lambda m: _check_digest_hit(m, 2), "digest_r2_status",
        poll_interval=args.poll_interval, timeout=args.digest_timeout,
    )
    if r2_hit is None:
        print("\n  ✗ NO R2 HIT — fleet exhausted without a hit. Bug.")
        return
    print(f"\n  ✓ R2 HIT from inst {r2_machine.instance_id}:")
    print(json.dumps(r2_hit, indent=2))

    # CPU-verify R2 — catches the same class of kernel bug.
    if not getattr(args, 'skip_digest_verify', False):
        r2_indices = r2_hit.get('combo') or r2_hit.get('indices') or ''
        r2_sighash = r2_hit.get('sighash')
        r2_pubhash = r2_hit.get('pubhash')
        if not r2_sighash:
            print(f"\n  hit JSON lacks sighash; pulling kernel summary from inst {r2_machine.instance_id}...")
            details = _fetch_digest_hit_details(r2_machine, round_num=2)
            if details:
                r2_sighash = details.get('sighash')
                r2_pubhash = details.get('pubhash')
                if not r2_indices and details.get('indices'):
                    r2_indices = details['indices']
        ok = _verify_digest_on_cpu(round_num=2, indices_csv=r2_indices,
                                    seq_hex=seq_hex, locktime=locktime,
                                    gpu_sighash=r2_sighash,
                                    gpu_pubhash=r2_pubhash)
        if not ok:
            print("\n  ✗ R2 VERIFICATION FAILED — hit does NOT reproduce on CPU.")
            print("    Saved hits anyway for manual inspection. DO NOT")
            print("    assemble a tx with these indices — it will fail on broadcast.")
            # Continue to save the hit files so user can inspect, but flag the issue.

    # Stop r2 everywhere
    for m in machines:
        ssh_exec(m, "pkill -x qsb_digest 2>/dev/null || true", timeout=15)

    # Build consolidated gpu_hits.json for test_spending_tx.py
    consolidated = {
        "pin_locktime": pin_hit['locktime'],
        "pin_sequence": seq_hex,
        "round1_indices": [int(x) for x in r1_hit['combo'].split(",")],
        "round2_indices": [int(x) for x in r2_hit['combo'].split(",")],
        "_meta": {
            "pin_machine": pin_machine.instance_id,
            "r1_machine": r1_machine.instance_id,
            "r2_machine": r2_machine.instance_id,
            "pin_hit": pin_hit, "r1_hit": r1_hit, "r2_hit": r2_hit,
        },
    }
    with open("gpu_hits.json", "w") as f:
        json.dump(consolidated, f, indent=2)

    print(f"\n═══════════════════════════════════════════════════════════════════")
    print(f"  ✅ FULL HIT CHAIN FOUND")
    print(f"═══════════════════════════════════════════════════════════════════")
    print(f"  Saved to gpu_hits.json")
    print(f"  Next steps:")
    print(f"    1. Verify on CPU:")
    print(f"       python3 ../verify/verify_hit.py pin --locktime {pin_hit['locktime']} \\")
    print(f"           --sequence {seq_hex} --funding-txid <txid>")
    print(f"    2. Run regtest spending tx test:")
    print(f"       python3 ../regtest/test_spending_tx.py --hits gpu_hits.json")
    print(f"    3. Tear down fleet (we have what we need):")
    print(f"       python3 .../qsb_fleet.py destroy -y")


def cmd_prefetch_debs(args):
    """Download libssl-dev .deb files locally so they can be bundled with the
    fleet bootstrap as an offline fallback.

    Scrapes packages.ubuntu.com/<codename>/amd64/<pkg>/download for the CURRENT
    download URL — robust against version rollovers (Ubuntu cycles point releases
    every few weeks and removes old ones from the archive, so hardcoded URLs
    eventually 404).

    After running this, future `fleet search` runs include the .deb files in
    the bundle automatically.
    """
    import urllib.request
    import urllib.error
    import re

    # (codename, package_name) for both Ubuntu LTS versions vast.ai uses.
    # libssl-dev needs its libssl3 (jammy) / libssl3t64 (noble) runtime dep.
    PACKAGES = [
        ("jammy",  "libssl-dev"),
        ("jammy",  "libssl3"),
        ("noble",  "libssl-dev"),
        ("noble",  "libssl3t64"),
    ]
    UA = ("Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0")

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Looking up current .deb URLs from packages.ubuntu.com")
    print(f"Saving to: {dest}/")

    def find_current_url(codename: str, pkg: str) -> Optional[str]:
        """Scrape /<codename>/amd64/<pkg>/download for a current .deb mirror URL."""
        url = f"https://packages.ubuntu.com/{codename}/amd64/{pkg}/download"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=15) as r:
                html = r.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as e:
            print(f"    (could not reach {url}: {e})")
            return None
        # Look for any URL ending in <pkg>_<version>_amd64.deb. Prefer
        # archive.ubuntu.com mirrors since they're publicly reachable.
        debs = re.findall(rf'https?://[^\s"\'<>]+/{re.escape(pkg)}_[^/\s"\'<>]+_amd64\.deb',
                          html)
        if not debs:
            return None
        # Prefer archive.ubuntu.com (anycast, always reachable globally)
        for d in debs:
            if "archive.ubuntu.com" in d:
                return d
        return debs[0]

    n_ok, n_fail = 0, 0
    for codename, pkg in PACKAGES:
        print(f"  [{codename}/{pkg}]", end=" ")
        deb_url = find_current_url(codename, pkg)
        if not deb_url:
            print(f"FAILED to find current URL")
            n_fail += 1
            continue
        fname = deb_url.rsplit("/", 1)[-1]
        out = dest / fname
        if out.exists() and out.stat().st_size > 0:
            print(f"already have {fname}")
            n_ok += 1
            continue
        # Clean stale jammy/noble files for this package — only keep current
        for old in dest.glob(f"{pkg}_*_amd64.deb"):
            if old.name != fname:
                old.unlink()
        print(f"fetching {fname}...", end=" ", flush=True)
        try:
            req = urllib.request.Request(deb_url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            out.write_bytes(data)
            print(f"OK ({len(data)} bytes)")
            n_ok += 1
        except (urllib.error.URLError, OSError) as e:
            print(f"FAILED: {e}")
            n_fail += 1

    print()
    print(f"Result: {n_ok}/{len(PACKAGES)} downloaded successfully.")
    if n_ok > 0:
        print(f"Bundle build will include them automatically. Bootstrap on each")
        print(f"machine will fall back to dpkg -i if apt-get install fails.")
    if n_fail > 0:
        print(f"\n{n_fail} downloads failed. The bootstrap will still try apt-get;")
        print(f"the .deb files are only a fallback, not strictly required.")


def cmd_calibrate(args):
    """Run the digest kernel in CALIBRATE mode (relaxed DER check, no
    r_on_curve filter — 512x more permissive) on the current fleet for the
    specified duration, then aggregate hit counts.

    Purpose: distinguish 'no R1 hits is bad luck' from 'no R1 hits is a bug'.

    At calibrate difficulty (~2^37), expected hits per fleet-hour:
      - 86 GPUs at 30M/s each → ~2.6 GH/s
      - In 30 min → ~33 hits expected.
      - P(0 hits in 30 min, no bug) ≈ 2e-15.

    Interpretation:
      - >20 calibrate hits  → kernel works correctly, R1 zero was bad luck
      - 0-3 calibrate hits → bug in suffix construction, midstate, or DER check
      - intermediate       → suspicious; investigate
    """
    machines = load_fleet()
    if not machines:
        print("No fleet on record. Run `search` or `launch` first.")
        return

    duration_sec = int(args.duration_minutes * 60)
    print(f"\n═══ CALIBRATE MODE ═══")
    print(f"  Fleet: {len(machines)} machines")
    print(f"  Duration: {args.duration_minutes} minutes")
    print(f"  Pin: seq={args.seq} lt={args.lt}")
    print(f"  Round: {args.round}")
    print()

    # Step 1: stop any prior digest searches
    print(f"─── stopping any prior searches on all machines ───")
    for m in machines:
        ssh_exec(m, "pkill -9 qsb_digest 2>/dev/null; pkill -9 qsb_real 2>/dev/null; "
                 "rm -f /root/qsb_run/results/digest_calibrate_*.txt", timeout=15)

    # Step 2: launch calibrate on each machine
    print(f"\n─── launching calibrate on all machines ───")
    seq_hex = args.seq if args.seq.startswith("0x") else f"0x{args.seq}"
    launched = 0
    for m in machines:
        # Each machine: launch all local GPUs with same slice (don't worry about
        # coverage — calibrate just needs work; same-slice = max throughput).
        # We also set total_gpus=1 so each GPU thinks it owns the whole space
        # and starts from the lex-smallest combo.
        cmd = (
            f"cd /root/qsb_run && "
            f"NGPU=$(nvidia-smi -L | wc -l); "
            f"for g in $(seq 0 $((NGPU-1))); do "
            f"  nohup ./qsb_digest digest_r{args.round}.bin $g {seq_hex} {args.lt} "
            f"    1 0 single_hash calibrate "
            f"    > results/calibrate_gpu$g.log 2>&1 & "
            f"done; "
            f"sleep 2; "
            f"echo CALIBRATE_RUNNING_$(pgrep -c qsb_digest)_PROCS"
        )
        rc, out, err = ssh_exec(m, cmd, timeout=30)
        if rc == 0 and "CALIBRATE_RUNNING_" in (out or ""):
            launched += 1
            n_procs = (out or "").strip().split("_")[-2]
            print(f"  inst {m.instance_id}: {n_procs} qsb_digest processes started")
        else:
            print(f"  inst {m.instance_id}: ✗ FAILED to start ({rc}): {(err or out or '')[:100]}")

    if launched == 0:
        print("\n✗ No machines launched calibrate. Aborting.")
        return
    print(f"\n  ✓ {launched}/{len(machines)} machines running calibrate")

    # Step 3: wait the duration
    print(f"\n─── running for {args.duration_minutes} minutes ───")
    import time
    t_start = time.time()
    while True:
        elapsed = time.time() - t_start
        if elapsed >= duration_sec:
            break
        # Print a status update every minute
        time.sleep(min(60, duration_sec - elapsed))
        elapsed = time.time() - t_start
        print(f"  [{elapsed:.0f}s / {duration_sec}s elapsed]")

    # Step 4: stop and collect
    print(f"\n─── stopping calibrate runs ───")
    for m in machines:
        ssh_exec(m, "pkill -9 qsb_digest 2>/dev/null; sleep 1", timeout=15)

    print(f"\n─── collecting hit counts from each machine ───")
    total_hits = 0
    per_machine_hits = []
    for m in machines:
        rc, out, err = ssh_exec(
            m,
            "cat /root/qsb_run/results/digest_calibrate_*.txt 2>/dev/null | "
            "grep -c '^indices=' || echo 0",
            timeout=15)
        try:
            n = int((out or "0").strip().split()[-1])
        except (ValueError, IndexError):
            n = 0
        per_machine_hits.append((m.instance_id, n))
        total_hits += n
        print(f"  inst {m.instance_id}: {n} calibrate hits")

    # Step 5: also fetch the throughput
    print(f"\n─── checking throughput ───")
    for m in machines[:1]:  # one machine is enough for spot check
        rc, out, err = ssh_exec(
            m,
            "tail -3 /root/qsb_run/results/calibrate_gpu0.log 2>/dev/null",
            timeout=10)
        if out:
            print(f"  inst {m.instance_id} GPU0 last 3 lines:")
            for ln in out.strip().split("\n"):
                print(f"    {ln}")

    # Step 6: verdict
    print()
    print("═" * 60)
    print(f"VERDICT")
    print("═" * 60)
    print(f"  Total calibrate hits across {launched} machines: {total_hits}")
    print()
    # Compute expected
    n_gpus_estimate = launched * 8  # rough; could be 14
    expected = n_gpus_estimate * 30e6 * duration_sec / (2**37)
    print(f"  Expected (assuming kernel works): ~{expected:.0f} hits")
    print()
    if total_hits == 0:
        print(f"  ✗ ZERO calibrate hits. Probability under no-bug ≈ e^(-{expected:.0f}) ≈ 0.")
        print(f"    There IS a bug somewhere. Run `debug-digest` to find it.")
    elif total_hits < expected / 10:
        print(f"  ⚠ Very few hits ({total_hits} vs expected ~{expected:.0f}).")
        print(f"    Suggests partial bug or upstream issue. Investigate.")
    elif total_hits >= expected / 4:
        print(f"  ✓ Plenty of calibrate hits. Kernel pipeline is working.")
        print(f"    Zero strict R1 hits was the 30%-bad-luck outcome.")
        print(f"    Restart full R1 with same pin (or get a new pin).")
    else:
        print(f"  ~ Some calibrate hits but fewer than expected. Marginal signal.")
        print(f"    Probably OK but worth running for longer to be sure.")


def cmd_smoke(args):
    """Rent 1 cheap GPU, compile kernels, run a tiny search batch, destroy.

    Purpose: end-to-end validation that everything works before the user
    commits to a real 10-machine fleet. Expected cost: $0.05 - $0.20."""
    work_dir = Path.cwd()
    gpu_dir = Path(args.gpu_dir).resolve()
    check_prereqs(work_dir)
    if not gpu_dir.exists():
        print(f"ERROR: --gpu-dir {gpu_dir} not found")
        sys.exit(1)

    print("═" * 67)
    print("  FLEET SMOKE TEST — rent 1 cheap GPU, verify kernels work")
    print("═" * 67)
    print(f"  GPU:     {args.gpu}")
    print(f"  Max dph: ${args.max_dph:.2f}/hour")
    print()

    print("─── searching for offers ───")
    try:
        offers = search_offers(gpu_name=args.gpu, max_dph=args.max_dph,
                               min_reliability=0.95)
    except Exception as e:
        print(f"vastai search failed: {e}")
        sys.exit(1)
    if not offers:
        print(f"No {args.gpu} offers found under ${args.max_dph}/hour.")
        print("Try --max-dph 0.30 or a different --gpu model.")
        sys.exit(1)

    offer = offers[0]
    print(f"  chosen: offer {offer['id']} ({offer.get('gpu_name')} × "
          f"{offer.get('num_gpus')}, ${offer.get('dph_total'):.3f}/h)")

    if not args.yes:
        resp = input(f"\nRent 1 smoke-test instance at ${offer['dph_total']:.3f}/h? [y/N] ")
        if resp.strip().lower() != "y":
            print("Aborted.")
            return

    print("\n─── building bundle ───")
    bundle = build_bundle(work_dir, gpu_dir)
    print(f"  bundle: {bundle.stat().st_size / 1024:.0f} KB")

    print("\n─── renting ───")
    try:
        iid = create_instance(offer["id"], image=args.image, disk=args.disk,
                               label="qsb-smoke")
    except Exception as e:
        print(f"create_instance failed: {e}")
        sys.exit(1)
    print(f"  instance {iid}; waiting for SSH...")

    m = Machine(
        instance_id=iid, offer_id=offer["id"],
        gpu_name=offer.get("gpu_name", "?"), num_gpus=offer.get("num_gpus", 1),
        dph=offer.get("dph_total", 0.0),
        seq_hex="0xfffffffe", lt_start=0, lt_range=1_000_000,
        global_gpu_offset=0,
    )
    if not wait_for_ssh(m, timeout=args.ssh_timeout):
        print(f"  ✗ instance {iid} did not come up in time")
        print(f"  Destroying to stop billing...")
        destroy_instance(iid)
        sys.exit(1)
    print(f"  SSH up at {m.ssh_host}:{m.ssh_port}")

    print("\n─── uploading bundle ───")
    if not scp_upload(m, bundle, "/root/bundle.tar.gz"):
        print("  ✗ scp upload failed")
        destroy_instance(iid)
        sys.exit(1)
    print("  ✓ uploaded")

    print("\n─── compiling kernels on remote ───")
    compile_script = r"""
set -e
cd /root && mkdir -p qsb_smoke && cd qsb_smoke
tar -xzf /root/bundle.tar.gz
apt-get update -qq && apt-get install -y -qq build-essential libssl-dev > /tmp/apt.log 2>&1
echo "── compiling qsb_real ──"
nvcc -O3 -o qsb_real qsb_real_search.cu -lcrypto -lm 2>/tmp/real.log
echo "── compiling qsb_digest ──"
nvcc -O3 -o qsb_digest qsb_digest_search.cu -lcrypto -lm 2>/tmp/digest.log
echo "── verifying binaries ──"
ls -la qsb_real qsb_digest
echo "── checking GPU ──"
nvidia-smi | head -20
echo "SMOKE_COMPILE_OK"
""".strip()
    rc, out, err = ssh_exec(m, compile_script, timeout=600)
    if rc != 0 or 'SMOKE_COMPILE_OK' not in out:
        print("  ✗ kernel compilation failed:")
        print(out[-1500:])
        print(err[-500:])
        print("\nDestroying instance to stop billing...")
        destroy_instance(iid)
        sys.exit(1)
    print("  ✓ both kernels compiled successfully")
    # Show nvidia-smi excerpt
    for line in out.split('\n'):
        if 'CUDA Version' in line or 'MiB' in line or 'Off' in line:
            print(f"  {line.strip()}")

    print("\n─── running 5-second pin search batch ───")
    # Start a short-lived search and watch for any output that proves the GPU
    # is running the kernel and producing sensible log output
    run_script = r"""
cd /root/qsb_smoke
export QSB_SEQUENCE=0xfffffffe
export QSB_LT_START=0
export QSB_LT_RANGE=1000000
export QSB_GLOBAL_OFFSET=0
export QSB_TOTAL_GPUS=1
timeout 30 ./qsb_real pinning.bin 0 0xfffffffe 0 single_hash 2>&1 | head -40 || true
echo "── search output end ──"
""".strip()
    rc, out, err = ssh_exec(m, run_script, timeout=90)
    print("  kernel stdout (first 40 lines):")
    for line in out.split('\n')[:40]:
        print(f"    {line}")
    kernel_started = any(k in out for k in
                         ['Loaded:', 'GTable', 'Searching', 'Done:', 'M/s'])
    if not kernel_started:
        print("\n  ⚠  pin kernel may not have started properly — inspect output above")
    else:
        print("  ✓ pin kernel ran and produced expected output")

    # ── DIGEST KERNEL ↔ EMULATOR EQUIVALENCE TEST ──
    # Picks a fixed (seq, lt, indices) tuple, runs qsb_digest in DEBUG mode
    # which prints the kernel's computed sighash, then runs the CPU emulator
    # with the SAME inputs and compares. Catches kernel/emulator mismatches
    # BEFORE the user commits to a real fleet.
    digest_match = None  # None=skipped, True=match, False=mismatch
    test_seq_hex = "0x80000001"
    test_lt = 12345
    test_indices = "0,1,2,3,4,5,6,7,8"  # any 9 sorted storage indices
    print("\n─── digest kernel ↔ CPU emulator equivalence test ───")
    print(f"    test inputs: seq={test_seq_hex} lt={test_lt} indices={test_indices}")
    debug_script = (
        "cd /root/qsb_smoke && "
        f"timeout 60 ./qsb_digest digest_r1.bin 0 {test_seq_hex} {test_lt} "
        f"single_hash debug {test_indices} 2>&1 | tail -80 || true; "
        'echo "── debug output end ──"'
    )
    rc, out, err = ssh_exec(m, debug_script, timeout=120)
    kernel_sighash = None
    import re as _re
    for line in (out or "").split('\n'):
        s = line.strip()
        if 'sighash' in s.lower():
            mat = _re.search(r'([0-9a-fA-F]{64})', s)
            if mat:
                kernel_sighash = mat.group(1).lower()
                break
    if not kernel_sighash:
        print("    ⚠  could not parse kernel\'s sighash from debug output")
        print("       Last 30 lines of kernel debug stdout:")
        for line in (out or "").split('\n')[-30:]:
            print(f"        {line}")
    else:
        print(f"    kernel sighash:   {kernel_sighash}")
        try:
            _spec = importlib.util.spec_from_file_location(
                "gpu_emulator",
                str(Path(__file__).resolve().parent.parent / 'verify' / 'gpu_emulator.py'),
            )
            _gem = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_gem)
            with open(work_dir / "gpu_digest_r1_params.json") as _f:
                _params = json.load(_f)
            n = _params['n']
            storage_idxs = [int(s) for s in test_indices.split(',')]
            state_idxs = sorted([n - 1 - i for i in storage_idxs])
            _emu = _gem.emulate_digest_round(
                _params, state_idxs, sighash_type=0x01,
                sequence=int(test_seq_hex, 0), locktime=test_lt,
            )
            emu_sighash = _emu['sighash'].hex()
            print(f"    emulator sighash: {emu_sighash}")
            digest_match = (emu_sighash == kernel_sighash)
            if digest_match:
                print(f"    ✓ KERNEL ↔ EMULATOR MATCH — sighash layout is correct")
            else:
                print(f"    ✗ KERNEL ↔ EMULATOR MISMATCH — DO NOT RUN A REAL FLEET")
                print(f"      Most likely cause: tx_suffix layout drift or stale .bin file.")
        except Exception as _e:
            print(f"    ⚠  emulator failed to run: {_e}")
            digest_match = None

    print("\n─── destroying smoke-test instance ───")
    destroy_instance(iid)
    print(f"  ✓ instance {iid} destroyed, billing stopped")

    print()
    print("═" * 67)
    if digest_match is False:
        print("  ❌ SMOKE TEST FAILED — kernel sighash does NOT match emulator.")
        print("     Do NOT launch a real fleet until this is fixed.")
    elif digest_match is True and kernel_started:
        print("  ✅ SMOKE TEST PASSED — kernels work AND match emulator. Safe to launch.")
    elif digest_match is None and kernel_started:
        print("  ⚠  SMOKE TEST PARTIAL — kernels run but equivalence check skipped.")
        print("     Inspect debug output above before scaling up.")
    else:
        print("  ⚠  SMOKE TEST INCONCLUSIVE — review the kernel output above")
    print("═" * 67)



def cmd_restart_digest(args):
    """Start digest R1 on the existing fleet using a previously-found pin.

    Use this when:
      - You have a verified pin (seq, lt) — typically because the launcher
        crashed or the laptop closed AFTER pin verification but BEFORE digest
        R1 fully started.
      - The fleet is still rented; you want to skip the bootstrap and pin
        stages and just kick off digest R1.

    Generates LPT tile files, uploads them, kicks off run_digest.sh on each
    machine, and EXITS. No polling. Use `check-results` to monitor.

    Pre-requisites:
      - Fleet must be loaded (`fleet.json` exists)
      - All target machines must have /root/qsb_run/qsb_digest already built
      - Local cwd must contain qsb_state.json (used to determine n_pool, t_sel)
    """
    machines = load_fleet()
    if not machines:
        print("No fleet on record.")
        return

    # Sanity-check binaries on each machine
    print(f"\n═══ RESTART DIGEST R{args.round} ═══")
    print(f"  pin: seq={args.seq} lt={args.lt}")
    print(f"  Probing {len(machines)} fleet machines for digest binary...")
    ready = []
    for m in machines:
        # Comprehensive probe: directory, binary, run_digest.sh, params files.
        # If any of these is missing, the machine can't run digest.
        probe_cmd = (
            "test -d /root/qsb_run && "
            "test -x /root/qsb_run/qsb_digest && "
            "test -f /root/qsb_run/run_digest.sh && "
            "test -f /root/qsb_run/digest_r1.bin && "
            "echo OK || echo MISSING"
        )
        rc, out, err = ssh_exec(m, probe_cmd, timeout=20)
        if rc == 0 and "OK" in (out or ""):
            ready.append(m)
            print(f"  ✓ inst {m.instance_id}: ready ({m.num_gpus} GPUs)")
        else:
            reason = (out or err or "").strip()[:80] or f"rc={rc}"
            print(f"  ✗ inst {m.instance_id}: SKIPPED ({reason})")

    if not ready:
        print("\n✗ No ready machines.")
        return

    # Re-pack global_gpu_offset over the ready set for tight partitioning
    cum = 0
    for m in ready:
        m.global_gpu_offset = cum
        cum += m.num_gpus
    save_fleet(ready)
    total_gpus = sum(m.num_gpus for m in ready)
    print(f"\n  Using {total_gpus} GPUs across {len(ready)} machines.")

    # Stop any prior digest runs and clear stale state
    print(f"\n  stopping any prior digest runs and clearing stale state...")
    for m in ready:
        ssh_exec(
            m,
            "pkill -x qsb_digest 2>/dev/null; "
            "pkill -x qsb_real 2>/dev/null; "
            "rm -f /root/qsb_run/digest_r1_status /root/qsb_run/digest_r2_status; "
            "rm -f /root/qsb_run/results/digest_hit_*.txt; "
            "rm -f /root/qsb_run/results/digest_summary_gpu*.txt; "
            "echo CLEAN",
            timeout=15)

    # Compute LPT tile partition
    print(f"\n  computing LPT tile partition for {total_gpus} GPUs...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
    try:
        from tile_partition import partition_for_fleet, write_tile_file  # type: ignore
    except ImportError:
        print("  ✗ tile_partition module not found")
        return

    state_path = Path("qsb_state.json")
    if not state_path.exists():
        print(f"  ✗ qsb_state.json not found in cwd")
        return
    with open(state_path) as f:
        state = json.load(f)
    n_pool = state["n"]
    t_sel_r1 = state["t1s"] + state["t1b"]
    t_sel_r2 = state["t2s"] + state["t2b"]

    tiles_dir = Path("digest_tiles")
    tiles_dir.mkdir(exist_ok=True)

    def gen_and_upload(round_num, t_sel):
        print(f"  round {round_num}: partitioning C({n_pool},{t_sel}) across {total_gpus} GPUs...")
        assignment, stats = partition_for_fleet(n_pool, t_sel, total_gpus)
        print(f"    → {stats['num_tiles']} tiles, "
              f"imbalance {stats['imbalance_ratio']:.4f}x, "
              f"covered={stats['covered']}")
        if not stats["covered"]:
            print(f"    ✗ partition does NOT fully cover the search space")
            return False
        for gpu_id, tlist in assignment.items():
            p = tiles_dir / f"digest_r{round_num}_tiles_gpu_{gpu_id}.bin"
            write_tile_file(str(p), tlist)
        print(f"    uploading tile files to {len(ready)} machines (parallel)...")
        import concurrent.futures
        def upload_to_machine(m):
            ok = True
            for g in range(m.num_gpus):
                gid = m.global_gpu_offset + g
                src = tiles_dir / f"digest_r{round_num}_tiles_gpu_{gid}.bin"
                if not scp_upload(m, src, f"/root/qsb_run/digest_r{round_num}_tiles_gpu_{gid}.bin"):
                    ok = False
            return ok
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(ready), 16)) as ex:
            results = list(ex.map(upload_to_machine, ready))
        success_count = sum(results)
        print(f"    ✓ uploaded to {success_count}/{len(ready)} machines")
        return success_count > 0

    if not gen_and_upload(args.round, t_sel_r1 if args.round == 1 else t_sel_r2):
        print(f"  ✗ failed to upload tiles")
        return

    # Optional: also pre-upload R2 tiles
    if args.round == 1:
        gen_and_upload(2, t_sel_r2)

    # Kick off digest R<round> on each machine
    seq_hex = args.seq if args.seq.startswith("0x") else f"0x{args.seq}"
    print(f"\n  starting digest r{args.round} on all {len(ready)} machines...")
    actually_running = []
    failed = []
    for m in ready:
        ok = _start_digest(m, args.round, seq_hex, args.lt, total_gpus)
        if ok:
            actually_running.append(m)
        else:
            failed.append(m)
            print(f"  WARN: failed to start digest r{args.round} on inst {m.instance_id}")

    if failed:
        print(f"\n  ⚠ {len(failed)} machines couldn't start digest:")
        for m in failed:
            print(f"      inst {m.instance_id}: tiles assigned but never run "
                  f"(coverage gap)")
        print()
        print(f"  COVERAGE GAP: this means ~{sum(m.num_gpus for m in failed)} GPUs' worth")
        print(f"  of tiles are unsearched. If R1 finds zero hits, you can't")
        print(f"  conclude C(150,9) was actually exhausted.")
        print()
        print(f"  RECOMMENDED: Ctrl+C any monitoring, destroy the failed instances,")
        print(f"  then re-run restart-digest. Failed machines:")
        for m in failed:
            print(f"      vastai destroy instance {m.instance_id}")
        print()
        print(f"  After destroying, re-run: python3 ... restart-digest {args.seq} {args.lt}")
        # Don't update fleet.json yet — wait for user to destroy. Re-running
        # restart-digest will probe again and skip the destroyed ones.

    if actually_running:
        running_gpus = sum(m.num_gpus for m in actually_running)
        print(f"\n  ✓ Digest R{args.round} kernels actually launched on "
              f"{len(actually_running)}/{len(ready)} machines ({running_gpus} GPUs).")
        if not failed:
            print(f"  Search runs ~{int(running_gpus * 30e6 / 1e9)} GH/s; "
                  f"expected ~9-12 hours to exhaust C({n_pool},{t_sel_r1}).")
            print()
            print(f"  Safe to close laptop.")
            print(f"  Monitor with: python3 .../qsb_fleet.py check-results")
    else:
        print(f"\n  ✗ ZERO machines actually running digest. Nothing to do.")


def cmd_resume_search(args):
    """Start the search on fleet machines that already have qsb_real and
    qsb_digest binaries built — without re-running bootstrap.

    Use this when the launcher is hanging because some machines never came
    up. Ctrl+C the launcher, run this, and it'll start search on whichever
    machines are actually ready, ignoring the dead ones.

    This skips the pin stage if --use-pin-locktime is supplied, otherwise
    it runs the pin stage on the ready machines.
    """
    machines = load_fleet()
    if not machines:
        print("No fleet on record.")
        return

    print(f"\n═══ RESUME SEARCH ═══")
    print(f"  Probing {len(machines)} fleet machines for ready binaries...")
    print()
    ready = []
    not_ready = []
    for m in machines:
        rc, out, err = ssh_exec(
            m,
            "test -x /root/qsb_run/qsb_real && test -x /root/qsb_run/qsb_digest "
            "&& echo BINARIES_OK || echo MISSING",
            timeout=20)
        if rc == 0 and "BINARIES_OK" in (out or ""):
            ready.append(m)
            print(f"  ✓ inst {m.instance_id}: binaries ready")
        else:
            not_ready.append(m)
            print(f"  ✗ inst {m.instance_id}: NOT ready (rc={rc}, out={(out or '').strip()[:50]})")

    if not ready:
        print("\n✗ No machines have binaries ready. Cannot resume.")
        print("  Either wait longer, run bootstrap manually, or destroy and relaunch.")
        return

    print(f"\n  {len(ready)}/{len(machines)} machines ready")
    if not_ready:
        print(f"  {len(not_ready)} not ready — they will be IGNORED for this search.")
        print(f"  (They're still rented and billed. Run `destroy` later or `reprovision`.)")

    if not args.yes:
        resp = input(f"\nProceed with {len(ready)} ready machines? [y/N] ")
        if resp.strip().lower() != "y":
            print("Aborted.")
            return

    # Re-pack global_gpu_offset to be contiguous on the ready set
    cum = 0
    for m in ready:
        m.global_gpu_offset = cum
        cum += m.num_gpus
    save_fleet(ready)

    total_gpus = sum(m.num_gpus for m in ready)
    print(f"\nUsing {total_gpus} GPUs across {len(ready)} machines.")

    # ── Pin stage ──
    pin_flags = "single_hash"
    if args.pin_seq_start:
        pin_flags = f"single_hash seq_start={args.pin_seq_start}"
        print(f"  Pin seq_start override: {args.pin_seq_start}")

    if args.use_pin_locktime is not None:
        seq_hex = args.use_pin_sequence
        if not seq_hex.startswith("0x"):
            seq_hex = f"0x{seq_hex}"
        pin_hit = {
            "sequence": int(seq_hex, 16),
            "sequence_hex": seq_hex,
            "locktime": args.use_pin_locktime,
            "hash_choice": args.use_pin_hash_choice,
            "recid": args.use_pin_recid,
            "machine_offset": -1, "_supplied": True,
        }
        print(f"\n═══ PIN STAGE — SKIPPED (using supplied) ═══")
        print(f"  seq={seq_hex} lt={args.use_pin_locktime}")
    else:
        print(f"\n═══ PIN STAGE — searching ═══")
        for m in ready:
            ok = _start_pin(m, total_gpus, extra_flags=pin_flags)
            if not ok:
                print(f"  WARN: failed to start pin on inst {m.instance_id}")
        print(f"  polling for first pin hit...")
        pin_hit, pin_machine = _poll_for_first_hit(
            ready, _check_pin_hit, "pin_status",
            poll_interval=30.0, timeout=4*3600.0)
        if pin_hit is None:
            print("\n✗ No pin hit. Aborting.")
            return
        print(f"\n  ✓ PIN HIT from inst {pin_machine.instance_id}")
        print(json.dumps(pin_hit, indent=2))
        # Stop pin everywhere
        for m in ready:
            ssh_exec(m, "pkill -x qsb_real 2>/dev/null", timeout=15)

    # ── Digest R1 ──
    print(f"\n═══ DIGEST R1 STAGE ═══")
    print(f"  pin: seq={pin_hit['sequence_hex']} lt={pin_hit['locktime']}")
    print(f"  Use the original `search` flow's logic now: kicking off digest on each machine")
    print(f"  Note: this command DOES NOT re-upload tile files. If you want balanced LPT")
    print(f"        tiles, use the full `search` command instead. Otherwise mod-N partitioning")
    print(f"        is used (slightly less balanced but still correct coverage).")

    seq_hex = pin_hit["sequence_hex"]
    lt = pin_hit["locktime"]
    for m in ready:
        cmd = (f"cd /root/qsb_run && rm -f digest_r1_status digest_r1_hit.json results/digest_hit_*.txt; "
               f"export QSB_TOTAL_GPUS={total_gpus}; "
               f"export QSB_GLOBAL_OFFSET={m.global_gpu_offset}; "
               f"chmod +x run_digest.sh 2>/dev/null; "
               f"nohup ./run_digest.sh 1 {seq_hex} {lt} single_hash "
               f"> /root/qsb_run/digest.log 2>&1 & "
               f"sleep 2; pgrep -c qsb_digest")
        rc, out, err = ssh_exec(m, cmd, timeout=30)
        n = (out or "0").strip().split()[-1] if out else "?"
        print(f"  inst {m.instance_id}: {n} qsb_digest processes")

    print(f"\n  Search running. Use `check-results` to monitor — safe to close laptop.")
    print(f"  When ready, run: python3 ... check-results")



    """Rent 1 cheap GPU, compile kernels, run a tiny search batch, destroy.

    Purpose: end-to-end validation that everything works before the user
    commits to a real 10-machine fleet. Expected cost: $0.05 - $0.20."""
    work_dir = Path.cwd()
    gpu_dir = Path(args.gpu_dir).resolve()
    check_prereqs(work_dir)
    if not gpu_dir.exists():
        print(f"ERROR: --gpu-dir {gpu_dir} not found")
        sys.exit(1)

    print("═" * 67)
    print("  FLEET SMOKE TEST — rent 1 cheap GPU, verify kernels work")
    print("═" * 67)
    print(f"  GPU:     {args.gpu}")
    print(f"  Max dph: ${args.max_dph:.2f}/hour")
    print()

    print("─── searching for offers ───")
    try:
        offers = search_offers(gpu_name=args.gpu, max_dph=args.max_dph,
                               min_reliability=0.95)
    except Exception as e:
        print(f"vastai search failed: {e}")
        sys.exit(1)
    if not offers:
        print(f"No {args.gpu} offers found under ${args.max_dph}/hour.")
        print("Try --max-dph 0.30 or a different --gpu model.")
        sys.exit(1)

    offer = offers[0]
    print(f"  chosen: offer {offer['id']} ({offer.get('gpu_name')} × "
          f"{offer.get('num_gpus')}, ${offer.get('dph_total'):.3f}/h)")

    if not args.yes:
        resp = input(f"\nRent 1 smoke-test instance at ${offer['dph_total']:.3f}/h? [y/N] ")
        if resp.strip().lower() != "y":
            print("Aborted.")
            return

    print("\n─── building bundle ───")
    bundle = build_bundle(work_dir, gpu_dir)
    print(f"  bundle: {bundle.stat().st_size / 1024:.0f} KB")

    print("\n─── renting ───")
    try:
        iid = create_instance(offer["id"], image=args.image, disk=args.disk,
                               label="qsb-smoke")
    except Exception as e:
        print(f"create_instance failed: {e}")
        sys.exit(1)
    print(f"  instance {iid}; waiting for SSH...")

    m = Machine(
        instance_id=iid, offer_id=offer["id"],
        gpu_name=offer.get("gpu_name", "?"), num_gpus=offer.get("num_gpus", 1),
        dph=offer.get("dph_total", 0.0),
        seq_hex="0xfffffffe", lt_start=0, lt_range=1_000_000,
        global_gpu_offset=0,
    )
    if not wait_for_ssh(m, timeout=args.ssh_timeout):
        print(f"  ✗ instance {iid} did not come up in time")
        print(f"  Destroying to stop billing...")
        destroy_instance(iid)
        sys.exit(1)
    print(f"  SSH up at {m.ssh_host}:{m.ssh_port}")

    print("\n─── uploading bundle ───")
    if not scp_upload(m, bundle, "/root/bundle.tar.gz"):
        print("  ✗ scp upload failed")
        destroy_instance(iid)
        sys.exit(1)
    print("  ✓ uploaded")

    print("\n─── compiling kernels on remote ───")
    compile_script = r"""
set -e
cd /root && mkdir -p qsb_smoke && cd qsb_smoke
tar -xzf /root/bundle.tar.gz
apt-get update -qq && apt-get install -y -qq build-essential libssl-dev > /tmp/apt.log 2>&1
echo "── compiling qsb_real ──"
nvcc -O3 -o qsb_real qsb_real_search.cu -lcrypto -lm 2>/tmp/real.log
echo "── compiling qsb_digest ──"
nvcc -O3 -o qsb_digest qsb_digest_search.cu -lcrypto -lm 2>/tmp/digest.log
echo "── verifying binaries ──"
ls -la qsb_real qsb_digest
echo "── checking GPU ──"
nvidia-smi | head -20
echo "SMOKE_COMPILE_OK"
""".strip()
    rc, out, err = ssh_exec(m, compile_script, timeout=600)
    if rc != 0 or 'SMOKE_COMPILE_OK' not in out:
        print("  ✗ kernel compilation failed:")
        print(out[-1500:])
        print(err[-500:])
        print("\nDestroying instance to stop billing...")
        destroy_instance(iid)
        sys.exit(1)
    print("  ✓ both kernels compiled successfully")
    # Show nvidia-smi excerpt
    for line in out.split('\n'):
        if 'CUDA Version' in line or 'MiB' in line or 'Off' in line:
            print(f"  {line.strip()}")

    print("\n─── running 5-second search batch ───")
    # Start a short-lived search and watch for any output that proves the GPU
    # is running the kernel and producing sensible log output
    run_script = r"""
cd /root/qsb_smoke
export QSB_SEQUENCE=0xfffffffe
export QSB_LT_START=0
export QSB_LT_RANGE=1000000
export QSB_GLOBAL_OFFSET=0
export QSB_TOTAL_GPUS=1
timeout 30 ./qsb_real pinning.bin 0 0xfffffffe 0 single_hash 2>&1 | head -40 || true
echo "── search output end ──"
""".strip()
    rc, out, err = ssh_exec(m, run_script, timeout=90)
    print("  kernel stdout (first 40 lines):")
    for line in out.split('\n')[:40]:
        print(f"    {line}")
    # Heuristic: did the kernel start? Look for the typical "Loaded" or similar signs
    kernel_started = any(k in out for k in
                         ['Loaded:', 'GTable', 'Searching', 'Done:', 'M/s'])
    if not kernel_started:
        print("\n  ⚠  kernel may not have started properly — inspect output above")
    else:
        print("  ✓ kernel ran and produced output consistent with normal operation")

    print("\n─── destroying smoke-test instance ───")
    destroy_instance(iid)
    print(f"  ✓ instance {iid} destroyed, billing stopped")

    print()
    print("═" * 67)
    if kernel_started:
        print("  ✅ SMOKE TEST PASSED — safe to run a real fleet with `launch`")
    else:
        print("  ⚠  SMOKE TEST INCONCLUSIVE — review the kernel output above")
    print("═" * 67)



def cmd_check_results(args):
    """Fetch the per-GPU summary files from every machine and aggregate.

    Each GPU writes results/digest_summary_gpu<N>.txt continuously during
    its run. The file always exists (even with 0 hits) and contains:
      STARTED <epoch> ...
      PROGRESS <epoch> ...   (every 60s)
      HIT <epoch> ...        (one per hit, fsync'd)
      STATUS={FOUND|EXHAUSTED|KILLED} <epoch> ...

    This command pulls all summaries and reports unambiguous status,
    independent of whether the launcher was alive when hits occurred.

    Pulls in parallel across machines, and uses ONE SSH per machine that
    streams all summary files concatenated — much faster than per-file SCP.
    """
    machines = load_fleet()
    if not machines:
        print("No fleet on record.")
        return

    out_dir = Path("fleet_summaries")
    out_dir.mkdir(exist_ok=True)
    print(f"\n═══ FLEET RESULT CHECK ═══")
    print(f"  saving summaries to: {out_dir}/")
    print(f"  fetching from {len(machines)} machines in parallel...")
    print()

    # ── Parallel fetch helper ──
    # Single SSH per machine that streams all summary files at once,
    # delimited by ===FILE: markers we can split on.
    def fetch_one(m: 'Machine'):
        """Returns (machine, list_of_(filename, content)) or (machine, None) on failure."""
        # awk-based emit: simpler than shell loop, deterministic delimiter.
        cmd = (
            "cd /root/qsb_run/results 2>/dev/null && "
            "for f in digest_summary_gpu*.txt; do "
            "  [ -f \"$f\" ] || continue; "
            "  echo \"===FILE:$f===\"; "
            "  cat \"$f\"; "
            "  echo \"===END===\"; "
            "done"
        )
        rc, out, err = ssh_exec(m, cmd, timeout=60)
        if rc != 0:
            return (m, None, err or "ssh failed")
        # Parse the stream
        files = []
        cur_name = None
        cur_lines = []
        for ln in (out or "").splitlines():
            if ln.startswith("===FILE:") and ln.endswith("==="):
                cur_name = ln[len("===FILE:"):-len("===")]
                cur_lines = []
            elif ln == "===END===":
                if cur_name:
                    files.append((cur_name, "\n".join(cur_lines) + "\n"))
                cur_name = None
                cur_lines = []
            elif cur_name is not None:
                cur_lines.append(ln)
        return (m, files, None)

    import concurrent.futures
    results = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(machines), 16)) as ex:
        futures = {ex.submit(fetch_one, m): m for m in machines}
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    # Sort by instance_id for stable output
    results.sort(key=lambda r: r[0].instance_id)

    total_hits_across_fleet = 0
    machines_status = []
    for m, files, err in results:
        m_dir = out_dir / f"inst_{m.instance_id}"
        m_dir.mkdir(exist_ok=True)
        if files is None:
            print(f"  inst {m.instance_id}: ✗ FETCH FAILED ({(err or '')[:60]})")
            machines_status.append((m.instance_id, "FETCH_FAILED", 0, 0))
            continue
        if not files:
            print(f"  inst {m.instance_id}: ✗ NO summary files found "
                  f"(kernel may have never started, or files were deleted)")
            machines_status.append((m.instance_id, "NO_SUMMARY", 0, 0))
            continue

        # Save each file locally + parse
        gpu_status = []
        machine_total_hits = 0
        machine_attempts = 0
        for fname, content in files:
            (m_dir / fname).write_text(content)
            hits = sum(1 for ln in content.splitlines() if ln.startswith("HIT "))
            machine_total_hits += hits
            # Find STATUS= line (or INCOMPLETE if missing)
            status = "INCOMPLETE"
            attempts = 0
            elapsed_s = 0.0
            for ln in content.splitlines():
                if ln.startswith("STATUS="):
                    status = ln.split()[0].split("=", 1)[1]
                    for tok in ln.split():
                        if tok.startswith("total_attempts="):
                            try: attempts = int(tok.split("=")[1])
                            except ValueError: pass
                        elif tok.startswith("elapsed_s="):
                            try: elapsed_s = float(tok.split("=")[1])
                            except ValueError: pass
                elif ln.startswith("PROGRESS ") and status == "INCOMPLETE":
                    for tok in ln.split():
                        if tok.startswith("attempts="):
                            try: attempts = int(tok.split("=")[1])
                            except ValueError: pass
                        elif tok.startswith("elapsed_s="):
                            try: elapsed_s = float(tok.split("=")[1])
                            except ValueError: pass
            machine_attempts += attempts
            gpu_status.append((fname, status, hits, attempts, elapsed_s))

        statuses = set(s for _, s, _, _, _ in gpu_status)
        status_str = ",".join(sorted(statuses))
        print(f"  inst {m.instance_id}: {len(gpu_status)} GPUs, "
              f"status={status_str}, hits={machine_total_hits}, "
              f"attempts={machine_attempts/1e9:.2f}G")
        # Show per-GPU detail only for hits or non-INCOMPLETE statuses
        for fname, st, h, a, e in gpu_status:
            if h > 0 or st not in ("INCOMPLETE",):
                print(f"      {fname}: status={st} hits={h} attempts={a/1e6:.0f}M elapsed={e:.0f}s")
        total_hits_across_fleet += machine_total_hits
        machines_status.append((m.instance_id, status_str, machine_total_hits, machine_attempts))

    print()
    print("═" * 60)
    print(f"AGGREGATE")
    print("═" * 60)
    print(f"  Total HIT lines across fleet: {total_hits_across_fleet}")
    print()
    if total_hits_across_fleet == 0:
        # Distinguish "everyone exhausted with 0 hits" from "some never finished"
        all_exhausted = all("EXHAUSTED" in s for _, s, _, _ in machines_status if s != "NO_SUMMARY")
        any_incomplete = any("INCOMPLETE" in s for _, s, _, _ in machines_status)
        any_killed = any("KILLED" in s for _, s, _, _ in machines_status)
        if all_exhausted and not any_incomplete and not any_killed:
            print(f"  Verdict: 0 hits but ALL GPUs marked EXHAUSTED → genuine null result")
            print(f"  This (seq, lt) gives 0 strict hits over full C(150,9).")
            print(f"  Try a different pin (--pin-seq-start past current first pin).")
        elif any_incomplete and not any_killed:
            # All running normally, no kills/missing. Just still searching.
            incomplete_count = sum(1 for _, s, _, _ in machines_status
                                    if "INCOMPLETE" in s)
            print(f"  Verdict: STILL SEARCHING — {incomplete_count} GPUs running, no hits yet.")
            print(f"  This is normal mid-search. Run check-results again later.")
        elif any_killed:
            print(f"  Verdict: AMBIGUOUS — some GPUs were KILLED.")
            print(f"  Cannot conclude 'no hits in space'. Consider relaunching.")
        else:
            print(f"  Verdict: see per-GPU details above.")
    else:
        print(f"  Verdict: {total_hits_across_fleet} hits found! Pull hit files:")
        print(f"    grep '^HIT ' {out_dir}/*/digest_summary_gpu*.txt")


def cmd_summary(args):
    """Read previously-fetched summary files from fleet_summaries/ and
    present a digest. NO SSH — works entirely on local files saved by the
    last check-results run. Use this for quick repeated checks without
    waiting on network.

    Output:
      - HIT lines (the most important — shown prominently, with fields parsed)
      - Per-machine status summary
      - Aggregate progress (% of C(n,t) covered, throughput, ETA)
      - Verdict
    """
    summaries_dir = Path("fleet_summaries")
    if not summaries_dir.exists():
        print(f"No fleet_summaries/ directory found.")
        print(f"Run `check-results` first to fetch summaries from remote.")
        return

    # Get current fleet's instance IDs to filter out stale data from
    # previous fleets that may still be in fleet_summaries/.
    current_fleet_ids = set()
    try:
        machines = load_fleet()
        current_fleet_ids = {str(m.instance_id) for m in machines}
    except Exception:
        pass  # If fleet.json missing/broken, fall back to showing everything

    # Collect all summary files
    all_summaries = sorted(summaries_dir.glob("inst_*/digest_summary_gpu*.txt"))
    if not all_summaries:
        print(f"No digest_summary_gpu*.txt files in fleet_summaries/.")
        print(f"Run `check-results` to fetch.")
        return

    # Filter: only include instances in current fleet
    if current_fleet_ids:
        summaries = []
        stale_inst_ids = set()
        for sf in all_summaries:
            inst_id = sf.parent.name.split("_", 1)[1]
            if inst_id in current_fleet_ids:
                summaries.append(sf)
            else:
                stale_inst_ids.add(inst_id)
        if stale_inst_ids:
            print(f"  (ignoring {len(stale_inst_ids)} stale inst dirs from previous fleets: "
                  f"{', '.join(sorted(stale_inst_ids)[:5])}"
                  f"{' …' if len(stale_inst_ids) > 5 else ''})")
            print(f"  (delete fleet_summaries/inst_<id>/ to suppress this notice)")
            print()
    else:
        summaries = all_summaries

    if not summaries:
        print(f"No summaries match the current fleet.")
        print(f"Run `check-results` to fetch fresh data.")
        return

    # Get fleet age — the freshest STARTED epoch as a proxy
    import time
    now = time.time()

    # Per-file parse
    @dataclass
    class GPUSummary:
        inst: int
        gpu_file: str
        seq: str = ""
        lt: int = 0
        calibrate: int = 0
        started_epoch: int = 0
        last_progress_epoch: int = 0
        attempts: int = 0
        slice_total: int = 0   # only known after STATUS=
        rate_M: float = 0.0
        elapsed_s: float = 0.0
        pct: float = 0.0
        status: str = "INCOMPLETE"
        hits: list = None     # list of HIT line dicts

    def parse_hit_line(ln):
        """Parse a HIT line into a dict. Format:
          HIT <epoch> combo=... hash_choice=N recid=N sighash=hex pubhash=hex combo_idx=N calibrate=N
        """
        h = {"raw": ln}
        toks = ln.split()
        if len(toks) >= 2 and toks[0] == "HIT":
            try: h["epoch"] = int(toks[1])
            except ValueError: pass
        for tok in toks[2:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                h[k] = v
        return h

    parsed = []
    for sf in summaries:
        inst_id = int(sf.parent.name.split("_")[1])
        gs = GPUSummary(inst=inst_id, gpu_file=sf.name, hits=[])
        for ln in sf.read_text().splitlines():
            if ln.startswith("STARTED "):
                # STARTED <epoch> gpu=N seq=0xHEX lt=N calibrate=N easy=N single_hash=N
                toks = ln.split()
                try: gs.started_epoch = int(toks[1])
                except (ValueError, IndexError): pass
                for tok in toks:
                    if tok.startswith("seq="): gs.seq = tok.split("=")[1]
                    elif tok.startswith("lt="):
                        try: gs.lt = int(tok.split("=")[1])
                        except ValueError: pass
                    elif tok.startswith("calibrate="):
                        try: gs.calibrate = int(tok.split("=")[1])
                        except ValueError: pass
            elif ln.startswith("PROGRESS "):
                toks = ln.split()
                try: gs.last_progress_epoch = int(toks[1])
                except (ValueError, IndexError): pass
                for tok in toks:
                    if tok.startswith("attempts="):
                        try: gs.attempts = int(tok.split("=")[1])
                        except ValueError: pass
                    elif tok.startswith("pct="):
                        try: gs.pct = float(tok.split("=")[1])
                        except ValueError: pass
                    elif tok.startswith("rate_M_per_s="):
                        try: gs.rate_M = float(tok.split("=")[1])
                        except ValueError: pass
                    elif tok.startswith("elapsed_s="):
                        try: gs.elapsed_s = float(tok.split("=")[1])
                        except ValueError: pass
            elif ln.startswith("HIT "):
                gs.hits.append(parse_hit_line(ln))
            elif ln.startswith("STATUS="):
                gs.status = ln.split()[0].split("=", 1)[1]
                for tok in ln.split():
                    if tok.startswith("total_attempts="):
                        try: gs.attempts = int(tok.split("=")[1])
                        except ValueError: pass
                    elif tok.startswith("slice_total="):
                        try: gs.slice_total = int(tok.split("=")[1])
                        except ValueError: pass
                    elif tok.startswith("elapsed_s="):
                        try: gs.elapsed_s = float(tok.split("=")[1])
                        except ValueError: pass
        parsed.append(gs)

    # ── Print summary ──
    print()
    print("═" * 64)
    print(f"  RESULTS SUMMARY — {len(parsed)} GPU summary file(s) "
          f"across {len(set(g.inst for g in parsed))} machines")
    print("═" * 64)

    # File freshness
    if summaries:
        newest = max(s.stat().st_mtime for s in summaries)
        oldest = min(s.stat().st_mtime for s in summaries)
        age = int(now - newest)
        print(f"  Local files updated: {age}s ago ({len(summaries)} files)")
        if age > 600:
            print(f"  ⚠ Older than 10 min — consider running `check-results` to refresh")

    # ── HITS section (prominent, shown first) ──
    all_hits = []
    for g in parsed:
        for h in g.hits:
            all_hits.append((g, h))

    print()
    if not all_hits:
        print(f"  HITS: none recorded yet")
    else:
        print(f"  ╔═══ {len(all_hits)} HIT(S) FOUND ═══╗")
        for i, (g, h) in enumerate(all_hits, 1):
            print(f"     [{i}] inst {g.inst} / {g.gpu_file}")
            for k in ("combo", "hash_choice", "recid", "sighash", "pubhash",
                      "combo_idx", "calibrate"):
                if k in h:
                    v = h[k]
                    label = k + ":"
                    print(f"          {label:14}{v}")
            if "epoch" in h:
                ago = int(now - h["epoch"])
                print(f"          {'found:':14}{ago}s ago")
        print(f"  ╚═══ end HITS ═══╝")

    # ── Status breakdown ──
    by_status = {}
    for g in parsed:
        by_status.setdefault(g.status, []).append(g)

    print()
    print(f"  STATUS BREAKDOWN:")
    for st in sorted(by_status):
        gpus = by_status[st]
        gpus_count = len(gpus)
        machines = len(set(g.inst for g in gpus))
        total_attempts = sum(g.attempts for g in gpus)
        marker = "✓" if st == "FOUND" else (
                 "✗" if st in ("KILLED",) else
                 "·")
        print(f"    {marker} {st:11}: {gpus_count:>3} GPUs across "
              f"{machines:>2} machines, {total_attempts/1e9:>7.1f}G attempts")

    # ── Throughput / ETA estimate (only if INCOMPLETE GPUs exist) ──
    incomplete = by_status.get("INCOMPLETE", [])
    if incomplete:
        # Sum throughput across all running GPUs
        # Use most recent rate_M per GPU.
        live_gpus = [g for g in incomplete if g.rate_M > 0]
        if live_gpus:
            total_rate_Mps = sum(g.rate_M for g in live_gpus)
            # Total work remaining: estimate by remaining percentage
            # Use mean pct as a rough estimate of how far along we are.
            mean_pct = sum(g.pct for g in live_gpus) / len(live_gpus)
            remaining_frac = max(0.0, 100.0 - mean_pct) / 100.0
            # Use sum of slice_totals if known, else estimate from C(150,9)
            import math
            n_pool = 150
            t_sel = 9  # could be 9 for both rounds in Config A
            full_space = math.comb(n_pool, t_sel)
            attempts_remaining = full_space * remaining_frac
            sec_remaining = attempts_remaining / (total_rate_Mps * 1e6)
            hr = int(sec_remaining // 3600)
            mn = int((sec_remaining % 3600) // 60)
            elapsed_med = sorted(g.elapsed_s for g in live_gpus)[len(live_gpus)//2]
            print()
            print(f"  PROGRESS:")
            print(f"    fleet throughput: {total_rate_Mps/1000:.2f} GH/s")
            print(f"    mean GPU progress: {mean_pct:.2f}%   (median elapsed: {elapsed_med:.0f}s)")
            print(f"    ETA to full exhaust: {hr}h{mn:02d}m")

    # ── Hits in the last 5 minutes (helpful for fast detection) ──
    recent_hits = [(g, h) for g, h in all_hits
                    if "epoch" in h and now - h["epoch"] < 300]
    if recent_hits:
        print()
        print(f"  ⚡ {len(recent_hits)} hit(s) in the last 5 minutes!")

    # ── Final verdict ──
    print()
    print("─" * 64)
    if all_hits:
        print(f"  ✅ HIT FOUND — verify with verify_digest_against_kernel.py")
        # Pick the first hit with calibrate=0
        strict_hits = [(g, h) for g, h in all_hits
                       if h.get("calibrate", "0") == "0"]
        if strict_hits:
            g, h = strict_hits[0]
            print(f"     Run:")
            print(f"       python3 .../verify_digest_against_kernel.py \\")
            print(f"           --round {'2' if 'r2' in g.gpu_file else '?'} \\")
            print(f"           --indices {h.get('combo','?')} \\")
            print(f"           --sequence {g.seq} \\")
            print(f"           --locktime {g.lt} \\")
            print(f"           --gpu-sighash {h.get('sighash','?')} \\")
            print(f"           --gpu-pubhash {h.get('pubhash','?')}")
    else:
        if "INCOMPLETE" in by_status:
            print(f"  ⏳ Still searching ({len(by_status['INCOMPLETE'])} GPUs running)")
        elif "EXHAUSTED" in by_status and len(by_status.get("EXHAUSTED", [])) == len(parsed):
            print(f"  ✗ Genuine null result — full coverage searched, 0 hits.")
            print(f"     Try a different pin (re-run with --pin-seq-start past current).")
        elif "KILLED" in by_status:
            print(f"  ⚠ Some GPUs killed — coverage incomplete. Consider rerun.")
        else:
            print(f"  See breakdown above.")
    print("─" * 64)




def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_launch = sub.add_parser("launch", help="rent N instances and compile kernels (does NOT start search — run `search --reuse` after)")
    p_launch.add_argument("--count", type=int, required=True,
                          help="number of instances to rent")
    p_launch.add_argument("--gpu", default="RTX_5090",
                          help="GPU model (default: RTX_5090)")
    p_launch.add_argument("--max-dph", type=float, default=2.50,
                          help="max price per MACHINE per hour in USD (default: 2.50). "
                               "For multi-GPU offers this is the TOTAL price, so set higher.")
    p_launch.add_argument("--min-gpus", type=int, default=1,
                          help="minimum num_gpus per machine (default: 1). "
                               "Set higher (e.g. 8) to prefer multi-GPU machines.")
    p_launch.add_argument("--prefer-multi-gpu", action="store_true",
                          help="when picking from filtered offers, prefer "
                               "machines with MORE GPUs (16 > 14 > … > 1) "
                               "over the cheapest $/GPU/hr. Reduces the "
                               "number of machines you have to manage.")
    p_launch.add_argument("--min-reliability", type=float, default=0.98)
    p_launch.add_argument("--datacenter-only", action="store_true",
                          help="only rent from datacenter-verified hosts")
    p_launch.add_argument("--disk", type=int, default=20,
                          help="disk size in GB (default: 20)")
    p_launch.add_argument("--image", default=DEFAULT_IMAGE,
                          help="docker image (default: nvidia/cuda CUDA devel)")
    p_launch.add_argument("--gpu-dir", default="../gpu",
                          help="path to directory with kernel .cu sources")
    p_launch.add_argument("--seq-base", type=lambda x: int(x, 0), default=0xfffffffe,
                          help="base sequence value (default: 0xfffffffe)")
    p_launch.add_argument("--lt-start", type=int, default=500_000_000,
                          help="starting locktime (default: 500000000)")
    p_launch.add_argument("--lt-total", type=int, default=2_147_483_647,
                          help="total locktime range across all machines (default: 2^31-1, full range)")
    p_launch.add_argument("--ssh-timeout", type=float, default=600.0)
    p_launch.add_argument("-y", "--yes", action="store_true",
                          help="skip confirmation prompt")
    p_launch.set_defaults(func=cmd_launch)

    p_status = sub.add_parser("status", help="show fleet status + recent progress")
    p_status.set_defaults(func=cmd_status)

    p_refresh = sub.add_parser("refresh-ssh",
        help="re-query vast.ai for each instance's current ssh_host/port and "
             "update qsb_fleet.json. Use this when SSH suddenly fails for "
             "the whole fleet.")
    p_refresh.set_defaults(func=cmd_refresh_ssh)

    p_hits = sub.add_parser("hits", help="pull hit logs from any instance that has them")
    p_hits.add_argument("--out", default="./hits", help="local directory for downloads")
    p_hits.set_defaults(func=cmd_hits)

    p_stop = sub.add_parser("stop",
        help="STOP (pause) all instances — SSH offline, storage preserved, "
             "billing reduced. Use `start` to resume.")
    p_stop.set_defaults(func=cmd_stop)

    p_start = sub.add_parser("start",
        help="START (resume) all paused instances and wait for SSH")
    p_start.set_defaults(func=cmd_start)

    p_kill = sub.add_parser("kill-kernels",
        help="kill running qsb_real/qsb_digest processes WITHOUT pausing "
             "instances (use to switch from one search to another)")
    p_kill.set_defaults(func=cmd_kill_kernels)

    p_destroy = sub.add_parser("destroy", help="destroy fleet instances (all by default, or specific ones with --ids)")
    p_destroy.add_argument("-y", "--yes", action="store_true")
    p_destroy.add_argument("--ids", default=None,
                           help="comma-separated instance IDs to destroy. "
                                "If omitted, destroys ALL fleet instances.")
    p_destroy.set_defaults(func=cmd_destroy)

    p_repro = sub.add_parser(
        "reprovision",
        help="re-run bootstrap on broken instances")
    g_repro = p_repro.add_mutually_exclusive_group(required=True)
    g_repro.add_argument("--ids", help="comma-separated instance IDs to reprovision")
    g_repro.add_argument("--auto", action="store_true",
                         help="auto-detect broken instances by probing each one")
    g_repro.add_argument("--all", action="store_true",
                         help="reprovision every instance in the fleet (nuclear)")
    p_repro.add_argument("--gpu-dir", default=None,
                         help="path to gpu/ dir (only needed if no bundle in cwd)")
    p_repro.add_argument("--ssh-timeout", type=float, default=300.0)
    p_repro.add_argument("-y", "--yes", action="store_true")
    p_repro.set_defaults(func=cmd_reprovision)

    p_smoke = sub.add_parser(
        "smoke",
        help="rent 1 cheap GPU, verify kernels compile and run, tear down")
    p_smoke.add_argument("--gpu", default="RTX_3060",
                         help="GPU model for smoke test (default: RTX_3060, cheapest usable)")
    p_smoke.add_argument("--max-dph", type=float, default=0.20,
                         help="max price per GPU-hour (default: 0.20)")
    p_smoke.add_argument("--image", default=DEFAULT_IMAGE)
    p_smoke.add_argument("--disk", type=int, default=20)
    p_smoke.add_argument("--gpu-dir", default="../gpu")
    p_smoke.add_argument("--ssh-timeout", type=float, default=600.0)
    p_smoke.add_argument("-y", "--yes", action="store_true")
    p_smoke.set_defaults(func=cmd_smoke)

    p_prefetch = sub.add_parser(
        "prefetch-debs",
        help="download libssl-dev .deb files locally for offline bootstrap fallback")
    p_prefetch.add_argument("--dest", default="deb_fallback",
                            help="local directory to download .deb files into "
                                 "(default: ./deb_fallback)")
    p_prefetch.set_defaults(func=cmd_prefetch_debs)

    p_search = sub.add_parser(
        "search",
        help="end-to-end coordinated search: pin → digest pivot, all machines together")
    p_search.add_argument("--reuse", action="store_true",
                          help="reuse the existing fleet instead of provisioning new")
    p_search.add_argument("--count", type=int, default=10,
                          help="number of MACHINES to rent (legacy; prefer --target-gpus)")
    p_search.add_argument("--target-gpus", type=int, default=None,
                          help="target total GPUs across the fleet. Picks offers "
                               "(any size) sorted by $/GPU until target reached. "
                               "Overrides --count when set.")
    p_search.add_argument("--over-provision-pct", type=float, default=0.0,
                          help="over-rent by this percentage to absorb provisioning "
                               "failures (e.g., 50 = rent 1.5x of --target-gpus)")
    p_search.add_argument("--gpu", default="RTX_5090,RTX_4090,RTX_6000Ada",
                          help="GPU model(s); comma-separated for multiple "
                               "(default: RTX_5090,RTX_4090,RTX_6000Ada)")
    p_search.add_argument("--min-gpus", type=int, default=8,
                          help="minimum num_gpus per machine (default: 8)")
    p_search.add_argument("--prefer-multi-gpu", action="store_true",
                          help="prefer machines with MORE GPUs (16 > 14 > … > 1) "
                               "over the cheapest $/GPU/hr. Reduces machines to manage.")
    p_search.add_argument("--max-dph", type=float, default=12.00,
                          help="max $/hour per MACHINE (default: 12.00 — covers 16-GPU machines)")
    p_search.add_argument("--min-reliability", type=float, default=0.98)
    p_search.add_argument("--datacenter-only", action="store_true")
    p_search.add_argument("--disk", type=int, default=20)
    p_search.add_argument("--image", default=DEFAULT_IMAGE)
    p_search.add_argument("--gpu-dir", default="../gpu")
    p_search.add_argument("--poll-interval", type=float, default=30.0,
                          help="seconds between hit-poll cycles (default: 30)")
    p_search.add_argument("--pin-timeout", type=float, default=4 * 3600.0,
                          help="abort pin stage after this many seconds (default: 4h)")
    p_search.add_argument("--digest-timeout", type=float, default=4 * 3600.0,
                          help="abort each digest stage after this many seconds (default: 4h)")
    # Skip pin stage if user already has a pinning result from a previous run
    p_search.add_argument("--use-pin-locktime", type=int, default=None,
                          help="skip pin stage; use this locktime from a previous pin hit")
    p_search.add_argument("--use-pin-sequence", type=str, default="0xfffffffe",
                          help="sequence (hex or dec) paired with --use-pin-locktime")
    p_search.add_argument("--use-pin-hash-choice", type=int, default=0,
                          help="hash_choice from the previous pin hit (default: 0)")
    p_search.add_argument("--use-pin-recid", type=int, default=0,
                          help="recid from the previous pin hit (default: 0)")
    p_search.add_argument("--funding-txid", type=str, default=None,
                          help="funding tx id (hex). Required for CPU pin "
                               "verification. If omitted, read from "
                               "regtest_funding.json in cwd.")
    p_search.add_argument("--funding-vout", type=int, default=0,
                          help="funding tx vout (default: 0)")
    p_search.add_argument("--skip-pin-verify", action="store_true",
                          help="DANGEROUS: skip CPU verification of pin hits. "
                               "Only use if you've verified manually. Without "
                               "this gate, a buggy GPU hit can waste hours of "
                               "fleet compute on impossible R1 search.")
    p_search.add_argument("--skip-digest-verify", action="store_true",
                          help="DANGEROUS: skip CPU verification of R1/R2 "
                               "digest hits. Without this gate, a kernel false "
                               "positive (e.g. wrong sighash layout) can lead "
                               "to assembling a tx that fails on broadcast.")
    p_search.add_argument("--start-round", type=int, default=1, choices=[1, 2],
                          help="which digest round to start at (default: 1 = "
                               "do both R1 and R2). Use 2 when R1 is already "
                               "solved and you want to skip straight to R2; "
                               "requires --use-pin-locktime / --use-pin-sequence.")
    p_search.add_argument("--pin-seq-start", type=str, default=None,
                          help="hex seq value to start pin search from (default: 0x80000000); "
                               "use to skip past a pin that gave zero digest hits")
    p_search.add_argument("--no-pipeline", action="store_true",
                          help="when --use-pin-locktime is supplied, force sequential mode "
                               "(bootstrap all → re-partition over survivors → start digest). "
                               "Slower wall time but no coverage gaps from bootstrap failures.")
    p_search.add_argument("-y", "--yes", action="store_true")
    p_search.set_defaults(func=cmd_search)

    p_diag = sub.add_parser(
        "diag",
        help="pull diagnostic logs from every fleet machine into fleet_diag/")
    p_diag.set_defaults(func=cmd_diag)

    p_debug_pin = sub.add_parser(
        "debug-pin",
        help="run kernel's single-point debug dump for (seq, lt) on remote, "
             "compare with CPU computation, print first divergence")
    p_debug_pin.add_argument("seq", help="sequence in hex (0x...) or decimal")
    p_debug_pin.add_argument("lt", type=int, help="locktime")
    p_debug_pin.add_argument("--funding-txid", default=None,
                              help="funding tx id (if not in regtest_funding.json)")
    p_debug_pin.add_argument("--funding-vout", type=int, default=0)
    p_debug_pin.set_defaults(func=cmd_debug_pin)

    p_debug_digest = sub.add_parser(
        "debug-digest",
        help="run kernel's single-subset digest debug dump on remote, "
             "compare with CPU computation, print first divergence")
    p_debug_digest.add_argument("round", type=int, help="round (1 or 2)")
    p_debug_digest.add_argument("subset", help="comma-separated subset indices")
    p_debug_digest.add_argument("seq", help="sequence (hex or decimal)")
    p_debug_digest.add_argument("lt", type=int, help="locktime")
    p_debug_digest.set_defaults(func=cmd_debug_digest)

    p_calibrate = sub.add_parser(
        "calibrate",
        help="run digest in CALIBRATE mode (relaxed DER) for X minutes; "
             "expected to find many hits if kernel works correctly")
    p_calibrate.add_argument("seq", help="sequence (hex or decimal)")
    p_calibrate.add_argument("lt", type=int, help="locktime")
    p_calibrate.add_argument("--round", type=int, default=1)
    p_calibrate.add_argument("--duration-minutes", type=float, default=15.0,
                              help="how long to run (default: 15 minutes)")
    p_calibrate.set_defaults(func=cmd_calibrate)

    p_check_results = sub.add_parser(
        "check-results",
        help="pull per-GPU summary files from every machine and report "
             "unambiguous hit/exhaust/killed status. Safe to run anytime, "
             "doesn't depend on the launcher having been alive.")
    p_check_results.set_defaults(func=cmd_check_results)

    p_summary = sub.add_parser(
        "summary",
        help="quick local summary of hits + status from previously-fetched "
             "fleet_summaries/ — no SSH, instant. Run after check-results.")
    p_summary.set_defaults(func=cmd_summary)

    p_resume = sub.add_parser(
        "resume-search",
        help="start search on machines that ALREADY have binaries built — "
             "use this when the launcher is hanging on dead machines")
    p_resume.add_argument("--pin-seq-start", type=str, default=None,
                          help="hex seq value to start pin search from")
    p_resume.add_argument("--use-pin-locktime", type=int, default=None)
    p_resume.add_argument("--use-pin-sequence", type=str, default="0xfffffffe")
    p_resume.add_argument("--use-pin-hash-choice", type=int, default=0)
    p_resume.add_argument("--use-pin-recid", type=int, default=0)
    p_resume.add_argument("-y", "--yes", action="store_true")
    p_resume.set_defaults(func=cmd_resume_search)

    p_restart_digest = sub.add_parser(
        "restart-digest",
        help="kick off digest R1/R2 on existing fleet using a known pin "
             "(seq, lt). Skips bootstrap, does proper LPT tile partitioning, "
             "exits after launch — use check-results to monitor")
    p_restart_digest.add_argument("seq", help="pin sequence (hex e.g. 0x8000dfd2)")
    p_restart_digest.add_argument("lt", type=int, help="pin locktime")
    p_restart_digest.add_argument("--round", type=int, default=1, choices=[1, 2])
    p_restart_digest.set_defaults(func=cmd_restart_digest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
