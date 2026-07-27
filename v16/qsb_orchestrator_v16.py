#!/usr/bin/env python3
"""
qsb_orchestrator.py — Rent and coordinate multiple vast.ai machines for QSB search.

Workflow:
1. Query vast.ai for available GPU rigs matching criteria
2. Rank by $/TFLOPS (cost efficiency)
3. Rent N machines until target GPU count reached
4. Upload qsb_v8_curvecheck.zip to each
5. Compile + run with proper total_gpus + global_offset
6. Poll for results files
7. Collect pinning hit, propagate to all machines for digest
8. Collect digest round 1 + round 2 hits
9. Terminate all machines

Prerequisites:
  pip install vastai requests paramiko
  export VAST_API_KEY=<your key>   # or set in ~/.vast_api_key
  rsync, ssh, scp available locally

Usage:
  python3 qsb_orchestrator.py rent
      Rent machines, upload, and start.
  python3 qsb_orchestrator.py status
      Check running machines.
  python3 qsb_orchestrator.py collect
      Poll for results and collect hits.
  python3 qsb_orchestrator.py stop
      Terminate all rented machines.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

# ============================================================
# Configuration
# ============================================================

TARGET_GPUS = 80                # total GPUs across all machines
MAX_MACHINES = 10               # hard limit on machine count
MIN_GPUS_PER_MACHINE = 8        # only rent machines with >= this many GPUs
ACCEPTABLE_GPU_NAMES = ["RTX 5090", "RTX 4090"]  # in order of preference
ZIP_FILE = "qsb_v16.zip"
MAX_DOLLARS_PER_GPU_HR = 0.50   # skip anything more expensive
SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")  # for uploads / control
STATE_FILE = Path(".qsb_orchestrator_state.json")
FUNDING_TXID = "4fab76e9b0538a49a77443030f8e0243a5d2558155647a839acea0efaa4edc91"

# ============================================================
# Data structures
# ============================================================

@dataclass
class Rental:
    instance_id: int
    ssh_host: str
    ssh_port: int
    num_gpus: int
    gpu_name: str
    dollars_per_hr: float
    global_offset: int      # first GPU index in the global numbering
    started: float
    status: str = "pending"  # pending, running, pinning_done, r1_done, r2_done, terminated

# ============================================================
# Vast.ai CLI wrapper
# ============================================================

def vast(*args, quiet: bool = False) -> str:
    """Run `vastai` CLI and return stdout.
    
    If quiet=True, don't print the traceback on failure — just raise.
    Useful for polling calls that may hit destroyed instances.
    """
    cmd = ["vastai"] + list(args)
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out
    except subprocess.CalledProcessError as e:
        if not quiet:
            print(f"vast command failed: {' '.join(cmd)}")
            print(e.output[:500])
        raise


def search_offers(min_gpus: int, gpu_names: List[str], max_dollars_per_gpu_hr: float = None) -> list:
    """Return list of vast.ai offers matching our criteria.
    
    We pull raw JSON via `vastai search offers --raw` and filter locally.
    """
    if max_dollars_per_gpu_hr is None:
        max_dollars_per_gpu_hr = MAX_DOLLARS_PER_GPU_HR
    # Build the query: rentable, on-demand, enough GPUs, reliable
    query_parts = [
        f"num_gpus>={min_gpus}",
        "rentable=true",
        "rented=false",
        "reliability>0.95",
        "verified=true",
        "inet_down>=100",   # at least 100Mbit/s
    ]
    query = " ".join(query_parts)
    raw = vast("search", "offers", query, "--raw", "-o", "dph_total")
    try:
        offers = json.loads(raw)
    except json.JSONDecodeError:
        print("Failed to parse vast output. Raw:")
        print(raw[:500])
        return []

    # Filter by GPU name
    filtered = []
    for o in offers:
        gpu = o.get("gpu_name", "")
        if not any(name in gpu for name in gpu_names):
            continue
        dph = float(o.get("dph_total", 999))
        ngpu = int(o.get("num_gpus", 0))
        if ngpu < min_gpus:
            continue
        if dph / ngpu > max_dollars_per_gpu_hr:
            continue
        filtered.append({
            "id": o["id"],
            "gpu_name": gpu,
            "num_gpus": ngpu,
            "dph_total": dph,
            "dph_per_gpu": dph / ngpu,
            "reliability": float(o.get("reliability2", 0)),
            "inet_down": float(o.get("inet_down", 0)),
            "machine_id": o.get("machine_id"),
            "host_id": o.get("host_id"),
        })

    # Sort by $/GPU/hr ascending (best value first)
    filtered.sort(key=lambda x: x["dph_per_gpu"])
    return filtered


def create_instance(offer_id: int) -> int:
    """Rent a machine. Uses vast.ai's cuda:pytorch image which has SSH,
    build tools, and openssl preinstalled, plus onstart to make sure."""
    # This image has SSH, CUDA, build tools, and openssl already
    image = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel"
    out = vast(
        "create", "instance", str(offer_id),
        "--image", image,
        "--disk", "16",
        "--ssh",                 # ensure direct SSH (not jupyter proxy)
        "--raw",
    )
    try:
        data = json.loads(out)
        inst_id = int(data.get("new_contract") or data.get("id"))
        return inst_id
    except Exception:
        print("Failed to parse create output:", out)
        raise


# Make absolutely sure required tools are installed on the remote before we
# try to use them. Apt state varies by image; we don't rely on it being ready.
SETUP_SCRIPT = r"""
set -e
# Retry apt a few times — sometimes dpkg is locked just after boot
for i in 1 2 3 4 5; do
    if apt-get update -qq && apt-get install -y -qq libssl-dev unzip; then break; fi
    echo "apt attempt $i failed, retrying in 10s..."
    sleep 10
done
# Confirm nvcc exists (part of the CUDA image already)
which nvcc >/dev/null 2>&1 || { echo 'nvcc missing!' >&2; exit 1; }
which unzip >/dev/null 2>&1 || { echo 'unzip missing!' >&2; exit 1; }
echo "SETUP_OK"
"""


def get_instance(instance_id: int) -> Optional[dict]:
    """Fetch single instance info. Returns None if the instance is gone
    (manually destroyed, never existed, etc.)."""
    try:
        out = vast("show", "instance", str(instance_id), "--raw", quiet=True)
    except subprocess.CalledProcessError:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


def wait_for_ssh(instance_id: int, timeout: int = 900, verbose: bool = False) -> tuple:
    """Wait until the instance reports SSH info. Returns (host, port).
    Raises TimeoutError if the instance never comes up, or RuntimeError
    if the instance is gone (destroyed)."""
    deadline = time.time() + timeout
    last_status = None
    consecutive_missing = 0
    while time.time() < deadline:
        info = get_instance(instance_id)
        if info is None:
            consecutive_missing += 1
            if consecutive_missing >= 3:
                raise RuntimeError(f"Instance {instance_id} is gone (destroyed or never existed)")
            time.sleep(15)
            continue
        consecutive_missing = 0
        host = info.get("ssh_host")
        port = info.get("ssh_port")
        status = info.get("actual_status", "")
        # Detect error states — don't just wait forever
        if status in ("offline", "exited", "error"):
            raise RuntimeError(f"Instance {instance_id} in bad state: {status}")
        if host and port and status == "running":
            return host, int(port)
        if verbose and status != last_status:
            print(f"  [{instance_id}] status={status}")
            last_status = status
        time.sleep(20)
    raise TimeoutError(f"Instance {instance_id} never came up (timeout {timeout}s)")


def destroy_instance(instance_id: int):
    try:
        vast("destroy", "instance", str(instance_id))
    except Exception as e:
        print(f"  warn: could not destroy {instance_id}: {e}")

# ============================================================
# SSH / rsync helpers
# ============================================================

def ssh_cmd(rental: Rental, cmd: str, timeout: int = 60, retries: int = 3) -> str:
    """Run an SSH command. Retries on transient network errors (timeouts,
    connection refused, etc.) but NOT on command-level failures (rc != 0
    means the remote command ran but returned an error).
    """
    full = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",           # suppress "warning: added to known hosts"
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-i", SSH_KEY_PATH,
        "-p", str(rental.ssh_port),
        f"root@{rental.ssh_host}",
        cmd,
    ]

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            proc = subprocess.run(
                full,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                text=True,
            )
        except subprocess.TimeoutExpired as e:
            last_exc = e
            if attempt < retries:
                time.sleep(5 * attempt)
                continue
            raise RuntimeError(f"ssh timed out after {retries} attempts") from e

        if proc.returncode == 0:
            return proc.stdout

        # rc != 0. If it's a network-level error (rc 255 from ssh itself,
        # with a "connect"/"timed out"/"refused" message), retry.
        # Otherwise the remote command actually ran and failed — don't retry.
        network_err = (
            proc.returncode == 255
            and proc.stderr
            and any(
                s in proc.stderr
                for s in (
                    "Operation timed out",
                    "Connection timed out",
                    "Connection refused",
                    "Connection reset",
                    "Network is unreachable",
                    "No route to host",
                    "kex_exchange_identification",
                    "Connection closed by",
                )
            )
        )
        last_exc = RuntimeError(
            f"rc={proc.returncode} stderr={proc.stderr.strip()[:500]} stdout={proc.stdout.strip()[:200]}"
        )
        if network_err and attempt < retries:
            time.sleep(5 * attempt)
            continue

        short_cmd = cmd[:120] + ("..." if len(cmd) > 120 else "")
        raise RuntimeError(
            f"ssh cmd failed (rc={proc.returncode})\n"
            f"  cmd: {short_cmd}\n"
            f"  stderr: {proc.stderr.strip()[:500]}\n"
            f"  stdout: {proc.stdout.strip()[:200]}"
        )

    raise last_exc or RuntimeError("ssh_cmd: unreachable")


def ssh_bg(rental: Rental, cmd: str):
    """Launch a command on the remote and return immediately without waiting.

    We CANNOT reuse ssh_cmd() here because that pipes stdout/stderr and waits
    for ssh to exit. Even with nohup/&, ssh holds the channel open waiting
    for the remote's file descriptors to close.

    Solution: write the command to a remote script, then start it with ssh
    using -f (go to background after authentication) -n (null stdin).
    The ssh process itself forks away immediately.
    """
    # Write the launcher to a remote script file first (via a normal ssh_cmd,
    # which is fast and doesn't hang)
    remote_script = "/work/_launch.sh"
    launcher_body = f"""#!/bin/bash
cd /work
exec nohup setsid bash -c {json.dumps(cmd)} < /dev/null > /work/orchestrator.log 2>&1 &
disown
"""
    # base64-encode to avoid shell-quoting issues, then decode on the remote
    import base64
    b64 = base64.b64encode(launcher_body.encode()).decode()
    ssh_cmd(
        rental,
        f"echo {b64} | base64 -d > {remote_script} && chmod +x {remote_script}",
        timeout=30,
    )

    # Now launch it with ssh -f -n. This ssh call returns as soon as the
    # remote command has been launched (not when it finishes).
    full = [
        "ssh",
        "-f",                              # go to background after auth
        "-n",                              # redirect stdin from /dev/null
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=15",
        "-i", SSH_KEY_PATH,
        "-p", str(rental.ssh_port),
        f"root@{rental.ssh_host}",
        f"bash {remote_script}",
    ]
    proc = subprocess.run(
        full,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=30,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ssh_bg launch failed: rc={proc.returncode}, stderr: {proc.stderr.strip()[:500]}")


def scp_to(rental: Rental, local_path: str, remote_path: str):
    full = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-q",                      # quiet: no progress meter or banners
        "-i", SSH_KEY_PATH,
        "-P", str(rental.ssh_port),
        local_path,
        f"root@{rental.ssh_host}:{remote_path}",
    ]
    subprocess.run(full, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def scp_from(rental: Rental, remote_path: str, local_path: str) -> bool:
    full = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-q",
        "-i", SSH_KEY_PATH,
        "-P", str(rental.ssh_port),
        f"root@{rental.ssh_host}:{remote_path}",
        local_path,
    ]
    try:
        subprocess.run(full, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60, check=True)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False

# ============================================================
# State persistence
# ============================================================

def save_state(rentals: List[Rental], extra: dict = None):
    data = {"rentals": [asdict(r) for r in rentals]}
    if extra:
        data.update(extra)
    STATE_FILE.write_text(json.dumps(data, indent=2))


def load_state() -> tuple:
    if not STATE_FILE.exists():
        return [], {}
    data = json.loads(STATE_FILE.read_text())
    rentals = [Rental(**r) for r in data.get("rentals", [])]
    extra = {k: v for k, v in data.items() if k != "rentals"}
    return rentals, extra

# ============================================================
# Commands
# ============================================================

def cmd_rent():
    if not Path(ZIP_FILE).exists():
        print(f"ERROR: {ZIP_FILE} not found in current directory.")
        sys.exit(1)

    print(f"Searching vast.ai for offers with >= {MIN_GPUS_PER_MACHINE} GPUs...")
    offers = search_offers(MIN_GPUS_PER_MACHINE, ACCEPTABLE_GPU_NAMES)
    if not offers:
        print("No matching offers found!")
        return
    print(f"\nFound {len(offers)} offers. Top 15 by $/GPU/hr:")
    print(f"  {'ID':>10} {'GPU':<12} {'N':>3} {'$/hr':>8} {'$/GPU/hr':>10} {'Rel':>5}")
    for o in offers[:15]:
        print(f"  {o['id']:>10} {o['gpu_name']:<12} {o['num_gpus']:>3} "
              f"{o['dph_total']:>7.3f}$ {o['dph_per_gpu']:>9.3f}$ {o['reliability']:.2f}")

    # Greedy selection toward TARGET_GPUS
    selected = []
    total_gpus = 0
    used_hosts = set()
    for o in offers:
        if len(selected) >= MAX_MACHINES:
            break
        if total_gpus >= TARGET_GPUS:
            break
        if o["id"] in used_hosts:
            continue
        selected.append(o)
        used_hosts.add(o["id"])
        total_gpus += o["num_gpus"]

    print(f"\nSelected {len(selected)} machines totaling {total_gpus} GPUs:")
    total_cost = sum(o["dph_total"] for o in selected)
    for o in selected:
        print(f"  id={o['id']}  {o['gpu_name']} × {o['num_gpus']}  "
              f"${o['dph_total']:.3f}/hr  (${o['dph_per_gpu']:.3f}/GPU/hr)")
    print(f"\nTotal cost: ${total_cost:.3f}/hr")

    ans = input("\nProceed to rent? [y/N]: ")
    if ans.strip().lower() != "y":
        print("Aborted.")
        return

    # Design: each instance runs the FULL pipeline independently and
    # launches IMMEDIATELY once its own provisioning succeeds.
    # Uses the planned total (sum of num_gpus from `selected`) — if some
    # machines fail, their GPU slice is just not covered. That's fine for
    # probabilistic search (we don't need to exhaust).

    import hashlib
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    local_zip_bytes = Path(ZIP_FILE).read_bytes()
    local_sha = hashlib.sha256(local_zip_bytes).hexdigest()
    local_size = len(local_zip_bytes)
    print(f"\nLocal zip: {ZIP_FILE} ({local_size} bytes, sha256 {local_sha[:16]}...)")

    # Assign global_offset upfront based on the planned list (so each instance
    # knows its offset before its siblings finish provisioning).
    # If some machines fail, their slice is just left uncovered.
    planned_total = sum(o["num_gpus"] for o in selected)
    planned = []
    off = 0
    for o in selected:
        planned.append((o, off))
        off += o["num_gpus"]
    print(f"\nPlanned: {planned_total} GPUs across {len(selected)} machines")
    print(f"Launching each machine independently — no waiting between them.\n")

    lock = threading.Lock()
    rentals: List[Rental] = []

    def _full_pipeline(offer, my_offset, total_gpus):
        tag = f"[offer {offer['id']}]"
        inst_id = None
        try:
            # 1) create
            inst_id = create_instance(offer["id"])
            tag = f"[{inst_id}]"
            print(f"{tag} created from offer {offer['id']} ({offer['gpu_name']} × {offer['num_gpus']}, offset={my_offset})")

            # 2) wait for ssh (vast.ai reports running)
            host, port = wait_for_ssh(inst_id, timeout=900, verbose=False)
            print(f"{tag} ssh ready at {host}:{port}")

            r = Rental(
                instance_id=inst_id,
                ssh_host=host,
                ssh_port=port,
                num_gpus=offer["num_gpus"],
                gpu_name=offer["gpu_name"],
                dollars_per_hr=offer["dph_total"],
                global_offset=my_offset,
                started=time.time(),
            )

            # 2a) ping ssh with retries — sometimes the daemon is not quite ready
            last_err = None
            for attempt in range(1, 13):  # up to ~2 min of retries
                try:
                    ssh_cmd(r, "echo PING", timeout=20)
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(10)
            else:
                raise RuntimeError(f"ssh never became responsive: {last_err}")
            print(f"{tag} ssh responsive")

            # 2b) install required tools (idempotent, retries apt)
            setup_out = ssh_cmd(r, SETUP_SCRIPT, timeout=300)
            if "SETUP_OK" not in setup_out:
                raise RuntimeError(f"setup script didn't report SETUP_OK; got: {setup_out[-400:]}")
            print(f"{tag} tools installed")

            # 3) upload zip + verify
            ssh_cmd(r, "mkdir -p /work && cd /work && rm -rf ./*")
            scp_to(r, ZIP_FILE, f"/work/{ZIP_FILE}")
            info = ssh_cmd(r, f"cd /work && stat -c %s {ZIP_FILE} && sha256sum {ZIP_FILE}").strip().splitlines()
            remote_size = int(info[0])
            remote_sha = info[1].split()[0]
            if remote_size != local_size or remote_sha != local_sha:
                raise RuntimeError(f"zip upload mismatch (got {remote_size}/{remote_sha[:16]})")
            print(f"{tag} zip uploaded & verified")

            # 4) compile
            ssh_cmd(r, f"cd /work && unzip -o {ZIP_FILE} && "
                       "nvcc -O3 -o qsb_real qsb_real_search.cu -lcrypto -lm && "
                       "nvcc -O3 -o qsb_digest qsb_digest_search.cu -lcrypto -lm && "
                       "nvcc -O3 -o verify_gpu verify_gpu.cu -lcrypto -lm && "
                       "mkdir -p results logs",
                    timeout=900)
            check = ssh_cmd(r, "cd /work && ls qsb_real qsb_digest verify_gpu 2>&1").strip()
            if "cannot access" in check or "No such file" in check:
                raise RuntimeError(f"compile failed: {check}")
            print(f"{tag} compiled")

            # 5) launch IMMEDIATELY
            launch_cmd = f"bash run_all.sh {total_gpus} {my_offset}"
            ssh_bg(r, launch_cmd)
            r.status = "running"
            print(f"{tag} LAUNCHED ({launch_cmd})")

            with lock:
                rentals.append(r)
                save_state(rentals, {"total_gpus": total_gpus, "phase": "running"})
            return r, None

        except Exception as e:
            msg = str(e)[:1200]
            print(f"{tag} FAILED: {msg}")
            if inst_id is not None:
                try:
                    destroy_instance(inst_id)
                    print(f"{tag} destroyed")
                except Exception:
                    pass
            return None, msg

    with ThreadPoolExecutor(max_workers=len(planned)) as ex:
        futures = [ex.submit(_full_pipeline, o, off, planned_total) for o, off in planned]
        for fut in as_completed(futures):
            fut.result()

    running = sum(1 for r in rentals if r.status == "running")
    total_running_gpus = sum(r.num_gpus for r in rentals if r.status == "running")
    print(f"\n{running}/{len(planned)} machines running ({total_running_gpus}/{planned_total} GPUs).")
    print("Use `status` to monitor, `collect` to gather results.")


def cmd_status():
    rentals, extra = load_state()
    if not rentals:
        print("No rentals on file. Run `rent` first.")
        return
    mode = extra.get("mode", "normal")
    print(f"Mode: {mode}  |  Total GPUs: {extra.get('total_gpus', '?')}  |  Phase: {extra.get('phase', '?')}")
    
    total_cost = 0.0
    for r in rentals:
        age = (time.time() - r.started) / 60
        cost = r.dollars_per_hr * age / 60
        total_cost += cost
        try:
            # Comprehensive status query in ONE ssh round-trip
            query = (
                # Mode-relevant: which logs exist?
                "echo '--- LOGS ---'; "
                "ls /work/logs/ 2>/dev/null | head -5; "
                # R1 status: look for hits / latest progress from GPU 0
                "echo '--- R1 ---'; "
                "if [ -f /work/results/round1_final.txt ]; then "
                "  echo 'R1 DONE:'; cat /work/results/round1_final.txt; "
                "elif [ -f /work/logs/dig1_gpu_0.log ]; then "
                "  grep -E '^  \\[GPU 0\\]' /work/logs/dig1_gpu_0.log | tail -1 2>/dev/null || echo '(no progress yet)'; "
                "else "
                "  echo '(R1 not started — pre-uploaded?)'; "
                "fi; "
                # R2 status
                "echo '--- R2 ---'; "
                "if [ -f /work/results/round2_final.txt ]; then "
                "  echo 'R2 DONE:'; cat /work/results/round2_final.txt; "
                "elif [ -f /work/logs/dig2_gpu_0.log ]; then "
                "  grep -E '^  \\[GPU 0\\]' /work/logs/dig2_gpu_0.log | tail -1 2>/dev/null || echo '(no progress yet)'; "
                "else "
                "  echo '(R2 not started)'; "
                "fi; "
                # Aggregate progress: how many GPUs are running?
                "echo '--- PROCESSES ---'; "
                "echo \"qsb_digest running: $(pgrep -c -x qsb_digest 2>/dev/null || echo 0)\"; "
                "echo \"qsb_real    running: $(pgrep -c -x qsb_real 2>/dev/null || echo 0)\"; "
                # Average rate across GPUs (last reported speed)
                "echo '--- RATES ---'; "
                "for log in /work/logs/dig2_gpu_*.log /work/logs/dig1_gpu_*.log; do "
                "  [ -f \"$log\" ] || continue; "
                "  last=$(grep -E '\\[GPU' \"$log\" | tail -1); "
                "  if [ -n \"$last\" ]; then echo \"$(basename $log): $last\"; fi; "
                "done | head -3"
            )
            tail = ssh_cmd(r, query, timeout=30).strip()
        except Exception as e:
            tail = f"(ssh err: {e})"
        print(f"\n[{r.instance_id}] {r.gpu_name}x{r.num_gpus}  offset {r.global_offset}  "
              f"age {age:.1f}min  cost ${cost:.2f}")
        for line in tail.splitlines():
            print(f"    {line}")
    
    print(f"\nTotal cost so far: ${total_cost:.2f}")


def cmd_collect():
    """Poll every 30s for digest results from any machine.
    
    In SOLO mode: each machine has its OWN pinning, so R1+R2 must come from
    the SAME machine. We track per-machine progress and only assemble when
    one machine has BOTH R1 and R2 done.
    
    Writes to local results/<instance_id>/ for solo mode.
    """
    rentals, extra = load_state()
    if not rentals:
        print("No rentals on file.")
        return

    is_solo = extra.get("mode") == "solo"
    is_r2_only = extra.get("mode") == "r2_only"
    if is_r2_only:
        print("Mode: R2-ONLY (R1 was pre-uploaded, only waiting for R2 hit)")
    Path("results").mkdir(exist_ok=True)

    # For solo mode: {instance_id: {"pinning": ..., "r1": ..., "r2": ...}}
    per_machine = {r.instance_id: {"rental": r, "pinning": None, "r1": None, "r2": None, "r1_round": None, "r2_round": None} for r in rentals}

    def _fetch_file(r, remote, local):
        return scp_from(r, remote, local)

    def _check_remote(r, remote_path):
        """Return True if the file exists on the remote."""
        try:
            out = ssh_cmd(r, f"ls {remote_path} 2>/dev/null || true", timeout=15).strip()
            return bool(out)
        except Exception:
            return False

    winner = None

    while winner is None:
        for r in rentals:
            m = per_machine[r.instance_id]
            try:
                # 1) Pinning (we already know for solo, but fetch for completeness)
                if m["pinning"] is None:
                    inst_dir = Path("results") / str(r.instance_id)
                    inst_dir.mkdir(exist_ok=True)
                    out = ssh_cmd(r, "ls /work/results/pinning_hit_*.txt 2>/dev/null || true", timeout=15).strip()
                    if out:
                        first = out.splitlines()[0]
                        tmp = inst_dir / "pinning_hit.txt"
                        if _fetch_file(r, first, str(tmp)):
                            m["pinning"] = tmp.read_text()
                            if is_solo or is_r2_only:
                                print(f"[{r.instance_id}] pinning: "
                                      f"seq={_parse_kv(m['pinning']).get('sequence')} "
                                      f"lt={_parse_kv(m['pinning']).get('locktime')}")

                # 2) Round 1 final
                if m["r1"] is None and _check_remote(r, "/work/results/round1_final.txt"):
                    inst_dir = Path("results") / str(r.instance_id)
                    inst_dir.mkdir(exist_ok=True)
                    tmp = inst_dir / "round1_final.txt"
                    if _fetch_file(r, "/work/results/round1_final.txt", str(tmp)):
                        m["r1"] = tmp.read_text()
                        m["r1_round"] = "R1"
                        kv = _parse_kv(m["r1"])
                        print(f"\n{'='*60}")
                        print(f"*** ROUND 1 HIT on {r.instance_id} ***")
                        print(f"{'='*60}")
                        pin_kv = _parse_kv(m["pinning"]) if m["pinning"] else {}
                        print(f"  seq={pin_kv.get('sequence','?')}  lt={pin_kv.get('locktime','?')}")
                        print(f"  indices={kv.get('indices','?')}")
                        print(f"  hc={kv.get('hash_choice','?')}  recid={kv.get('recid','?')}")

                # 3) Round 2 final
                if m["r2"] is None and _check_remote(r, "/work/results/round2_final.txt"):
                    inst_dir = Path("results") / str(r.instance_id)
                    inst_dir.mkdir(exist_ok=True)
                    tmp = inst_dir / "round2_final.txt"
                    if _fetch_file(r, "/work/results/round2_final.txt", str(tmp)):
                        m["r2"] = tmp.read_text()
                        m["r2_round"] = "R2"
                        kv = _parse_kv(m["r2"])
                        print(f"\n{'='*60}")
                        print(f"*** ROUND 2 HIT on {r.instance_id} ***")
                        print(f"{'='*60}")
                        pin_kv = _parse_kv(m["pinning"]) if m["pinning"] else {}
                        print(f"  seq={pin_kv.get('sequence','?')}  lt={pin_kv.get('locktime','?')}")
                        print(f"  indices={kv.get('indices','?')}")
                        print(f"  hc={kv.get('hash_choice','?')}  recid={kv.get('recid','?')}")

                # Check if THIS machine has pinning + R1 + R2 all from itself
                if m["pinning"] and m["r1"] and m["r2"]:
                    winner = r.instance_id
                    break

            except Exception as e:
                print(f"[{r.instance_id}] poll error: {e}")

        if winner is not None:
            break

        # Progress summary
        ts = time.strftime('%H:%M:%S')
        parts = []
        for r in rentals:
            m = per_machine[r.instance_id]
            state = []
            state.append("P" if m["pinning"] else "-")
            state.append("1" if m["r1"] else "-")
            state.append("2" if m["r2"] else "-")
            parts.append(f"{r.instance_id}={''.join(state)}")
        print(f"[{ts}] " + "  ".join(parts))
        time.sleep(30)

    # Winner found
    m = per_machine[winner]
    pin_kv = _parse_kv(m["pinning"])
    r1_kv = _parse_kv(m["r1"])
    r2_kv = _parse_kv(m["r2"])
    print("\n" + "=" * 60)
    print(f"  ALL PHASES COMPLETE on instance {winner}")
    print("=" * 60)
    print(f"  Pinning: seq={pin_kv.get('sequence')} lt={pin_kv.get('locktime')}")
    print(f"  Round 1: {r1_kv.get('indices','?')}")
    print(f"  Round 2: {r2_kv.get('indices','?')}")
    print()
    print("Assembly command:\n")
    print(f"python3 qsb_pipeline.py assemble \\")
    print(f"  --funding-txid {FUNDING_TXID} \\")
    print(f"  --funding-vout 0 --funding-value 10000 --version 2 \\")
    print(f"  --locktime {pin_kv.get('locktime','?')} --sequence {pin_kv.get('sequence','?')} \\")
    print(f"  --round1 {r1_kv.get('indices','?')} --round2 {r2_kv.get('indices','?')}")


def _parse_kv(text: str) -> dict:
    out = {}
    for line in text.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def cmd_stop():
    rentals, _ = load_state()
    if not rentals:
        print("No rentals on file.")
        return
    for r in rentals:
        print(f"Destroying instance {r.instance_id}...")
        destroy_instance(r.instance_id)
    STATE_FILE.unlink(missing_ok=True)
    print("All instances destroyed, state file removed.")


def cmd_inspect():
    """Collect current results and logs from all machines WITHOUT changing anything.
    
    Downloads all /work/results/*.txt files and last lines of orchestrator.log
    from each machine into ./inspect/<instance_id>/.
    """
    rentals, _ = load_state()
    if not rentals:
        print("No rentals on file.")
        return

    out_root = Path("inspect")
    out_root.mkdir(exist_ok=True)

    print(f"Inspecting {len(rentals)} machines...\n")
    for r in rentals:
        tag = f"[{r.instance_id}]"
        mdir = out_root / str(r.instance_id)
        mdir.mkdir(exist_ok=True)
        print(f"{tag} {r.gpu_name} × {r.num_gpus}  offset={r.global_offset}")

        # 1) what processes are running?
        try:
            ps_out = ssh_cmd(r,
                "ps -eo pid,etime,cmd | grep -E 'qsb_(real|digest)' | grep -v grep || true",
                timeout=20).strip()
            (mdir / "processes.txt").write_text(ps_out + "\n")
            if ps_out:
                first_line = ps_out.splitlines()[0][:120]
                print(f"  proc: {first_line}")
            else:
                print(f"  proc: (nothing running)")
        except Exception as e:
            print(f"  proc: ERR {e}")
            (mdir / "processes.txt").write_text(f"error: {e}\n")

        # 2) which results files exist?
        try:
            ls_out = ssh_cmd(r,
                "ls -la /work/results/ 2>/dev/null || echo '(no results dir)'",
                timeout=20).strip()
            (mdir / "results_listing.txt").write_text(ls_out + "\n")
            # count .txt files
            num_txt = sum(1 for l in ls_out.splitlines() if l.endswith(".txt"))
            print(f"  results: {num_txt} .txt files")
        except Exception as e:
            print(f"  results: ERR {e}")
            (mdir / "results_listing.txt").write_text(f"error: {e}\n")

        # 3) pull all result .txt files
        try:
            names = ssh_cmd(r,
                "cd /work/results 2>/dev/null && ls *.txt 2>/dev/null || true",
                timeout=20).strip().splitlines()
            for name in names:
                name = name.strip()
                if not name:
                    continue
                local_path = mdir / name
                if scp_from(r, f"/work/results/{name}", str(local_path)):
                    content = local_path.read_text()
                    # print the interesting ones inline
                    if "pinning_hit" in name or "final" in name:
                        print(f"  === {name} ===")
                        for line in content.strip().splitlines()[:6]:
                            print(f"    {line}")
        except Exception as e:
            print(f"  pull results: ERR {e}")

        # 4) orchestrator log tail
        try:
            tail = ssh_cmd(r, "tail -60 /work/orchestrator.log 2>/dev/null || echo '(no log)'", timeout=20)
            (mdir / "orchestrator.log.tail").write_text(tail)
        except Exception as e:
            (mdir / "orchestrator.log.tail").write_text(f"error: {e}\n")

        # 5) individual GPU logs (last 20 lines each)
        try:
            log_list = ssh_cmd(r,
                "cd /work/logs 2>/dev/null && ls 2>/dev/null || true",
                timeout=20).strip().splitlines()
            for lname in log_list:
                lname = lname.strip()
                if not lname:
                    continue
                tail = ssh_cmd(r, f"tail -20 /work/logs/{lname} 2>/dev/null || true", timeout=20)
                (mdir / f"log_{lname}").write_text(tail)
        except Exception as e:
            print(f"  pull logs: ERR {e}")

        print()

    # Summary: scan for pinning hits across all machines
    print("=" * 60)
    print("SUMMARY — pinning hits found across all machines:")
    print("=" * 60)
    for mdir in sorted(out_root.iterdir()):
        if not mdir.is_dir():
            continue
        for f in sorted(mdir.glob("pinning_hit*.txt")):
            kv = _parse_kv(f.read_text())
            seq = kv.get("sequence", "?")
            lt = kv.get("locktime", "?")
            print(f"  {mdir.name}/{f.name}:  seq={seq}  lt={lt}")
    for mdir in sorted(out_root.iterdir()):
        if not mdir.is_dir():
            continue
        for f in sorted(mdir.glob("round1_final.txt")) + sorted(mdir.glob("round2_final.txt")):
            kv = _parse_kv(f.read_text())
            print(f"  {mdir.name}/{f.name}:  indices={kv.get('indices','?')[:60]}")
    print(f"\nAll data saved to {out_root}/")


# Embedded digest-only launcher script. Pushed to each machine during relaunch.
RUN_ALL_DIGEST_SH = r"""#!/bin/bash
set -e
SEQUENCE=$1
LOCKTIME=$2
TOTAL_GPUS=$3
GLOBAL_OFFSET=$4

NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "=== QSB Digest Pipeline ==="
echo "  Pinning: seq=$SEQUENCE lt=$LOCKTIME"
echo "  Local GPUs: $NUM_GPUS"
echo "  Total GPUs: $TOTAL_GPUS (offset $GLOBAL_OFFSET)"

mkdir -p results logs
rm -f results/digest_hit_*.txt results/round1_final.txt results/round2_final.txt
killall qsb_real qsb_digest 2>/dev/null || true
sleep 1

echo "=== Phase 1: Digest Round 1 ==="
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    stdbuf -oL ./qsb_digest digest_r1.bin $gpu $SEQUENCE $LOCKTIME $TOTAL_GPUS $GLOBAL_OFFSET > logs/dig1_gpu_${gpu}.log 2>&1 &
    sleep 1
done

while true; do
    sleep 15
    if compgen -G "results/digest_hit_*.txt" > /dev/null 2>&1; then
        echo "  *** DIGEST R1 HIT! ***"
        killall qsb_digest 2>/dev/null || true
        sleep 2
        cat results/digest_hit_*.txt
        break
    fi
    if ! pgrep -x qsb_digest > /dev/null; then
        echo "  (R1 local slice done, waiting for others...)"
        sleep 60
    fi
done

R1_FILE=$(ls results/digest_hit_*.txt | head -1)
R1_INDICES=$(grep "^indices=" "$R1_FILE" | head -1 | cut -d= -f2)
cp "$R1_FILE" results/round1_final.txt
echo "  Round 1: $R1_INDICES"

echo "=== Phase 2: Digest Round 2 ==="
rm -f results/digest_hit_*.txt
sleep 1

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    stdbuf -oL ./qsb_digest digest_r2.bin $gpu $SEQUENCE $LOCKTIME $TOTAL_GPUS $GLOBAL_OFFSET > logs/dig2_gpu_${gpu}.log 2>&1 &
    sleep 1
done

while true; do
    sleep 15
    if compgen -G "results/digest_hit_*.txt" > /dev/null 2>&1; then
        echo "  *** DIGEST R2 HIT! ***"
        killall qsb_digest 2>/dev/null || true
        sleep 2
        cat results/digest_hit_*.txt
        break
    fi
    if ! pgrep -x qsb_digest > /dev/null; then
        echo "  (R2 local slice done, waiting for others...)"
        sleep 60
    fi
done

R2_FILE=$(ls results/digest_hit_*.txt | head -1)
R2_INDICES=$(grep "^indices=" "$R2_FILE" | head -1 | cut -d= -f2)
cp "$R2_FILE" results/round2_final.txt
echo "  Round 2: $R2_INDICES"

echo "=============================================="
echo "  DIGEST COMPLETE!"
echo "=============================================="
echo "  seq=$SEQUENCE lt=$LOCKTIME"
echo "  Round 1: $R1_INDICES"
echo "  Round 2: $R2_INDICES"
"""


RUN_R2_ONLY_SH = r"""#!/bin/bash
# R2-only search: skip pinning + R1, go straight to R2.
# R1 was already verified on a previous run.
set -e
SEQUENCE=$1
LOCKTIME=$2
TOTAL_GPUS=$3
GLOBAL_OFFSET=$4

NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "=== R2-Only Digest Pipeline ==="
echo "  Pinning: seq=$SEQUENCE lt=$LOCKTIME (R1 already verified)"
echo "  Local GPUs: $NUM_GPUS"
echo "  Total GPUs: $TOTAL_GPUS (offset $GLOBAL_OFFSET)"

mkdir -p results logs
# Don't delete pinning_hit_assigned.txt or round1_final.txt — they were pre-uploaded
rm -f results/digest_hit_*.txt results/round2_final.txt
killall qsb_real qsb_digest 2>/dev/null || true
sleep 1

echo "=== Phase 1 (skipped — using pre-uploaded R1) ==="
if [ -f results/round1_final.txt ]; then
    R1_INDICES=$(grep "^indices=" results/round1_final.txt | head -1 | cut -d= -f2)
    echo "  Round 1 (pre-uploaded): $R1_INDICES"
else
    echo "  WARNING: no round1_final.txt found!"
fi

echo "=== Phase 2: Digest Round 2 ==="
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    stdbuf -oL ./qsb_digest digest_r2.bin $gpu $SEQUENCE $LOCKTIME $TOTAL_GPUS $GLOBAL_OFFSET > logs/dig2_gpu_${gpu}.log 2>&1 &
    sleep 1
done

while true; do
    sleep 15
    if compgen -G "results/digest_hit_*.txt" > /dev/null 2>&1; then
        echo "  *** DIGEST R2 HIT! ***"
        killall qsb_digest 2>/dev/null || true
        sleep 2
        cat results/digest_hit_*.txt
        break
    fi
    if ! pgrep -x qsb_digest > /dev/null; then
        echo "  (R2 local slice done, waiting for others...)"
        sleep 60
    fi
done

R2_FILE=$(ls results/digest_hit_*.txt | head -1)
R2_INDICES=$(grep "^indices=" "$R2_FILE" | head -1 | cut -d= -f2)
cp "$R2_FILE" results/round2_final.txt
echo "  Round 2: $R2_INDICES"

echo "=============================================="
echo "  R2 COMPLETE!"
echo "=============================================="
echo "  seq=$SEQUENCE lt=$LOCKTIME"
echo "  Round 2: $R2_INDICES"
"""


def cmd_rent_solo():
    """Rent the N biggest machines available (within budget) and assign each
    ONE different pinning hit. Each machine runs digest SOLO (total_gpus =
    local count, offset = 0) so it covers the full search space independently.
    
    Reads pinning hits from results/solo_pinnings.txt (one pinning per machine).
    Format per pinning: 4 lines (sequence=.., locktime=.., hash_choice=.., recid=..)
    separated by blank lines.
    """
    pin_path = Path("results/solo_pinnings.txt")
    if not pin_path.exists():
        print("ERROR: results/solo_pinnings.txt not found.")
        print("Create it with your chosen pinning hits, one block per machine, e.g.:")
        print()
        print("sequence=2147501802")
        print("locktime=702856043")
        print("hash_choice=0")
        print("recid=1")
        print()
        print("sequence=2147507660")
        print("...")
        return

    # Parse the file into a list of pinning dicts
    pinnings = []
    current = {}
    for line in pin_path.read_text().splitlines():
        line = line.strip()
        if not line:
            if current:
                pinnings.append(current)
                current = {}
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            current[k.strip()] = v.strip()
    if current:
        pinnings.append(current)

    if not pinnings:
        print("ERROR: no pinnings parsed from file.")
        return

    print(f"Found {len(pinnings)} pinnings to assign to {len(pinnings)} solo machines.")
    for i, p in enumerate(pinnings):
        print(f"  {i}: seq={p.get('sequence')} lt={p.get('locktime')} hc={p.get('hash_choice')} recid={p.get('recid')}")

    # Search for offers. For solo mode we want BIG machines (>=10 GPUs preferred).
    # Relax the price cap substantially because big-GPU rigs tend to be pricier per GPU.
    SOLO_MIN_GPUS = 10          # only consider rigs with >= 10 GPUs
    SOLO_MAX_PPG = 1.50         # up to $1.50/GPU/hr acceptable for bigger machines
    print(f"\nSearching vast.ai for offers with >= {SOLO_MIN_GPUS} GPUs (up to ${SOLO_MAX_PPG}/GPU/hr)...")
    offers = search_offers(SOLO_MIN_GPUS, ACCEPTABLE_GPU_NAMES, max_dollars_per_gpu_hr=SOLO_MAX_PPG)
    if not offers:
        print("No matching offers found — try lowering SOLO_MIN_GPUS or raising SOLO_MAX_PPG.")
        return

    # Sort: prefer bigger rigs, then 5090 over 4090 (5090 is ~1.5x faster),
    # then cheaper $/GPU. A simple composite: "effective GPUs" = num_gpus × speed_factor,
    # and we want to maximize effective GPUs while minimizing cost per effective GPU.
    def speed_factor(gpu_name):
        if "5090" in gpu_name: return 1.5
        if "4090" in gpu_name: return 1.0
        return 1.0
    for o in offers:
        o["eff_gpus"] = o["num_gpus"] * speed_factor(o["gpu_name"])
        o["dph_per_eff"] = o["dph_total"] / o["eff_gpus"]

    # Primary sort: most effective GPUs first. Tiebreak: cheapest per effective GPU.
    offers_big = sorted(offers, key=lambda x: (-x["eff_gpus"], x["dph_per_eff"]))

    print(f"\nTop offers (most effective GPUs first; 5090 weighted 1.5x):")
    for o in offers_big[:10]:
        print(f"  id={o['id']}  {o['gpu_name']} × {o['num_gpus']}  "
              f"eff={o['eff_gpus']:.0f}  ${o['dph_total']:.3f}/hr  "
              f"(${o['dph_per_gpu']:.3f}/GPU/hr, ${o['dph_per_eff']:.3f}/eff/hr)  Rel={o['reliability']:.2f}")

    # Take the first N biggest (one per pinning)
    selected = offers_big[:len(pinnings)]
    if len(selected) < len(pinnings):
        print(f"Only {len(selected)} offers available for {len(pinnings)} pinnings.")
        return

    total_gpus_all = sum(o["num_gpus"] for o in selected)
    total_cost = sum(o["dph_total"] for o in selected)
    print(f"\nWill rent {len(selected)} machines, {total_gpus_all} total GPUs, ${total_cost:.2f}/hr")
    ans = input("Proceed? [y/N]: ")
    if ans.strip().lower() != "y":
        print("Aborted.")
        return

    import hashlib
    import base64
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    local_zip_bytes = Path(ZIP_FILE).read_bytes()
    local_sha = hashlib.sha256(local_zip_bytes).hexdigest()
    local_size = len(local_zip_bytes)
    print(f"\nLocal zip: {ZIP_FILE} ({local_size} bytes, sha256 {local_sha[:16]}...)")

    script_b64 = base64.b64encode(RUN_ALL_DIGEST_SH.encode()).decode()

    lock = threading.Lock()
    rentals: List[Rental] = []

    def _full_pipeline(offer, pinning, idx):
        tag = f"[offer {offer['id']}]"
        inst_id = None
        try:
            inst_id = create_instance(offer["id"])
            tag = f"[{inst_id}]"
            print(f"{tag} created from offer {offer['id']} (pinning #{idx}: seq={pinning.get('sequence')})")

            host, port = wait_for_ssh(inst_id, timeout=900, verbose=False)
            print(f"{tag} ssh ready at {host}:{port}")

            r = Rental(
                instance_id=inst_id,
                ssh_host=host,
                ssh_port=port,
                num_gpus=offer["num_gpus"],
                gpu_name=offer["gpu_name"],
                dollars_per_hr=offer["dph_total"],
                global_offset=0,        # SOLO: each machine's offset is 0
                started=time.time(),
            )

            last_err = None
            for attempt in range(1, 13):
                try:
                    ssh_cmd(r, "echo PING", timeout=20)
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(10)
            else:
                raise RuntimeError(f"ssh never became responsive: {last_err}")
            print(f"{tag} ssh responsive")

            setup_out = ssh_cmd(r, SETUP_SCRIPT, timeout=300)
            if "SETUP_OK" not in setup_out:
                raise RuntimeError(f"setup failed: {setup_out[-400:]}")
            print(f"{tag} tools installed")

            ssh_cmd(r, "mkdir -p /work && cd /work && rm -rf ./*")
            scp_to(r, ZIP_FILE, f"/work/{ZIP_FILE}")
            info = ssh_cmd(r, f"cd /work && stat -c %s {ZIP_FILE} && sha256sum {ZIP_FILE}").strip().splitlines()
            remote_size = int(info[0])
            remote_sha = info[1].split()[0]
            if remote_size != local_size or remote_sha != local_sha:
                raise RuntimeError(f"zip upload mismatch")
            print(f"{tag} zip uploaded & verified")

            ssh_cmd(r, f"cd /work && unzip -o {ZIP_FILE} && "
                       "nvcc -O3 -o qsb_real qsb_real_search.cu -lcrypto -lm && "
                       "nvcc -O3 -o qsb_digest qsb_digest_search.cu -lcrypto -lm && "
                       "nvcc -O3 -o verify_gpu verify_gpu.cu -lcrypto -lm && "
                       "mkdir -p results logs",
                    timeout=900)
            print(f"{tag} compiled")

            # Write run_all_digest.sh
            ssh_cmd(r,
                f"echo {script_b64} | base64 -d > /work/run_all_digest.sh && "
                "chmod +x /work/run_all_digest.sh",
                timeout=30)

            # SOLO mode: total_gpus = local num_gpus, offset = 0
            # This GPU set covers the WHOLE search space for this pinning.
            seq = pinning["sequence"]
            lt = pinning["locktime"]

            # Write the pinning_hit file to the remote so `collect` can find it.
            # This file also records the assigned pinning for inspection/recovery.
            pin_lines = "\n".join(f"{k}={v}" for k, v in pinning.items())
            pin_b64 = base64.b64encode(pin_lines.encode()).decode()
            ssh_cmd(r,
                "mkdir -p /work/results && "
                f"echo {pin_b64} | base64 -d > /work/results/pinning_hit_assigned.txt",
                timeout=30)

            launch_cmd = f"bash run_all_digest.sh {seq} {lt} {offer['num_gpus']} 0"
            ssh_bg(r, launch_cmd)
            r.status = "digest_solo"
            # Attach the pinning to the rental for later reference
            r.__dict__["pinning"] = pinning
            print(f"{tag} LAUNCHED ({launch_cmd})")

            with lock:
                rentals.append(r)
                save_state(rentals, {"mode": "solo"})
            return r, None
        except Exception as e:
            msg = str(e)[:500]
            print(f"{tag} FAILED: {msg}")
            if inst_id is not None:
                try:
                    destroy_instance(inst_id)
                    print(f"{tag} destroyed")
                except Exception:
                    pass
            return None, msg

    with ThreadPoolExecutor(max_workers=len(selected)) as ex:
        futures = [ex.submit(_full_pipeline, o, p, i)
                   for i, (o, p) in enumerate(zip(selected, pinnings))]
        for fut in as_completed(futures):
            fut.result()

    running = sum(1 for r in rentals if r.status == "digest_solo")
    total_running_gpus = sum(r.num_gpus for r in rentals if r.status == "digest_solo")
    print(f"\n{running}/{len(selected)} machines running ({total_running_gpus} GPUs in solo mode).")


def cmd_relaunch_digest():
    """Given a verified pinning hit in results/pinning_hit.txt, kill
    whatever's running on every machine and launch digest round 1+2 fresh
    with the given seq/lt. Uses the PLANNED total_gpus for proper sharding.
    """
    rentals, extra = load_state()
    if not rentals:
        print("No rentals on file.")
        return

    pin_path = Path("results/pinning_hit.txt")
    if not pin_path.exists():
        print("ERROR: results/pinning_hit.txt not found. Save the chosen")
        print("pinning hit to that path first. Format should be lines like:")
        print("  sequence=2147501802")
        print("  locktime=702856043")
        return

    kv = _parse_kv(pin_path.read_text())
    try:
        seq = int(kv["sequence"])
        lt = int(kv["locktime"])
    except Exception as e:
        print(f"ERROR: couldn't parse sequence/locktime: {e}")
        return

    total_gpus = extra.get("total_gpus") or sum(r.num_gpus for r in rentals)
    print(f"Relaunching {len(rentals)} machines onto digest.")
    print(f"  Pinning: seq={seq} lt={lt}")
    print(f"  Total GPUs: {total_gpus}")
    print()

    ans = input("Kill all running processes and restart digest-only on every machine? [y/N]: ")
    if ans.strip().lower() != "y":
        print("Aborted.")
        return

    import base64
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # b64-encode the script to avoid shell-quoting pain
    script_b64 = base64.b64encode(RUN_ALL_DIGEST_SH.encode()).decode()

    def _relaunch_one(r: Rental):
        tag = f"[{r.instance_id}]"
        try:
            # 1) kill everything
            ssh_cmd(r,
                "killall -9 qsb_real qsb_digest 2>/dev/null || true; "
                "pkill -9 -f run_all.sh 2>/dev/null || true; "
                "pkill -9 -f run_all_digest.sh 2>/dev/null || true; "
                "sleep 1",
                timeout=30)
            print(f"{tag} killed previous processes")

            # 2) write new launcher script
            ssh_cmd(r,
                f"echo {script_b64} | base64 -d > /work/run_all_digest.sh && "
                "chmod +x /work/run_all_digest.sh",
                timeout=30)

            # 3) launch in background
            launch_cmd = f"bash run_all_digest.sh {seq} {lt} {total_gpus} {r.global_offset}"
            ssh_bg(r, launch_cmd)
            r.status = "digest"
            print(f"{tag} relaunched ({launch_cmd})")
            return r, None
        except Exception as e:
            print(f"{tag} RELAUNCH FAILED: {str(e)[:500]}")
            return r, str(e)

    with ThreadPoolExecutor(max_workers=len(rentals)) as ex:
        futures = [ex.submit(_relaunch_one, r) for r in rentals]
        for fut in as_completed(futures):
            fut.result()

    save_state(rentals, {"total_gpus": total_gpus, "phase": "digest", "sequence": seq, "locktime": lt})
    print(f"\nAll machines relaunched onto digest. Use `collect` to gather R1 and R2 hits.")


# ============================================================
# main
# ============================================================

def cmd_rent_r2_only():
    """R2-ONLY: Rent ONE machine, upload zip + pre-existing R1 result + pinning,
    run R2 search only.
    
    Reads:
      - results/r2_only_pinning.txt    (pinning to use, kv format)
      - results/r2_only_round1.txt     (R1 hit content to pre-upload)
    
    Format of r2_only_pinning.txt:
      sequence=2147507660
      locktime=570428883
      hash_choice=0
      recid=1
    
    Format of r2_only_round1.txt (same as a digest_hit_*.txt):
      indices=14,22,35,77,78,99,115,123,125
      hash_choice=0
      recid=0
    """
    pin_path = Path("results/r2_only_pinning.txt")
    r1_path = Path("results/r2_only_round1.txt")
    if not pin_path.exists():
        print(f"ERROR: {pin_path} not found.")
        print("Create it with the verified pinning, e.g.:")
        print("  sequence=2147507660")
        print("  locktime=570428883")
        print("  hash_choice=0")
        print("  recid=1")
        return
    if not r1_path.exists():
        print(f"ERROR: {r1_path} not found.")
        print("Create it with the verified R1 hit, e.g.:")
        print("  indices=14,22,35,77,78,99,115,123,125")
        print("  hash_choice=0")
        print("  recid=0")
        return

    # Parse pinning
    pinning = {}
    for line in pin_path.read_text().splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            pinning[k.strip()] = v.strip()
    
    if "sequence" not in pinning or "locktime" not in pinning:
        print(f"ERROR: pinning file must contain at least sequence= and locktime=")
        return

    r1_content = r1_path.read_text()
    print(f"R2-ONLY mode")
    print(f"  Pinning: seq={pinning['sequence']} lt={pinning['locktime']}")
    print(f"  R1 (pre-loaded): {r1_content.strip()[:100]}")

    # Search for ONE big machine
    R2_MIN_GPUS = 10           # prefer rigs with >= 10 GPUs
    R2_MAX_PPG = 1.50          # up to $1.50/GPU/hr acceptable
    print(f"\nSearching vast.ai for offers with >= {R2_MIN_GPUS} GPUs (up to ${R2_MAX_PPG}/GPU/hr)...")
    offers = search_offers(R2_MIN_GPUS, ACCEPTABLE_GPU_NAMES, max_dollars_per_gpu_hr=R2_MAX_PPG)
    if not offers:
        print("No matching offers found.")
        return

    def speed_factor(gpu_name):
        if "5090" in gpu_name: return 1.5
        if "4090" in gpu_name: return 1.0
        return 1.0
    for o in offers:
        o["eff_gpus"] = o["num_gpus"] * speed_factor(o["gpu_name"])
        o["dph_per_eff"] = o["dph_total"] / o["eff_gpus"]
    offers_big = sorted(offers, key=lambda x: (-x["eff_gpus"], x["dph_per_eff"]))

    print(f"\nTop offers:")
    for o in offers_big[:5]:
        print(f"  id={o['id']}  {o['gpu_name']} x {o['num_gpus']}  "
              f"eff={o['eff_gpus']:.0f}  ${o['dph_total']:.3f}/hr  Rel={o['reliability']:.2f}")

    selected = offers_big[:1]   # Just one machine
    if not selected:
        print("No offers selected.")
        return

    o = selected[0]
    est_cost = o["dph_total"] * 4  # 4-hour estimate
    print(f"\nWill rent ONE machine: {o['gpu_name']} x {o['num_gpus']} at ${o['dph_total']:.3f}/hr")
    print(f"  Estimated 4-hour cost: ${est_cost:.2f}")
    ans = input("Proceed? [y/N]: ")
    if ans.strip().lower() != "y":
        print("Aborted.")
        return

    import hashlib, base64

    local_zip_bytes = Path(ZIP_FILE).read_bytes()
    local_sha = hashlib.sha256(local_zip_bytes).hexdigest()
    local_size = len(local_zip_bytes)
    print(f"\nLocal zip: {ZIP_FILE} ({local_size} bytes, sha256 {local_sha[:16]}...)")

    script_b64 = base64.b64encode(RUN_R2_ONLY_SH.encode()).decode()
    pin_lines = "\n".join(f"{k}={v}" for k, v in pinning.items())
    pin_b64 = base64.b64encode(pin_lines.encode()).decode()
    r1_b64 = base64.b64encode(r1_content.encode()).decode()

    rentals: List[Rental] = []
    inst_id = None
    try:
        inst_id = create_instance(o["id"])
        tag = f"[{inst_id}]"
        print(f"{tag} created from offer {o['id']}")

        host, port = wait_for_ssh(inst_id, timeout=900, verbose=False)
        print(f"{tag} ssh ready at {host}:{port}")

        r = Rental(
            instance_id=inst_id,
            ssh_host=host,
            ssh_port=port,
            num_gpus=o["num_gpus"],
            gpu_name=o["gpu_name"],
            dollars_per_hr=o["dph_total"],
            global_offset=0,
            started=time.time(),
        )

        last_err = None
        for attempt in range(1, 13):
            try:
                ssh_cmd(r, "echo PING", timeout=20)
                break
            except Exception as e:
                last_err = e
                time.sleep(10)
        else:
            raise RuntimeError(f"ssh never became responsive: {last_err}")
        print(f"{tag} ssh responsive")

        setup_out = ssh_cmd(r, SETUP_SCRIPT, timeout=300)
        if "SETUP_OK" not in setup_out:
            raise RuntimeError(f"setup failed: {setup_out[-400:]}")
        print(f"{tag} tools installed")

        ssh_cmd(r, "mkdir -p /work && cd /work && rm -rf ./*")
        scp_to(r, ZIP_FILE, f"/work/{ZIP_FILE}")
        info = ssh_cmd(r, f"cd /work && stat -c %s {ZIP_FILE} && sha256sum {ZIP_FILE}").strip().splitlines()
        remote_size = int(info[0])
        remote_sha = info[1].split()[0]
        if remote_size != local_size or remote_sha != local_sha:
            raise RuntimeError(f"zip upload mismatch")
        print(f"{tag} zip uploaded & verified ({local_size} bytes)")

        # Unzip (flat structure, files go directly into /work)
        ssh_cmd(r, f"cd /work && unzip -o {ZIP_FILE}")
        # Compile (now includes test_der for DER sanity check)
        ssh_cmd(r, "cd /work && "
                   "nvcc -O3 -o qsb_real qsb_real_search.cu -lcrypto -lm && "
                   "nvcc -O3 -o qsb_digest qsb_digest_search.cu -lcrypto -lm && "
                   "nvcc -O3 -o verify_gpu verify_gpu.cu -lcrypto -lm && "
                   "nvcc -O3 -o test_der test_der.cu && "
                   "mkdir -p results logs",
                timeout=900)
        # SANITY CHECK: run test_der to verify gpu_is_valid_der rejects 0xd1... 
        # If it returns 1, the GPU/compiler is broken and we should NOT continue
        try:
            test_out = ssh_cmd(r, "cd /work && ./test_der", timeout=30).strip()
            print(f"{tag} test_der output:")
            for line in test_out.splitlines(): print(f"      {line}")
            if "returned: 0" in test_out:
                print(f"{tag} ✓ GPU is_valid_der correctly rejects non-0x30 hash")
            elif "returned: 1" in test_out:
                print(f"{tag} ✗✗✗ GPU is_valid_der WRONGLY accepts non-0x30 hash!")
                print(f"{tag} This is the bug! Aborting before launching search.")
                raise RuntimeError("test_der confirms GPU DER bug")
            else:
                print(f"{tag} ⚠️  test_der output unclear, continuing anyway")
        except Exception as te:
            print(f"{tag} test_der check failed: {te}")
            # Don't abort the run for this; continue with the main search
        print(f"{tag} compiled")

        # Write run_r2_only.sh
        ssh_cmd(r,
            f"echo {script_b64} | base64 -d > /work/run_r2_only.sh && "
            "chmod +x /work/run_r2_only.sh",
            timeout=30)
        print(f"{tag} run_r2_only.sh installed")

        # Pre-upload pinning result
        ssh_cmd(r,
            "mkdir -p /work/results && "
            f"echo {pin_b64} | base64 -d > /work/results/pinning_hit_assigned.txt",
            timeout=30)
        print(f"{tag} pinning pre-uploaded")

        # Pre-upload R1 result (so collect can find it)
        ssh_cmd(r,
            f"echo {r1_b64} | base64 -d > /work/results/round1_final.txt",
            timeout=30)
        print(f"{tag} R1 result pre-uploaded")

        # Sanity: list /work to confirm
        listing = ssh_cmd(r, "ls -la /work/ /work/results/", timeout=30)
        print(f"{tag} /work contents:")
        for line in listing.splitlines():
            print(f"      {line}")

        # Launch R2-only
        seq = pinning["sequence"]
        lt = pinning["locktime"]
        launch_cmd = f"bash run_r2_only.sh {seq} {lt} {o['num_gpus']} 0"
        ssh_bg(r, launch_cmd)
        r.status = "r2_only"
        r.__dict__["pinning"] = pinning
        print(f"{tag} LAUNCHED ({launch_cmd})")

        rentals.append(r)
        save_state(rentals, {"mode": "r2_only"})
        print(f"\n[OK] One machine running R2-only search.")
        print(f"  Run `python3 {sys.argv[0]} collect` to monitor.")

    except Exception as e:
        msg = str(e)[:500]
        print(f"FAILED: {msg}")
        if inst_id is not None:
            try:
                destroy_instance(inst_id)
                print(f"[{inst_id}] destroyed")
            except Exception:
                pass


def cmd_rent_r2_fast():
    """Rent ONE machine directly with the FAST config (total=32, offset=16) targeting
    first=30, where the previous (false) hit was reported.
    
    Use this when you want JUST ONE fast machine instead of running a slow + fast pair.
    
    WARNING: this single machine only covers firsts {16-31, 48-63, 80-95, 112-121}.
    If the real R2 hit's first index is OUTSIDE this range, this machine will NEVER find it.
    Use rent_r2_only + add_r2_machine for full coverage.
    
    Best used together with v14's test_der sanity check + verify_r2_hit.py CPU verifier.
    """
    pin_path = Path("results/r2_only_pinning.txt")
    r1_path = Path("results/r2_only_round1.txt")
    if not pin_path.exists() or not r1_path.exists():
        print(f"ERROR: need {pin_path} and {r1_path}")
        return

    # Parse pinning + R1
    pinning = {}
    for line in pin_path.read_text().splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            pinning[k.strip()] = v.strip()
    r1_content = r1_path.read_text()
    
    print(f"R2-FAST mode (single machine, targeted at first=30)")
    print(f"  Pinning: seq={pinning['sequence']} lt={pinning['locktime']}")
    print(f"  Strategy: machine's GPU 14 starts at first=30 (where previous hit was reported)")
    print(f"  Expected hit time: ~9-30 minutes")
    print(f"  Coverage: firsts {{16-31, 48-63, 80-95, 112-121}} ONLY")
    print(f"  WARNING: if real hit is outside this range, this machine will not find it.")

    # Search for ONE big machine
    R2_MIN_GPUS = 10
    R2_MAX_PPG = 1.50
    print(f"\nSearching vast.ai for offers with >= {R2_MIN_GPUS} GPUs...")
    offers = search_offers(R2_MIN_GPUS, ACCEPTABLE_GPU_NAMES, max_dollars_per_gpu_hr=R2_MAX_PPG)
    if not offers:
        print("No matching offers found.")
        return

    def speed_factor(gpu_name):
        if "5090" in gpu_name: return 1.5
        if "4090" in gpu_name: return 1.0
        return 1.0
    for o in offers:
        o["eff_gpus"] = o["num_gpus"] * speed_factor(o["gpu_name"])
        o["dph_per_eff"] = o["dph_total"] / o["eff_gpus"]
    offers_big = sorted(offers, key=lambda x: (-x["eff_gpus"], x["dph_per_eff"]))

    print(f"\nTop offers:")
    for o in offers_big[:5]:
        print(f"  id={o['id']}  {o['gpu_name']} x {o['num_gpus']}  "
              f"eff={o['eff_gpus']:.0f}  ${o['dph_total']:.3f}/hr  Rel={o['reliability']:.2f}")

    selected = offers_big[:1]
    if not selected:
        print("No offers selected.")
        return
    o = selected[0]

    # Compute fast config
    desired_first = 30
    target_local_gpu = min(o["num_gpus"] - 1, 14)
    new_offset = desired_first - target_local_gpu
    new_total = max(32, o["num_gpus"] * 2)
    assert new_offset + target_local_gpu == desired_first

    print(f"\nPlan: 1 machine, {o['num_gpus']} GPUs at ${o['dph_total']:.3f}/hr")
    print(f"  total_gpus_override = {new_total}, global_offset = {new_offset}")
    print(f"  GPU {target_local_gpu} of this machine starts at first={desired_first}")
    print(f"  Estimated cost if hit found in 30min: ${o['dph_total']*0.5:.2f}")
    ans = input("Proceed? [y/N]: ")
    if ans.strip().lower() != "y":
        print("Aborted.")
        return

    import hashlib, base64

    local_zip_bytes = Path(ZIP_FILE).read_bytes()
    local_sha = hashlib.sha256(local_zip_bytes).hexdigest()
    local_size = len(local_zip_bytes)

    script_b64 = base64.b64encode(RUN_R2_ONLY_SH.encode()).decode()
    pin_lines = "\n".join(f"{k}={v}" for k, v in pinning.items())
    pin_b64 = base64.b64encode(pin_lines.encode()).decode()
    r1_b64 = base64.b64encode(r1_content.encode()).decode()

    rentals: List[Rental] = []
    inst_id = None
    try:
        inst_id = create_instance(o["id"])
        tag = f"[{inst_id}]"
        print(f"{tag} created from offer {o['id']}")

        host, port = wait_for_ssh(inst_id, timeout=900, verbose=False)
        print(f"{tag} ssh ready at {host}:{port}")

        r = Rental(
            instance_id=inst_id,
            ssh_host=host,
            ssh_port=port,
            num_gpus=o["num_gpus"],
            gpu_name=o["gpu_name"],
            dollars_per_hr=o["dph_total"],
            global_offset=new_offset,
            started=time.time(),
        )

        last_err = None
        for attempt in range(1, 13):
            try:
                ssh_cmd(r, "echo PING", timeout=20)
                break
            except Exception as e:
                last_err = e
                time.sleep(10)
        else:
            raise RuntimeError(f"ssh never became responsive: {last_err}")
        print(f"{tag} ssh responsive")

        setup_out = ssh_cmd(r, SETUP_SCRIPT, timeout=300)
        if "SETUP_OK" not in setup_out:
            raise RuntimeError(f"setup failed: {setup_out[-400:]}")
        print(f"{tag} tools installed")

        ssh_cmd(r, "mkdir -p /work && cd /work && rm -rf ./*")
        scp_to(r, ZIP_FILE, f"/work/{ZIP_FILE}")
        info = ssh_cmd(r, f"cd /work && stat -c %s {ZIP_FILE} && sha256sum {ZIP_FILE}").strip().splitlines()
        if int(info[0]) != local_size or info[1].split()[0] != local_sha:
            raise RuntimeError(f"zip upload mismatch")
        print(f"{tag} zip uploaded & verified")

        ssh_cmd(r, f"cd /work && unzip -o {ZIP_FILE}")
        ssh_cmd(r, "cd /work && "
                   "nvcc -O3 -o qsb_real qsb_real_search.cu -lcrypto -lm && "
                   "nvcc -O3 -o qsb_digest qsb_digest_search.cu -lcrypto -lm && "
                   "nvcc -O3 -o verify_gpu verify_gpu.cu -lcrypto -lm && "
                   "nvcc -O3 -o test_der test_der.cu && "
                   "mkdir -p results logs",
                timeout=900)
        print(f"{tag} compiled")
        # SANITY CHECK
        try:
            test_out = ssh_cmd(r, "cd /work && ./test_der", timeout=30).strip()
            print(f"{tag} test_der output:")
            for line in test_out.splitlines(): print(f"      {line}")
            if "returned: 1" in test_out:
                print(f"{tag} XXX GPU is_valid_der WRONGLY accepts non-0x30 hash!")
                print(f"{tag} BUG CONFIRMED at compiler level. Aborting.")
                raise RuntimeError("test_der confirms GPU DER bug")
        except Exception as te:
            if "GPU DER bug" in str(te):
                raise
            print(f"{tag} test_der check failed: {te}")

        ssh_cmd(r,
            f"echo {script_b64} | base64 -d > /work/run_r2_only.sh && "
            "chmod +x /work/run_r2_only.sh",
            timeout=30)
        ssh_cmd(r,
            "mkdir -p /work/results && "
            f"echo {pin_b64} | base64 -d > /work/results/pinning_hit_assigned.txt",
            timeout=30)
        ssh_cmd(r,
            f"echo {r1_b64} | base64 -d > /work/results/round1_final.txt",
            timeout=30)
        print(f"{tag} pinning + R1 pre-uploaded")

        seq = pinning["sequence"]
        lt = pinning["locktime"]
        launch_cmd = f"bash run_r2_only.sh {seq} {lt} {new_total} {new_offset}"
        ssh_bg(r, launch_cmd)
        r.status = "r2_only"
        r.__dict__["pinning"] = pinning
        print(f"{tag} LAUNCHED ({launch_cmd})")
        print(f"{tag} GPU {target_local_gpu} starts at first={desired_first} immediately")

        rentals.append(r)
        save_state(rentals, {"mode": "r2_only"})
        print(f"\n[OK] One FAST machine running R2 search.")
        print(f"  Run `python3 {sys.argv[0]} collect` to monitor.")
        print(f"  Expected hit: ~9-30 minutes if R2 is at first=30")

    except Exception as e:
        msg = str(e)[:500]
        print(f"FAILED: {msg}")
        if inst_id is not None:
            try:
                destroy_instance(inst_id)
                print(f"[{inst_id}] destroyed")
            except Exception:
                pass


def cmd_rent_r2_split4():
    """Rent FOUR machines in parallel with full coverage of the R2 search space.
    
    Each machine has 16 GPUs configured with total_gpus=64 and offsets:
      Machine A: offset=0   → covers firsts {0-15, 64-79}
      Machine B: offset=16  → covers firsts {16-31, 80-95}
      Machine C: offset=32  → covers firsts {32-47, 96-111}
      Machine D: offset=48  → covers firsts {48-63, 112-121}
    
    Together they cover ALL firsts 0-121 with 4x parallelism.
    Expected total search time: ~30-45 minutes.
    Cost: ~$15-25 (depends on which offers are selected).
    """
    pin_path = Path("results/r2_only_pinning.txt")
    r1_path = Path("results/r2_only_round1.txt")
    if not pin_path.exists() or not r1_path.exists():
        print(f"ERROR: need {pin_path} and {r1_path}")
        return

    pinning = {}
    for line in pin_path.read_text().splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            pinning[k.strip()] = v.strip()
    r1_content = r1_path.read_text()
    
    print(f"R2-SPLIT4 mode (4 machines in parallel, full coverage)")
    print(f"  Pinning: seq={pinning['sequence']} lt={pinning['locktime']}")
    print(f"  Strategy: 4 machines, each covering 1/4 of the firsts")
    print(f"  Expected total time: ~30-45 minutes for full search")

    R2_MIN_GPUS = 10
    R2_MAX_PPG = 1.50
    print(f"\nSearching vast.ai for offers with >= {R2_MIN_GPUS} GPUs...")
    offers = search_offers(R2_MIN_GPUS, ACCEPTABLE_GPU_NAMES, max_dollars_per_gpu_hr=R2_MAX_PPG)
    if not offers:
        print("No matching offers found.")
        return

    def speed_factor(gpu_name):
        if "5090" in gpu_name: return 1.5
        if "4090" in gpu_name: return 1.0
        return 1.0
    for o in offers:
        o["eff_gpus"] = o["num_gpus"] * speed_factor(o["gpu_name"])
        o["dph_per_eff"] = o["dph_total"] / o["eff_gpus"]
    offers_big = sorted(offers, key=lambda x: (-x["eff_gpus"], x["dph_per_eff"]))

    print(f"\nTop offers:")
    for o in offers_big[:8]:
        print(f"  id={o['id']}  {o['gpu_name']} x {o['num_gpus']}  "
              f"eff={o['eff_gpus']:.0f}  ${o['dph_total']:.3f}/hr  Rel={o['reliability']:.2f}")

    # Need 4 distinct offers. Prefer different machines/hosts when possible to avoid 
    # contention, but fall back to any 4 different offers if we can't.
    selected = []
    used_machines = set()
    used_offer_ids = set()
    
    # First pass: pick offers from distinct machines (if machine_id is available)
    for o in offers_big:
        if o["id"] in used_offer_ids:
            continue
        mid = o.get("machine_id")
        if mid is not None and mid in used_machines:
            continue
        selected.append(o)
        used_offer_ids.add(o["id"])
        if mid is not None:
            used_machines.add(mid)
        if len(selected) == 4:
            break
    
    # Second pass if we don't have 4 yet — accept duplicates of machine_id (just need 
    # different offer_ids)
    if len(selected) < 4:
        for o in offers_big:
            if o["id"] in used_offer_ids:
                continue
            selected.append(o)
            used_offer_ids.add(o["id"])
            if len(selected) == 4:
                break
    
    if len(selected) < 4:
        print(f"\nOnly found {len(selected)} distinct offers (need 4).")
        return

    # Configure each machine: offset = i * 16, total = 64, target 16 GPUs each
    # If a machine has != 16 GPUs, adjust offsets accordingly
    plan = []
    total_gpus = 64
    cum_offset = 0
    for i, o in enumerate(selected):
        ng = o["num_gpus"]
        # We want ~16 GPUs per machine. If a machine has more, it covers more firsts.
        # If less, it covers fewer.
        plan.append({
            "offer": o,
            "offset": cum_offset,
            "num_gpus": ng,
        })
        cum_offset += ng
    # Update total_gpus based on actual sum
    total_gpus = cum_offset
    
    print(f"\nPlan: 4 machines, total {total_gpus} GPUs:")
    for i, p in enumerate(plan):
        o = p["offer"]
        print(f"  Machine {chr(65+i)}: id={o['id']}  {o['gpu_name']} x {o['num_gpus']}  "
              f"offset={p['offset']}  ${o['dph_total']:.3f}/hr")
    total_hourly = sum(p["offer"]["dph_total"] for p in plan)
    print(f"  Total hourly: ${total_hourly:.2f}/hr")
    print(f"  Estimated cost (45 min): ${total_hourly*0.75:.2f}")
    ans = input("Proceed? [y/N]: ")
    if ans.strip().lower() != "y":
        print("Aborted.")
        return

    import hashlib, base64
    import threading

    local_zip_bytes = Path(ZIP_FILE).read_bytes()
    local_sha = hashlib.sha256(local_zip_bytes).hexdigest()
    local_size = len(local_zip_bytes)

    script_b64 = base64.b64encode(RUN_R2_ONLY_SH.encode()).decode()
    pin_lines = "\n".join(f"{k}={v}" for k, v in pinning.items())
    pin_b64 = base64.b64encode(pin_lines.encode()).decode()
    r1_b64 = base64.b64encode(r1_content.encode()).decode()
    
    seq = pinning["sequence"]
    lt = pinning["locktime"]

    rentals: List[Rental] = []
    rentals_lock = threading.Lock()
    failures = []
    failures_lock = threading.Lock()

    def setup_one_machine(idx, p):
        o = p["offer"]
        offset = p["offset"]
        machine_label = chr(65 + idx)
        inst_id = None
        try:
            inst_id = create_instance(o["id"])
            tag = f"[{inst_id}/{machine_label}]"
            print(f"{tag} created from offer {o['id']}")

            host, port = wait_for_ssh(inst_id, timeout=900, verbose=False)
            print(f"{tag} ssh ready at {host}:{port}")

            r = Rental(
                instance_id=inst_id,
                ssh_host=host,
                ssh_port=port,
                num_gpus=o["num_gpus"],
                gpu_name=o["gpu_name"],
                dollars_per_hr=o["dph_total"],
                global_offset=offset,
                started=time.time(),
            )
            r.__dict__["machine_label"] = machine_label

            last_err = None
            for attempt in range(1, 13):
                try:
                    ssh_cmd(r, "echo PING", timeout=20)
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(10)
            else:
                raise RuntimeError(f"ssh never became responsive: {last_err}")
            print(f"{tag} ssh responsive")

            setup_out = ssh_cmd(r, SETUP_SCRIPT, timeout=300)
            if "SETUP_OK" not in setup_out:
                raise RuntimeError(f"setup failed: {setup_out[-400:]}")
            print(f"{tag} tools installed")

            ssh_cmd(r, "mkdir -p /work && cd /work && rm -rf ./*")
            scp_to(r, ZIP_FILE, f"/work/{ZIP_FILE}")
            info = ssh_cmd(r, f"cd /work && stat -c %s {ZIP_FILE} && sha256sum {ZIP_FILE}").strip().splitlines()
            if int(info[0]) != local_size or info[1].split()[0] != local_sha:
                raise RuntimeError(f"zip upload mismatch")
            print(f"{tag} zip uploaded & verified")

            ssh_cmd(r, f"cd /work && unzip -o {ZIP_FILE}")
            ssh_cmd(r, "cd /work && "
                       "nvcc -O3 -o qsb_real qsb_real_search.cu -lcrypto -lm && "
                       "nvcc -O3 -o qsb_digest qsb_digest_search.cu -lcrypto -lm && "
                       "nvcc -O3 -o verify_gpu verify_gpu.cu -lcrypto -lm && "
                       "nvcc -O3 -o test_der test_der.cu && "
                       "mkdir -p results logs",
                    timeout=900)
            print(f"{tag} compiled")
            try:
                test_out = ssh_cmd(r, "cd /work && ./test_der", timeout=30).strip()
                if "returned: 1" in test_out:
                    print(f"{tag} XXX test_der reports DER bug! Aborting this machine.")
                    raise RuntimeError("test_der confirms GPU DER bug")
            except Exception as te:
                if "GPU DER bug" in str(te):
                    raise
                print(f"{tag} test_der check failed: {te}")

            ssh_cmd(r,
                f"echo {script_b64} | base64 -d > /work/run_r2_only.sh && "
                "chmod +x /work/run_r2_only.sh",
                timeout=30)
            ssh_cmd(r,
                "mkdir -p /work/results && "
                f"echo {pin_b64} | base64 -d > /work/results/pinning_hit_assigned.txt",
                timeout=30)
            ssh_cmd(r,
                f"echo {r1_b64} | base64 -d > /work/results/round1_final.txt",
                timeout=30)
            print(f"{tag} pinning + R1 pre-uploaded")

            launch_cmd = f"bash run_r2_only.sh {seq} {lt} {total_gpus} {offset}"
            ssh_bg(r, launch_cmd)
            r.status = "r2_only"
            r.__dict__["pinning"] = pinning
            print(f"{tag} LAUNCHED ({launch_cmd})")
            print(f"{tag} covers firsts: {{{offset}-{offset+o['num_gpus']-1}}} ∪ {{{offset+total_gpus}-{offset+total_gpus+o['num_gpus']-1}}}")

            with rentals_lock:
                rentals.append(r)
                save_state(rentals, {"mode": "r2_only", "split4": True, "total_gpus": total_gpus})

        except Exception as e:
            msg = str(e)[:500]
            print(f"[{machine_label}] FAILED: {msg}")
            with failures_lock:
                failures.append((machine_label, msg))
            if inst_id is not None:
                try:
                    destroy_instance(inst_id)
                    print(f"[{inst_id}/{machine_label}] destroyed")
                except Exception:
                    pass

    # Launch all 4 setups in parallel
    threads = []
    for idx, p in enumerate(plan):
        t = threading.Thread(target=setup_one_machine, args=(idx, p))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    if failures:
        print(f"\n[WARN] {len(failures)} machine(s) failed to set up.")
        for ml, msg in failures:
            print(f"  Machine {ml}: {msg[:200]}")
    print(f"\n[OK] {len(rentals)}/4 machines running R2 search.")
    print(f"  Run `python3 {sys.argv[0]} collect` to monitor.")
    print(f"  Expected: full R2 search complete in ~30-45 min")


def cmd_add_split4_machine():
    """Add a replacement machine for a failed slot in the split4 setup.
    
    Reads the existing state file to determine which offsets {0, 16, 32, 48} are 
    already covered. If state is missing, asks user which offsets to add.
    Uses total_gpus=64 to match the split4 scheme.
    """
    rentals, extra = load_state()
    expected_offsets = {0, 16, 32, 48}
    
    if not rentals:
        print("WARNING: no rentals in local state file.")
        print("This can happen if state was deleted while machines are still running on vast.ai.")
        print()
        print("Which offsets need to be added (comma-separated, e.g. '16' or '16,32')?")
        print("Possible offsets: 0, 16, 32, 48")
        ans = input("Offsets to add: ").strip()
        try:
            missing = {int(x.strip()) for x in ans.split(",") if x.strip()}
            missing = missing & expected_offsets
        except Exception:
            print("Invalid input.")
            return
        if not missing:
            print("No valid offsets specified.")
            return
        existing_offsets = expected_offsets - missing
        rentals = []  # empty so we don't try to filter against existing offer_ids
    else:
        existing_offsets = {r.global_offset for r in rentals}
        missing = expected_offsets - existing_offsets
    
    if not missing:
        print("All split4 offsets are already covered. Nothing to add.")
        for r in rentals:
            print(f"  [{r.instance_id}] offset={r.global_offset}")
        return
    
    print(f"Currently covered offsets: {sorted(existing_offsets)}")
    print(f"Missing offsets:           {sorted(missing)}")
    
    pin_path = Path("results/r2_only_pinning.txt")
    r1_path = Path("results/r2_only_round1.txt")
    if not pin_path.exists() or not r1_path.exists():
        print(f"ERROR: need {pin_path} and {r1_path}")
        return
    
    pinning = {}
    for line in pin_path.read_text().splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            pinning[k.strip()] = v.strip()
    r1_content = r1_path.read_text()
    
    R2_MIN_GPUS = 10
    R2_MAX_PPG = 1.50
    print(f"\nSearching vast.ai for offers...")
    offers = search_offers(R2_MIN_GPUS, ACCEPTABLE_GPU_NAMES, max_dollars_per_gpu_hr=R2_MAX_PPG)
    if not offers:
        print("No matching offers found.")
        return
    
    def speed_factor(gpu_name):
        if "5090" in gpu_name: return 1.5
        if "4090" in gpu_name: return 1.0
        return 1.0
    for o in offers:
        o["eff_gpus"] = o["num_gpus"] * speed_factor(o["gpu_name"])
        o["dph_per_eff"] = o["dph_total"] / o["eff_gpus"]
    
    # Avoid offers from machines we already rented
    existing_offer_ids = {getattr(r, "offer_id", None) for r in rentals}
    offers_filtered = [o for o in offers if o["id"] not in existing_offer_ids]
    offers_big = sorted(offers_filtered, key=lambda x: (-x["eff_gpus"], x["dph_per_eff"]))
    
    print(f"\nTop offers:")
    for o in offers_big[:5]:
        print(f"  id={o['id']}  {o['gpu_name']} x {o['num_gpus']}  "
              f"eff={o['eff_gpus']:.0f}  ${o['dph_total']:.3f}/hr  Rel={o['reliability']:.2f}")
    
    # Pick one offer per missing offset
    sorted_missing = sorted(missing)
    plan = []
    for i, off in enumerate(sorted_missing):
        if i >= len(offers_big):
            print(f"Not enough distinct offers for all {len(sorted_missing)} missing offsets.")
            break
        plan.append({"offer": offers_big[i], "offset": off})
    
    if not plan:
        print("No offers to use.")
        return
    
    total_gpus = 64  # MUST match the original split4 setup
    print(f"\nPlan ({len(plan)} machine(s)):")
    for p in plan:
        o = p["offer"]
        print(f"  offset={p['offset']}: id={o['id']}  {o['gpu_name']} x {o['num_gpus']}  "
              f"${o['dph_total']:.3f}/hr  → covers firsts {{{p['offset']}-{p['offset']+o['num_gpus']-1}}} ∪ {{{p['offset']+total_gpus}-{p['offset']+total_gpus+o['num_gpus']-1}}}")
    total_hourly = sum(p["offer"]["dph_total"] for p in plan)
    print(f"  Total: ${total_hourly:.2f}/hr")
    print(f"  Estimated cost (45 min): ${total_hourly*0.75:.2f}")
    ans = input("Proceed? [y/N]: ")
    if ans.strip().lower() != "y":
        print("Aborted.")
        return
    
    import hashlib, base64
    import threading
    
    local_zip_bytes = Path(ZIP_FILE).read_bytes()
    local_sha = hashlib.sha256(local_zip_bytes).hexdigest()
    local_size = len(local_zip_bytes)
    
    script_b64 = base64.b64encode(RUN_R2_ONLY_SH.encode()).decode()
    pin_lines = "\n".join(f"{k}={v}" for k, v in pinning.items())
    pin_b64 = base64.b64encode(pin_lines.encode()).decode()
    r1_b64 = base64.b64encode(r1_content.encode()).decode()
    
    seq = pinning["sequence"]
    lt = pinning["locktime"]
    
    new_rentals_lock = threading.Lock()
    failures = []
    failures_lock = threading.Lock()
    
    def setup_one(p):
        o = p["offer"]
        offset = p["offset"]
        machine_label = "REPLACE"
        inst_id = None
        try:
            inst_id = create_instance(o["id"])
            tag = f"[{inst_id}/{machine_label}]"
            print(f"{tag} created from offer {o['id']}")
            
            host, port = wait_for_ssh(inst_id, timeout=900, verbose=False)
            print(f"{tag} ssh ready at {host}:{port}")
            
            r = Rental(
                instance_id=inst_id, ssh_host=host, ssh_port=port,
                num_gpus=o["num_gpus"], gpu_name=o["gpu_name"],
                dollars_per_hr=o["dph_total"], global_offset=offset,
                started=time.time(),
            )
            
            last_err = None
            for attempt in range(1, 13):
                try:
                    ssh_cmd(r, "echo PING", timeout=20)
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(10)
            else:
                raise RuntimeError(f"ssh never became responsive: {last_err}")
            print(f"{tag} ssh responsive")
            
            setup_out = ssh_cmd(r, SETUP_SCRIPT, timeout=300)
            if "SETUP_OK" not in setup_out:
                raise RuntimeError(f"setup failed: {setup_out[-400:]}")
            print(f"{tag} tools installed")
            
            ssh_cmd(r, "mkdir -p /work && cd /work && rm -rf ./*")
            scp_to(r, ZIP_FILE, f"/work/{ZIP_FILE}")
            info = ssh_cmd(r, f"cd /work && stat -c %s {ZIP_FILE} && sha256sum {ZIP_FILE}").strip().splitlines()
            if int(info[0]) != local_size or info[1].split()[0] != local_sha:
                raise RuntimeError(f"zip upload mismatch")
            print(f"{tag} zip uploaded & verified")
            
            ssh_cmd(r, f"cd /work && unzip -o {ZIP_FILE}")
            ssh_cmd(r, "cd /work && "
                       "nvcc -O3 -o qsb_real qsb_real_search.cu -lcrypto -lm && "
                       "nvcc -O3 -o qsb_digest qsb_digest_search.cu -lcrypto -lm && "
                       "nvcc -O3 -o verify_gpu verify_gpu.cu -lcrypto -lm && "
                       "nvcc -O3 -o test_der test_der.cu && "
                       "mkdir -p results logs",
                    timeout=900)
            print(f"{tag} compiled")
            try:
                test_out = ssh_cmd(r, "cd /work && ./test_der", timeout=30).strip()
                if "returned: 1" in test_out:
                    raise RuntimeError("test_der confirms GPU DER bug")
            except Exception as te:
                if "GPU DER bug" in str(te):
                    raise
                print(f"{tag} test_der check failed: {te}")
            
            ssh_cmd(r,
                f"echo {script_b64} | base64 -d > /work/run_r2_only.sh && "
                "chmod +x /work/run_r2_only.sh", timeout=30)
            ssh_cmd(r,
                "mkdir -p /work/results && "
                f"echo {pin_b64} | base64 -d > /work/results/pinning_hit_assigned.txt", timeout=30)
            ssh_cmd(r,
                f"echo {r1_b64} | base64 -d > /work/results/round1_final.txt", timeout=30)
            print(f"{tag} pinning + R1 pre-uploaded")
            
            launch_cmd = f"bash run_r2_only.sh {seq} {lt} {total_gpus} {offset}"
            ssh_bg(r, launch_cmd)
            r.status = "r2_only"
            r.__dict__["pinning"] = pinning
            print(f"{tag} LAUNCHED ({launch_cmd})")
            print(f"{tag} covers firsts: {{{offset}-{offset+o['num_gpus']-1}}} ∪ {{{offset+total_gpus}-{offset+total_gpus+o['num_gpus']-1}}}")
            
            with new_rentals_lock:
                rentals.append(r)
                save_state(rentals, extra)
                
        except Exception as e:
            msg = str(e)[:500]
            print(f"[REPLACE] FAILED: {msg}")
            with failures_lock:
                failures.append((offset, msg))
            if inst_id is not None:
                try:
                    destroy_instance(inst_id)
                    print(f"[{inst_id}/REPLACE] destroyed")
                except Exception:
                    pass
    
    threads = []
    for p in plan:
        t = threading.Thread(target=setup_one, args=(p,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    
    if failures:
        print(f"\n[WARN] {len(failures)} replacement(s) failed.")
    print(f"\n[OK] Replacement attempt complete.")
    print(f"  Run `python3 {sys.argv[0]} status` to check.")


def cmd_add_r2_machine():
    """Add a SECOND machine to the R2 search, with total_gpus=32 and offset=16.
    
    This makes the second machine's GPU 14 IMMEDIATELY start at first=30, where
    the previous (false) hit was reported. If the real R2 hit is anywhere near
    first=30, this machine will find it in ~10 minutes instead of 2.5 hours.
    
    Requires:
    - State file with first R2 machine already running (mode="r2_only")
    - results/r2_only_pinning.txt and results/r2_only_round1.txt as before
    """
    pin_path = Path("results/r2_only_pinning.txt")
    r1_path = Path("results/r2_only_round1.txt")
    if not pin_path.exists() or not r1_path.exists():
        print(f"ERROR: need {pin_path} and {r1_path}")
        return
    
    rentals, extra = load_state()
    if not rentals:
        print("ERROR: no existing rentals. Run rent_r2_only first.")
        return
    if extra.get("mode") != "r2_only":
        existing_mode = extra.get("mode")
        print(f"WARNING: existing mode is '{existing_mode}', not 'r2_only'.")
        ans = input("Continue anyway? [y/N]: ")
        if ans.strip().lower() != "y":
            return

    # Parse pinning + R1
    pinning = {}
    for line in pin_path.read_text().splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            pinning[k.strip()] = v.strip()
    r1_content = r1_path.read_text()
    
    print(f"Adding 2nd R2 machine with total_gpus=32, offset=16")
    print(f"  Pinning: seq={pinning['sequence']} lt={pinning['locktime']}")
    print(f"  Strategy: machine B's GPU 14 starts at first=30 (previous hit area)")
    print(f"  Expected hit time: ~10-30 minutes if hit is near first=30")
    print(f"  Existing machines: {len(rentals)}")

    # Search for an additional machine
    R2_MIN_GPUS = 10
    R2_MAX_PPG = 1.50
    print(f"\nSearching vast.ai for offers with >= {R2_MIN_GPUS} GPUs...")
    offers = search_offers(R2_MIN_GPUS, ACCEPTABLE_GPU_NAMES, max_dollars_per_gpu_hr=R2_MAX_PPG)
    if not offers:
        print("No matching offers found.")
        return

    def speed_factor(gpu_name):
        if "5090" in gpu_name: return 1.5
        if "4090" in gpu_name: return 1.0
        return 1.0
    for o in offers:
        o["eff_gpus"] = o["num_gpus"] * speed_factor(o["gpu_name"])
        o["dph_per_eff"] = o["dph_total"] / o["eff_gpus"]
    # Skip any offers we already rented
    existing_ids = {r.instance_id for r in rentals}
    offers = [o for o in offers if o["id"] not in existing_ids]
    offers_big = sorted(offers, key=lambda x: (-x["eff_gpus"], x["dph_per_eff"]))

    print(f"\nTop offers:")
    for o in offers_big[:5]:
        print(f"  id={o['id']}  {o['gpu_name']} x {o['num_gpus']}  "
              f"eff={o['eff_gpus']:.0f}  ${o['dph_total']:.3f}/hr  Rel={o['reliability']:.2f}")

    selected = offers_big[:1]
    if not selected:
        print("No offers selected.")
        return
    o = selected[0]

    # Need 16 GPUs ideally so that GPU 14 exists. If fewer, the offset trick still works
    # but we'd want to adjust. With total=32, the Nth local GPU handles first=offset+N.
    # So if a machine has 16 GPUs and offset=16, GPU 0..15 handles first=16..31. 
    # GPU 14 handles first=30. ✓
    # If machine has fewer than 15 GPUs, GPU 14 doesn't exist — adjust offset.
    
    # Calculate what offset gives us GPU at first=30
    desired_first = 30
    # Use min(num_gpus, 15) as the local GPU we want to assign first=30 to
    # (we want to use the LAST local GPU so others cover lower firsts too)
    target_local_gpu = min(o["num_gpus"] - 1, 14)
    new_offset = desired_first - target_local_gpu
    new_total = max(32, o["num_gpus"] * 2)  # ensure we don't reuse firsts
    
    # Verify the calculation: GPU `target_local_gpu` will have effective_id = new_offset + target_local_gpu
    # We want this to equal 30
    assert new_offset + target_local_gpu == desired_first
    
    print(f"\nPlan: 1 machine, {o['num_gpus']} GPUs at ${o['dph_total']:.3f}/hr")
    print(f"  total_gpus_override = {new_total}, global_offset = {new_offset}")
    print(f"  Machine B's GPU {target_local_gpu} → first={desired_first} immediately")
    print(f"  Estimated cost if hit found in 30min: ${o['dph_total']*0.5:.2f}")
    ans = input("Proceed? [y/N]: ")
    if ans.strip().lower() != "y":
        print("Aborted.")
        return

    import hashlib, base64

    local_zip_bytes = Path(ZIP_FILE).read_bytes()
    local_sha = hashlib.sha256(local_zip_bytes).hexdigest()
    local_size = len(local_zip_bytes)

    script_b64 = base64.b64encode(RUN_R2_ONLY_SH.encode()).decode()
    pin_lines = "\n".join(f"{k}={v}" for k, v in pinning.items())
    pin_b64 = base64.b64encode(pin_lines.encode()).decode()
    r1_b64 = base64.b64encode(r1_content.encode()).decode()

    inst_id = None
    try:
        inst_id = create_instance(o["id"])
        tag = f"[{inst_id}]"
        print(f"{tag} created from offer {o['id']}")

        host, port = wait_for_ssh(inst_id, timeout=900, verbose=False)
        print(f"{tag} ssh ready at {host}:{port}")

        r = Rental(
            instance_id=inst_id,
            ssh_host=host,
            ssh_port=port,
            num_gpus=o["num_gpus"],
            gpu_name=o["gpu_name"],
            dollars_per_hr=o["dph_total"],
            global_offset=new_offset,
            started=time.time(),
        )

        last_err = None
        for attempt in range(1, 13):
            try:
                ssh_cmd(r, "echo PING", timeout=20)
                break
            except Exception as e:
                last_err = e
                time.sleep(10)
        else:
            raise RuntimeError(f"ssh never became responsive: {last_err}")
        print(f"{tag} ssh responsive")

        setup_out = ssh_cmd(r, SETUP_SCRIPT, timeout=300)
        if "SETUP_OK" not in setup_out:
            raise RuntimeError(f"setup failed: {setup_out[-400:]}")
        print(f"{tag} tools installed")

        ssh_cmd(r, "mkdir -p /work && cd /work && rm -rf ./*")
        scp_to(r, ZIP_FILE, f"/work/{ZIP_FILE}")
        info = ssh_cmd(r, f"cd /work && stat -c %s {ZIP_FILE} && sha256sum {ZIP_FILE}").strip().splitlines()
        if int(info[0]) != local_size or info[1].split()[0] != local_sha:
            raise RuntimeError(f"zip upload mismatch")
        print(f"{tag} zip uploaded & verified")

        ssh_cmd(r, f"cd /work && unzip -o {ZIP_FILE}")
        ssh_cmd(r, "cd /work && "
                   "nvcc -O3 -o qsb_real qsb_real_search.cu -lcrypto -lm && "
                   "nvcc -O3 -o qsb_digest qsb_digest_search.cu -lcrypto -lm && "
                   "nvcc -O3 -o verify_gpu verify_gpu.cu -lcrypto -lm && "
                   "nvcc -O3 -o test_der test_der.cu && "
                   "mkdir -p results logs",
                timeout=900)
        print(f"{tag} compiled")
        # SANITY CHECK
        try:
            test_out = ssh_cmd(r, "cd /work && ./test_der", timeout=30).strip()
            print(f"{tag} test_der output:")
            for line in test_out.splitlines(): print(f"      {line}")
            if "returned: 1" in test_out:
                print(f"{tag} ✗✗✗ GPU is_valid_der WRONGLY accepts non-0x30 hash!")
                print(f"{tag} This confirms the bug. Aborting.")
                raise RuntimeError("test_der confirms GPU DER bug")
        except Exception as te:
            if "GPU DER bug" in str(te):
                raise
            print(f"{tag} test_der check failed: {te}")

        ssh_cmd(r,
            f"echo {script_b64} | base64 -d > /work/run_r2_only.sh && "
            "chmod +x /work/run_r2_only.sh",
            timeout=30)

        ssh_cmd(r,
            "mkdir -p /work/results && "
            f"echo {pin_b64} | base64 -d > /work/results/pinning_hit_assigned.txt",
            timeout=30)
        ssh_cmd(r,
            f"echo {r1_b64} | base64 -d > /work/results/round1_final.txt",
            timeout=30)
        print(f"{tag} pinning + R1 pre-uploaded")

        seq = pinning["sequence"]
        lt = pinning["locktime"]
        # Launch with custom total_gpus and offset
        launch_cmd = f"bash run_r2_only.sh {seq} {lt} {new_total} {new_offset}"
        ssh_bg(r, launch_cmd)
        r.status = "r2_only"
        r.__dict__["pinning"] = pinning
        print(f"{tag} LAUNCHED ({launch_cmd})")
        print(f"{tag} GPU {target_local_gpu} of this machine starts at first={desired_first}")

        rentals.append(r)
        save_state(rentals, extra)  # keep mode="r2_only"
        print(f"\n[OK] Now {len(rentals)} machines running R2-only search.")

    except Exception as e:
        msg = str(e)[:500]
        print(f"FAILED: {msg}")
        if inst_id is not None:
            try:
                destroy_instance(inst_id)
                print(f"[{inst_id}] destroyed")
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["rent", "rent_solo", "rent_r2_only", "rent_r2_fast", "rent_r2_split4", "add_split4_machine", "add_r2_machine", "status", "collect", "inspect", "relaunch_digest", "stop"])
    args = ap.parse_args()
    if args.action == "rent":
        cmd_rent()
    elif args.action == "rent_solo":
        cmd_rent_solo()
    elif args.action == "rent_r2_only":
        cmd_rent_r2_only()
    elif args.action == "rent_r2_fast":
        cmd_rent_r2_fast()
    elif args.action == "rent_r2_split4":
        cmd_rent_r2_split4()
    elif args.action == "add_split4_machine":
        cmd_add_split4_machine()
    elif args.action == "add_r2_machine":
        cmd_add_r2_machine()
    elif args.action == "status":
        cmd_status()
    elif args.action == "collect":
        cmd_collect()
    elif args.action == "inspect":
        cmd_inspect()
    elif args.action == "relaunch_digest":
        cmd_relaunch_digest()
    elif args.action == "stop":
        cmd_stop()


if __name__ == "__main__":
    main()
