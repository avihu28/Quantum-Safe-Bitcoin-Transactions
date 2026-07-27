#!/usr/bin/env python3
"""test_kernel_static.py — static analysis of CUDA kernels using g++.

Since this test environment doesn't have nvcc, we use a carefully-constructed
stub header (cuda_stub.h) that provides no-op versions of CUDA keywords and
signature declarations for CUDA runtime + OpenSSL, letting g++ parse the .cu
files as C++.

This catches:
  - syntax errors (braces, semicolons, typos)
  - type mismatches on function arguments
  - wrong number of args to a function
  - undefined symbols
  - declaration / definition mismatches
  - parameter list mismatches between __global__ kernel and its call site

It does NOT catch:
  - runtime errors (alignment, illegal memory access, race conditions)
  - correctness of GPU math (tested separately in gpu_emulator.py)
  - NVCC-specific extensions (e.g., __launch_bounds__)

The NVCC compile itself happens at fleet-launch time via the bootstrap script.
If this test passes, the NVCC compile is highly likely to succeed too.
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GPU_DIR = REPO / 'gpu'
STUB_DIR = Path(__file__).resolve().parent / 'static_checks'
STUB_H = STUB_DIR / 'cuda_stub.h'


def preprocess_cu(src: Path, dst: Path):
    """Rewrite a .cu for g++ consumption:
       1. Strip kernel-launch syntax <<<...>>>
       2. Replace CUDA/OpenSSL headers with our stub
    """
    text = src.read_text()
    # Remove <<<...>>> kernel launch triple brackets
    text = re.sub(r'<<<[^>]*>>>', '', text)
    # Replace cuda/openssl headers
    text = re.sub(r'#include\s*<cuda_runtime\.h>', f'#include "{STUB_H}"', text)
    text = re.sub(r'#include\s*<cuda\.h>', '// cuda.h', text)
    text = re.sub(r'#include\s*<openssl/[a-z_]+\.h>', '// openssl', text)
    dst.write_text(text)


def check_kernel(src: Path) -> tuple[bool, str]:
    """Run g++ -fsyntax-only on the preprocessed kernel. Returns (ok, output)."""
    # Write preprocessed file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as tmp:
        preprocess_cu(src, Path(tmp.name))
        tmp_path = tmp.name
    try:
        r = subprocess.run(
            ['g++', '-std=c++14', '-fsyntax-only', '-w',
             '-I', str(STUB_DIR), '-I', str(GPU_DIR),
             tmp_path],
            capture_output=True, text=True, timeout=60,
        )
        return (r.returncode == 0, r.stdout + r.stderr)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main():
    print("═" * 70)
    print("  GPU kernel static-analysis test (g++ with CUDA stubs)")
    print("═" * 70)

    if not STUB_H.exists():
        print(f"ERROR: stub header not found: {STUB_H}")
        return 2

    # Check g++ is available
    if subprocess.run(['which', 'g++'], capture_output=True).returncode != 0:
        print("SKIP: g++ not available in this environment")
        print("(On a real CUDA host, nvcc compiles kernels via fleet bootstrap)")
        return 0

    results = {}
    for kernel in ['qsb_digest_search.cu', 'qsb_real_search.cu']:
        src = GPU_DIR / kernel
        if not src.exists():
            print(f"\n  ✗ {kernel} — not found at {src}")
            results[kernel] = False
            continue
        print(f"\n  checking {kernel}...")
        ok, out = check_kernel(src)
        if ok:
            print(f"  ✓ {kernel} parses cleanly")
        else:
            print(f"  ✗ {kernel} has errors:")
            # Show the first few error lines
            for line in out.split('\n')[:20]:
                print(f"      {line}")
        results[kernel] = ok

    print()
    print("═" * 70)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'}  {name}")
    all_ok = all(results.values())
    print("═" * 70)
    if all_ok:
        print("📝 Note: this is static analysis only. Actual nvcc compile happens")
        print("    on each vast.ai instance via the fleet bootstrap script.")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
