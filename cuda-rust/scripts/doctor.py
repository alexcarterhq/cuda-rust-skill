#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only CUDA Rust inventory. Python 3.10+, no third-party packages."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


ENV_KEYS = (
    "CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUTILE_TILEIRAS_PATH",
    "CUTILE_BYTECODE_VERSION", "CUDA_OXIDE_DEVICE_ARCH", "CUDA_RUST_DEBUG",
)
STANDARD_TOOLKITS = (
    "/usr/local/cuda-13.4", "/usr/local/cuda-13.3", "/usr/local/cuda-13.2",
    "/usr/local/cuda-13", "/usr/local/cuda", "/opt/cuda",
)


def probe(argv, timeout=4):
    """Run a fixed argv without a shell or rustup auto-installation."""
    env = dict(os.environ, RUSTUP_AUTO_INSTALL="0", RUSTUP_NO_UPDATE_CHECK="1")
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, errors="replace", timeout=timeout,
            check=False, env=env, stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return {"status": "missing", "command": argv}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "command": argv}
    except OSError as error:
        return {"status": "error", "command": argv, "output": str(error)}
    output = (result.stdout + result.stderr).strip()
    return {
        "status": "available" if result.returncode == 0 else "error",
        "command": argv, "returncode": result.returncode,
        "output": output[:8192], "truncated": len(output) > 8192,
    }


def select_toolkit(env):
    for key in ("CUDA_TOOLKIT_PATH", "CUDA_HOME"):
        if env.get(key):
            return Path(env[key]).expanduser()
    return next((Path(p) for p in STANDARD_TOOLKITS if Path(p).is_dir()), None)


def select_tileiras(env, toolkit):
    if env.get("CUTILE_TILEIRAS_PATH"):
        return str(Path(env["CUTILE_TILEIRAS_PATH"]).expanduser())
    if toolkit is not None:
        return str(toolkit / "bin" / "tileiras")
    return shutil.which("tileiras") or "tileiras"


def first_tool(names):
    return next((found for name in names if (found := shutil.which(name))), names[0])


def inventory(mode="inspect", upstream=None, timeout=4):
    snapshot = json.loads(
        (Path(__file__).resolve().parents[1] / "references" / "upstream.json").read_text()
    )
    toolkit = select_toolkit(os.environ)
    nvcc = str(toolkit / "bin" / "nvcc") if toolkit else "nvcc"
    commands = {
        "rustc": ["rustc", "-Vv"],
        "cargo": ["cargo", "--version"],
        "rustup": ["rustup", "toolchain", "list"],
        "gpu": ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
        "toolkit": [nvcc, "--version"],
        "tileiras": [select_tileiras(os.environ, toolkit), "--version"],
        "clang": [first_tool(("clang-21", "clang-22", "clang")), "--version"],
        "llc": [first_tool(("llc-23", "llc-22", "llc-21", "llc")), "--version"],
        "sanitizer": ["compute-sanitizer", "--version"],
        "nsight_compute": ["ncu", "--version"],
        "nsight_systems": ["nsys", "--version"],
    }
    checks = {name: probe(argv, timeout) for name, argv in commands.items()}
    required = ["rustc", "cargo", "gpu", "toolkit"] if mode != "inspect" else []
    if mode == "tile":
        required.append("tileiras")
    if mode == "simt":
        required.append("clang")
    gaps = [name for name in required if checks[name]["status"] != "available"]
    notes = [
        "Inventory only: availability does not certify version, architecture, ABI, or kernel support.",
        "nvidia-smi driver capability is separate from the installed toolkit version.",
        "llc inventory covers PATH tools, not every toolchain-private LLVM binary.",
        "cargo oxide doctor is intentionally not executed; it may build or resolve dependencies.",
    ]
    if platform.system() != "Linux":
        notes.append("Upstream documents Linux execution; this host cannot establish Linux CUDA readiness.")
    report = {
        "mode": mode,
        "host": {"system": platform.system(), "machine": platform.machine()},
        "snapshot_commit": snapshot["commit"],
        "snapshot_toolchains": snapshot["toolchains"],
        "environment": {key: os.environ[key] for key in ENV_KEYS if key in os.environ},
        "selected_toolkit": str(toolkit) if toolkit else None,
        "checks": checks, "required_probe_gaps": gaps, "notes": notes,
        "gpu_validation": "not run",
    }
    if upstream is not None:
        checkout = Path(upstream).expanduser().resolve()
        revision = probe(["git", "-C", str(checkout), "rev-parse", "HEAD"], timeout)
        report["upstream"] = {
            "path": str(checkout), "revision": revision,
            "matches_snapshot": revision.get("output") == snapshot["commit"]
            if revision["status"] == "available" else None,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "simt", "tile"), default="inspect")
    parser.add_argument("--upstream", help="Optional local NVIDIA/cuda-rust checkout")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output")
    parser.add_argument("--strict", action="store_true", help="Exit 2 for missing required probes or non-Linux execution modes")
    parser.add_argument("--timeout", type=float, default=4, help="Seconds per probe (0 < value <= 30)")
    args = parser.parse_args()
    if not 0 < args.timeout <= 30:
        parser.error("--timeout must be greater than 0 and at most 30")
    report = inventory(args.mode, args.upstream, args.timeout)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"CUDA Rust inventory: {report['host']['system']} / {report['host']['machine']}")
        print(f"Reference commit: {report['snapshot_commit']}")
        for name, check in report["checks"].items():
            first_line = check.get("output", "").splitlines()
            print(f"{name}: {check['status']}" + (f" — {first_line[0]}" if first_line else ""))
        if "upstream" in report:
            print(f"Checkout matches snapshot: {report['upstream']['matches_snapshot']}")
        for note in report["notes"]:
            print(f"Note: {note}")
        print("GPU validation: not run")
    incomplete = report["required_probe_gaps"] or (
        args.mode != "inspect" and report["host"]["system"] != "Linux"
    )
    return 2 if args.strict and incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
