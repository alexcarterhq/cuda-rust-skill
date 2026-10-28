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
    "CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_TOOLKIT_TARGET_DIR", "CUTILE_TILEIRAS_PATH",
    "CUTILE_BYTECODE_VERSION", "CUDA_OXIDE_DEVICE_ARCH", "CUDA_RUST_DEBUG",
)
STANDARD_TOOLKITS = (
    "/usr/local/cuda-13.3", "/usr/local/cuda-13.2",
    "/usr/local/cuda-13", "/usr/local/cuda", "/opt/cuda",
)
# The build's header discovery and the Tile JIT's assembler discovery have
# different candidate lists and version floors at the recorded revision.
TILEIRAS_TOOLKITS = ("/usr/local/cuda-13.4", *STANDARD_TOOLKITS)


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


def toolkit_version(root, target_dir=None):
    """Read native-platform cuda.h, mirroring the recorded upstream sources."""
    includes = [root / "include"]
    if target_dir and target_dir.strip():
        includes = [root / "targets" / target_dir / "include"]
    elif platform.system() == "Linux":
        targets = {
            "x86_64": ("x86_64-linux",),
            "aarch64": ("sbsa-linux", "aarch64-linux"),
            "arm64": ("sbsa-linux", "aarch64-linux"),
        }.get(platform.machine(), ())
        includes.extend(root / "targets" / target / "include" for target in targets)
    for include in includes:
        header = include / "cuda.h"
        if not header.is_file():
            continue
        try:
            for line in header.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[:2] == ["#define", "CUDA_VERSION"]:
                    value = parts[2]
                    if value.isascii() and value.isdecimal() and int(value) <= 0xffffffff:
                        return int(value)
        except (OSError, UnicodeError) as error:
            raise ValueError(f"Cannot read {header}: {error}") from error
        raise ValueError(f"No valid CUDA_VERSION in {header}")
    raise ValueError(f"No cuda.h found in {', '.join(str(p) for p in includes)}")


def select_toolkit(env):
    """Resolve build headers; explicit invalid settings must not fall back."""
    for key in ("CUDA_TOOLKIT_PATH", "CUDA_HOME"):
        if key not in env:
            continue
        if not env[key]:
            raise ValueError(f"{key} is set to an empty string")
        root = Path(env[key])
        try:
            if toolkit_version(root, env.get("CUDA_TOOLKIT_TARGET_DIR")) < 13000:
                raise ValueError("CUDA headers must be 13.0 or newer")
        except ValueError as error:
            raise ValueError(f"{key}={env[key]} is invalid: {error}") from error
        return root
    for candidate in STANDARD_TOOLKITS:
        root = Path(candidate)
        try:
            if toolkit_version(root, env.get("CUDA_TOOLKIT_TARGET_DIR")) >= 13000:
                return root
        except ValueError:
            continue
    return None


def select_tileiras(env):
    """Resolve the JIT assembler independently of the build's CUDA headers."""
    if env.get("CUTILE_TILEIRAS_PATH"):
        return env["CUTILE_TILEIRAS_PATH"]
    for key in ("CUDA_TOOLKIT_PATH", "CUDA_HOME"):
        if env.get(key):
            binary = Path(env[key]) / "bin" / "tileiras"
            if binary.is_file():
                return str(binary)
            # Upstream uses only the first nonempty toolkit variable, then
            # default installations, rather than trying the second variable.
            break
    for candidate in TILEIRAS_TOOLKITS:
        root = Path(candidate)
        try:
            if toolkit_version(root) >= 13020 and (root / "bin/tileiras").is_file():
                return str(root / "bin/tileiras")
        except ValueError:
            continue
    return shutil.which("tileiras") or "tileiras"


def first_tool(names):
    return next((found for name in names if (found := shutil.which(name))), names[0])


def inventory(mode="inspect", upstream=None, timeout=4):
    snapshot = json.loads(
        (Path(__file__).resolve().parents[1] / "references" / "upstream.json").read_text()
    )
    toolkit = None
    toolkit_error = None
    try:
        toolkit = select_toolkit(os.environ)
    except ValueError as error:
        toolkit_error = str(error)
    commands = {
        "rustc": ["rustc", "-Vv"],
        "cargo": ["cargo", "--version"],
        "rustup": ["rustup", "toolchain", "list"],
        "gpu": ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
        "tileiras": [select_tileiras(os.environ), "--version"],
        "clang": [first_tool(("clang-21", "clang-22", "clang")), "--version"],
        "llc": [first_tool(("llc-23", "llc-22", "llc-21", "llc")), "--version"],
        "sanitizer": ["compute-sanitizer", "--version"],
        "nsight_compute": ["ncu", "--version"],
        "nsight_systems": ["nsys", "--version"],
    }
    checks = {name: probe(argv, timeout) for name, argv in commands.items()}
    checks["toolkit"] = probe([str(toolkit / "bin/nvcc"), "--version"], timeout) if toolkit else {
        "status": "error" if toolkit_error else "missing",
        "output": toolkit_error or "No CUDA 13.0+ headers found in default toolkit locations",
    }
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
        "Toolkit headers and Tile JIT assembler are resolved separately using the recorded snapshot's Linux rules.",
        "Header discovery targets this host unless CUDA_TOOLKIT_TARGET_DIR is set; Cargo cross-target configuration is not inspected.",
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
