#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate this skill package without network, CUDA, or third-party modules."""

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tomllib
from urllib.parse import unquote, urlsplit


REPO = Path(__file__).resolve().parents[1]
UPSTREAM_URL = "https://github.com/NVIDIA/cuda-rust"


def validate(root=REPO, upstream=None):
    root = Path(root).resolve()
    skill = root / "cuda-rust"
    errors = []
    for required in (
        "README.md", "LICENSE", "NOTICE", "cuda-rust/SKILL.md",
        "cuda-rust/agents/openai.yaml", "cuda-rust/scripts/doctor.py",
        "cuda-rust/references/upstream.json",
    ):
        if not (root / required).is_file():
            errors.append(f"Missing required file: {required}")
    if errors:
        return errors

    entry = (skill / "SKILL.md").read_text()
    frontmatter = re.match(r"\A---\n(.*?)\n---\n", entry, re.S)
    if not frontmatter:
        errors.append("SKILL.md needs YAML frontmatter")
    else:
        header = frontmatter.group(1)
        if not re.search(r"^name: cuda-rust$", header, re.M):
            errors.append("Skill name must match the install directory: cuda-rust")
        description = re.search(r"^description: (.+)$", header, re.M)
        if not description or len(description.group(1)) > 1024:
            errors.append("Need a nonempty description of at most 1024 characters")
    if len(entry.splitlines()) > 500:
        errors.append("Entrypoint exceeds 500 lines; move task-specific detail to references")
    if "[TODO" in entry:
        errors.append("Unfinished skill initializer placeholder")

    for markdown in root.rglob("*.md"):
        if any(part in (".git", "target") for part in markdown.parts) or markdown.name.startswith("._"):
            continue
        for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", markdown.read_text()):
            target = target.strip("<>")
            if urlsplit(target).scheme or target.startswith("#"):
                continue
            linked = (markdown.parent / unquote(target.split("#", 1)[0])).resolve()
            if not linked.is_relative_to(root):
                errors.append(f"Link escapes package: {markdown.relative_to(root)} -> {target}")
            elif not linked.exists():
                errors.append(f"Broken local link: {markdown.relative_to(root)} -> {target}")

    for script in root.rglob("*.py"):
        if any(part in (".git", "target") for part in script.parts) or script.name.startswith("._"):
            continue
        try:
            ast.parse(script.read_text(), filename=str(script))
        except SyntaxError as error:
            errors.append(f"Python syntax error: {error}")

    try:
        snapshot = json.loads((skill / "references/upstream.json").read_text())
        commit = snapshot["commit"]
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            errors.append("Upstream commit must be a full SHA-1 revision")
        if snapshot["repository"] != UPSTREAM_URL:
            errors.append("Unexpected upstream repository")
        if not snapshot["sources"]:
            errors.append("Source manifest is empty")
        for mode in ("simt", "tile"):
            folder = skill / "assets" / f"{mode}-vector-add"
            manifest = tomllib.loads((folder / "Cargo.toml").read_text())
            toolchain = tomllib.loads((folder / "rust-toolchain.toml").read_text())
            if toolchain["toolchain"]["channel"] != snapshot["toolchains"][mode]:
                errors.append(f"{mode} starter toolchain differs from the source snapshot")
            for name, dependency in manifest["dependencies"].items():
                if dependency.get("git") != UPSTREAM_URL + ".git" or dependency.get("rev") != commit:
                    errors.append(f"{mode}/{name}: dependency must use the same pinned upstream revision")
            if manifest["package"].get("publish") is not False:
                errors.append(f"{mode}: starter must not be accidentally publishable to crates.io")
            if not (folder / "src/main.rs").is_file():
                errors.append(f"{mode}: missing starter source")
        config = tomllib.loads((skill / "assets/simt-vector-add/.cargo/config.toml").read_text())
        if config.get("env", {}).get("RUSTC_BOOTSTRAP") != "1":
            errors.append("SIMT compiler-private configuration is missing")

        seen = set()
        for source in snapshot["sources"]:
            relative = source["path"]
            parsed = PurePosixPath(relative)
            if parsed.is_absolute() or ".." in parsed.parts or "\\" in relative or not relative:
                errors.append(f"Unsafe upstream source path: {relative}")
                continue
            if relative in seen:
                errors.append(f"Duplicate upstream source path: {relative}")
            seen.add(relative)
            if not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
                errors.append(f"Invalid source hash: {relative}")
            if source["url"] != f"{UPSTREAM_URL}/blob/{commit}/{relative}":
                errors.append(f"Source URL is not pinned to manifest commit: {relative}")
            if upstream:
                checkout = Path(upstream).resolve()
                candidate = (checkout / relative).resolve()
                if not candidate.is_relative_to(checkout) or not candidate.is_file():
                    errors.append(f"Missing or external upstream source: {relative}")
                elif hashlib.sha256(candidate.read_bytes()).hexdigest() != source["sha256"]:
                    errors.append(f"Upstream source drift: {relative}")
        if upstream:
            revision = subprocess.run(
                ["git", "-C", str(Path(upstream).resolve()), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if revision.returncode or revision.stdout.strip() != commit:
                errors.append("Upstream checkout revision differs from snapshot or is not a Git checkout")
    except (KeyError, TypeError, AttributeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        errors.append(f"Invalid or incomplete package metadata: {error}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, help="Also verify exact checkout revision and source hashes")
    args = parser.parse_args()
    errors = validate(upstream=args.upstream)
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print("PASS: skill structure, local links, Python syntax, starter TOML/pins, and source manifest")
    if args.upstream:
        print("PASS: upstream commit and all recorded SHA-256 source hashes")
    print("GPU build, JIT, execution, sanitizers, and performance: NOT RUN by this validator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
