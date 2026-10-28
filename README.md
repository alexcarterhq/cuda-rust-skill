# CUDA Rust skill for Codex and Claude Code

A source-grounded Agent Skill for [NVIDIA CUDA Rust](https://github.com/NVIDIA/cuda-rust): CUDA Oxide SIMT, cuTile Rust, and the shared host runtime. It guides kernel implementation, review, debugging, performance work, interoperability, and compiler contributions.

The skill targets upstream commit [`a4a939a9b5f0`](https://github.com/NVIDIA/cuda-rust/tree/a4a939a9b5f01f3bf712e92386922e1bc7c434a9), inspected on 2026-10-28. It tells the agent to inspect your actual revision before applying an API or version requirement. Independent project; not affiliated with or endorsed by NVIDIA.

## Install

Install for both Codex and Claude Code across your projects with [the skills CLI](https://github.com/vercel-labs/skills):

```bash
npx skills add alexcarterhq/cuda-rust-skill --skill cuda-rust --agent codex claude-code --global
```

Requires Node.js 22.20+ and Git. The command installs the complete skill, including its references, scripts, and starter projects. To install for just one agent, use `--agent codex` or `--agent claude-code`. Omit `--global` to install into the current project. Add `--copy` to copy files instead of using the CLI's default symlinks, or `--yes` to skip its confirmation prompts.

Preview the available skill without installing:

```bash
npx skills add alexcarterhq/cuda-rust-skill --list
```

### Manual installation

Alternatively, clone this repository, then install the **`cuda-rust/` subdirectory**, which contains `SKILL.md` and all supporting resources:

```bash
git clone https://github.com/alexcarterhq/cuda-rust-skill.git
cd cuda-rust-skill
```

For Codex personal discovery:

```bash
mkdir -p ~/.agents/skills
test ! -e ~/.agents/skills/cuda-rust && cp -R cuda-rust ~/.agents/skills/cuda-rust
```

For Claude Code personal discovery:

```bash
mkdir -p ~/.claude/skills
test ! -e ~/.claude/skills/cuda-rust && cp -R cuda-rust ~/.claude/skills/cuda-rust
```

These copy commands deliberately do nothing if a destination already exists. Review and replace an existing installation explicitly when updating. For team use, copy the same directory into your project's `.agents/skills/cuda-rust/` for Codex or `.claude/skills/cuda-rust/` for Claude Code. Start a new session if it does not appear. See the official [Codex](https://learn.chatgpt.com/docs/build-skills) and [Claude Code](https://code.claude.com/docs/en/skills) documentation for discovery details.

Invoke with `$cuda-rust` in Codex or `/cuda-rust` in Claude Code. Automatic discovery is also enabled. No MCP server, third-party Python package, or other skill is required. The skill itself runs on any host; **CUDA execution needs a supported Linux/NVIDIA environment**.

## Example requests

- “Use $cuda-rust to implement a CUDA Oxide reduction with correct tail handling and a CPU reference.”
- “Use $cuda-rust to port this dense tensor operation to cuTile and explain the partition shapes.”
- “Use $cuda-rust to diagnose a prepared-launch error without disabling the launch contract.”
- “Use $cuda-rust to review this async pipeline for cross-stream races and buffer lifetime bugs.”
- “Use $cuda-rust to measure first-launch and warm execution separately, then tune the bottleneck.”

For Claude Code, replace `$cuda-rust` with `/cuda-rust`.

## Included

- A concise [skill entry point](cuda-rust/SKILL.md) with task-specific reference routing.
- Eight focused references: setup, SIMT, Tile, runtime/interop, debugging, performance, advanced/compiler work, and sources.
- A [read-only environment inventory](cuda-rust/scripts/doctor.py), with human or JSON output and optional upstream revision comparison.
- [SIMT](cuda-rust/assets/simt-vector-add/) and [Tile](cuda-rust/assets/tile-vector-add/) vector-add starter projects with pinned Git dependencies and full-output checks.
- A [source manifest](cuda-rust/references/upstream.json) with immutable URLs and file hashes, package validation, helper tests, and CPU-only CI.

The entry point loads shared constraints first; the agent opens only references needed for the task. No broad tool preapproval, automatic installation, or hidden network request is embedded in the skill.

## Environment inventory

Python 3.10+ is sufficient for the inventory script:

```bash
python3 cuda-rust/scripts/doctor.py --mode simt
python3 cuda-rust/scripts/doctor.py --mode tile --json
python3 cuda-rust/scripts/doctor.py --upstream /path/to/NVIDIA-cuda-rust
```

The script probes versions/availability without building or installing software. It disables rustup auto-installation for its probes. It prints only selected CUDA-related environment variables, not your full environment. `--strict` exits 2 for missing required probes or a non-Linux execution target; a zero exit does **not** certify version/architecture compatibility. Upstream `cargo oxide doctor` remains the deeper SIMT-specific diagnostic.

## Starter projects and validation limits

Copy a starter directory to your project location, including the SIMT starter's hidden `.cargo` folder. Follow [setup](cuda-rust/references/setup.md). Run SIMT with `cargo oxide run`; run Tile with `cargo run --release`. CUDA/Rust dependencies are pinned to the inspected upstream revision. Generate and commit your application's lockfile after successful dependency resolution.

The SIMT starter covers singleton and block-boundary tails and explicitly skips empty work. The Tile starter verifies several divisible shapes with index-varying data; define a tail/empty policy before accepting arbitrary shapes.

**GPU builds, JIT compilation, execution, sanitizers, and performance measurements have not been run for these starters.** Package checks cover structure, local links, TOML/revision consistency, helper behavior, source hashes, and Rust parsing/formatting; they do not establish device correctness. This package was authored on macOS without a CUDA execution environment. Public CI intentionally validates the skill package on a CPU worker rather than claiming GPU coverage.

## Maintain and test

Python 3.11+ is needed for the package validator's standard-library TOML parser:

```bash
python3 tools/validate.py
python3 -m unittest discover -s tests -v
python3 tools/validate.py --upstream /path/to/NVIDIA-cuda-rust
```

The optional upstream check verifies both the commit and every recorded source hash, including dirty-tree drift. See [sources and maintenance](cuda-rust/references/sources.md) before refreshing a revision. Existing projects keep their own source/toolchain choices.

## License

[Apache-2.0](LICENSE). Starter patterns adapted from NVIDIA CUDA Rust retain attribution in [NOTICE](NOTICE) and their source headers. The skill prose and helper scripts are independently authored. Copies of the license and notices are included inside `cuda-rust/` so they travel with npx and manual installations.
