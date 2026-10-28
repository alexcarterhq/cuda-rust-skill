# CUDA Rust skill for Codex and Claude Code

A source-grounded Agent Skill for [NVIDIA CUDA Rust](https://github.com/NVIDIA/cuda-rust): CUDA Oxide SIMT, cuTile Rust, and the shared host runtime. It guides kernel implementation, review, debugging, performance work, interoperability, and compiler contributions.

## Install

### Codex

```bash
npx skills add alexcarterhq/cuda-rust-skill --skill cuda-rust --agent codex --global
```

### Claude Code

```bash
npx skills add alexcarterhq/cuda-rust-skill --skill cuda-rust --agent claude-code --global
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

## Example requests

- “Use /cuda-rust to implement a CUDA Oxide reduction with correct tail handling and a CPU reference.”
- “Use /cuda-rust to port this dense tensor operation to cuTile and explain the partition shapes.”
- “Use /cuda-rust to diagnose a prepared-launch error without disabling the launch contract.”
- “Use /cuda-rust to review this async pipeline for cross-stream races and buffer lifetime bugs.”
- “Use /cuda-rust to measure first-launch and warm execution separately, then tune the bottleneck.”
