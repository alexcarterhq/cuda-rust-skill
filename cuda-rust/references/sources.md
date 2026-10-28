# Sources and maintenance

The bundled instructions are an independent engineering synthesis of NVIDIA's CUDA Rust source and books, inspected on **2026-10-28** at commit **a4a939a9b5f01f3bf712e92386922e1bc7c434a9**. They are not NVIDIA documentation or an NVIDIA-endorsed product.

[upstream.json](upstream.json) records exact source paths, immutable GitHub URLs, and SHA-256 hashes. Use it to open a pinned source directly or compare a local checkout. It records evidence, not a claim that every GPU path was executed. GPU builds, JIT compilation, execution, sanitizers, and performance measurements remain unverified for the bundled starters.

## Primary entry points

- [NVIDIA CUDA Rust repository](https://github.com/NVIDIA/cuda-rust) and [documentation landing page](https://nvidia.github.io/cuda-rust/).
- [CUDA Oxide book](https://nvidia.github.io/cuda-rust/cuda-oxide/): installation, safety, launching, memory, async operations, advanced hardware, compiler stages.
- [cuTile Rust book](https://nvidia.github.io/cuda-rust/cutile/): compatibility, host/device separation, shapes, operations, JIT, tuning, and interop. Published docs can change; resolve against the target commit when they disagree.
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/): architecture, execution/memory model, synchronization, and launch requirements.
- [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html): driver/toolkit compatibility rules.
- [Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/): tool coverage and options. Use documentation matching the installed toolkit.
- [Rust Reference: undefined behavior](https://doc.rust-lang.org/reference/behavior-considered-undefined.html) and [Rustonomicon](https://doc.rust-lang.org/nomicon/): host Rust aliasing and unsafe contracts. They do not replace GPU-specific ordering/address-space rules.

## Resolve a task to source

| Task | Relevant pinned source paths |
|---|---|
| Choose model/workspace/toolchain | Root README, AGENTS, Cargo.toml; all three rust-toolchain.toml files |
| Scaffold or diagnose dependency identity | SIMT `commands/scaffold.rs`, `backend_source.rs`, installation and hello-gpu chapters |
| Resolve toolkit headers and Tile assembler | `cuda-bindings/build.rs`, `cuda-bindings/toolkit_target.rs`, `cutile-rs/cutile-compiler/src/cuda_tile_runtime_utils.rs`; build and JIT resolution differ |
| Safe SIMT launch/indexing | SIMT safety model and launching-kernels chapters |
| Shared memory/barriers | SIMT shared-memory-and-synchronization chapter and the nearest working example |
| New device language feature | SIMT supported-features reference and regression examples |
| Lazy execution/drop/graphs | `cuda-async/README.md`, actual `device_operation`/future/graph source and tests |
| Tile signature/shape ownership | Tile host-vs-device, tensors-and-tiles, and vector-add tutorial |
| GPU/IR version gate | Tile compatibility reference and `cutile-ir/src/requirements.rs` |
| Residual check cost | Tile bounds-check-placement chapter |
| JIT specialization/cache | Tile jit-compilation chapter |
| Debug mode/profiling | Tile debugging-and-profiling chapter |
| Raw kernels/foreign handles | Tile interoperability chapter plus current runtime retention implementation |
| Tune candidates | Tile autotuning chapter; `experimental-tune` feature |
| Intrinsic contribution | SIMT catalog-generated-intrinsics and adding-new-intrinsics chapters |

## Package checks

From the skill repository root, use Python 3.11+ for the validator and test suite:

```bash
python3 tools/validate.py
python3 -m unittest discover -s tests -v
python3 tools/validate.py --upstream /path/to/NVIDIA-cuda-rust
```

The optional upstream check verifies the commit and recorded source hashes, including dirty-tree drift. Tests exercise multiple/incomplete toolkit installations, explicit overrides, native header layouts, CLI exit codes, installation outside the repository, and rejected package mutations. Skill-local links must remain inside the installable `cuda-rust/` directory, and starters must retain their required dependencies and compiler components. These CPU checks do not establish device correctness.

The installed inventory helper needs Python 3.10+. It performs no installation or compilation, disables rustup auto-installation, and prints only selected CUDA-related environment variables. `--strict` exits 2 for missing required probes, an invalid toolkit setting, or a non-Linux host in SIMT/Tile modes. Inspect mode remains informational. A zero exit does not certify version, architecture, or kernel compatibility; upstream `cargo oxide doctor` remains the deeper SIMT diagnostic.

## Refreshing the package

1. Identify the intended new upstream revision and inspect changes to the source paths in the manifest. Read owning repository instructions for any source work.
2. Reconcile version requirements, workspace membership, generated-launch signatures, runtime retention, and compiler flags. Update guidance only for observed changes; distinguish minimum supported versions from tested pins. Recheck the inventory's separate build-header and Tile-assembler discovery rules against their recorded sources, including environment precedence, candidate order, header layouts, and version floors.
3. Update starters and all coupled Git dependency revisions together. Preserve attribution. Resolve dependencies on a supported environment and commit application lockfiles in generated projects after a successful build.
4. Run the package validator/tests; compare recorded hashes against the exact upstream checkout. Refresh hashes only after reviewing the changed sources. A matching commit alone cannot detect a dirty working tree, which is why file hashes are included.
5. On Linux with suitable CUDA hardware, build and execute both starter projects, check all outputs, and run applicable sanitizers. Record the actual machine and toolchain. GPU validation remains unverified until these steps really run.
6. Recheck skill installation paths against the official [Codex skill documentation](https://learn.chatgpt.com/docs/build-skills) and [Claude Code skill documentation](https://code.claude.com/docs/en/skills) when changing distribution instructions.
