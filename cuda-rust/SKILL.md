---
name: cuda-rust
description: Build, review, debug, and optimize GPU software using NVIDIA/cuda-rust, including CUDA Oxide SIMT kernels, cuTile Rust kernels, and the shared cuda-core/cuda-async runtime. Use for cargo oxide, cutile, launch contracts, GPU memory safety, CUDA interoperability, or compiler contributions in this stack.
license: Apache-2.0
metadata:
  author: alexcarterhq
  version: "1.0.0"
---

# CUDA Rust engineering

Produce correct Rust GPU programs against the user's actual NVIDIA CUDA Rust revision. Preserve the selected programming model unless the task calls for a change. This is an early-stage stack: installed source, lockfiles, and toolchain pins take precedence over remembered APIs.

## Establish the target

Inspect the owning project's instructions, manifests, lockfiles, CUDA dependencies, toolchain files, and representative kernels before editing. Record the upstream revision, host platform, GPU compute capability, driver, toolkit, selected assembler, numerical contract, and workload shapes. Discover available facts locally before asking for missing constraints.

Run `python3 <skill-dir>/scripts/doctor.py --mode simt` or `--mode tile` for a read-only inventory; add `--upstream /path/to/cuda-rust` to compare the checkout with the bundled snapshot. It performs no installation or compilation. Treat its `unknown` findings as missing evidence, not proof of support. Resolve paths from this skill's directory, not the current project.

The bundled reference snapshot is `a4a939a9b5f01f3bf712e92386922e1bc7c434a9`, inspected on 2026-10-28. If the target differs, inspect its source and matching docs before using a signature, environment flag, feature gate, or version requirement. Do not upgrade dependencies or float toolchains merely to match this skill.

## Choose the relevant path

| Work | Read next |
|---|---|
| Install, scaffold, resolve toolchains or dependencies | [Setup and compatibility](references/setup.md) |
| Threads, blocks, disjoint slices, launch contracts | [SIMT authoring](references/simt.md) |
| Tensor tiles, shapes, partitions, JIT | [Tile authoring](references/tile.md) |
| Buffers, streams, async operations, graphs, external CUDA | [Runtime and interoperability](references/runtime.md) |
| Memory faults, compiler errors, correctness testing | [Debugging and validation](references/debugging.md) |
| Benchmark, tune, reduce transfers, accelerate GEMM/attention | [Performance engineering](references/performance.md) |
| Compiler, device intrinsics, lowering, CuTe/CTM | [Advanced and compiler work](references/advanced.md) |
| Find the authoritative chapter or refresh this skill | [Sources and maintenance](references/sources.md) and [source manifest](references/upstream.json) |

Choose per kernel: SIMT suits irregular indexing, pointer-rich algorithms, explicit synchronization, and hardware control. Tile suits dense tensor expressions and compiler-managed scheduling. Both can share CUDA streams and device allocations. Use the existing host runtime; importing `cust`, Rust-CUDA's `cuda_std`, rust-gpu, or another similarly named project does not substitute for this stack.

## Implement with explicit invariants

1. State input/output shapes, strides, dtypes, aliasing, tail handling, accumulation precision, and error tolerance. Account for empty inputs and checked integer conversions at the host boundary.
2. Start from the nearest working example at the same revision. For a new standalone project, the bundled [SIMT starter](assets/simt-vector-add/Cargo.toml) and [Tile starter](assets/tile-vector-add/Cargo.toml) pin dependencies to this snapshot; read setup before copying either. Their GPU execution is unverified in this package.
3. Keep host allocation, orchestration, and error propagation outside device code. Verify support before introducing CPU library types, allocation, trait dispatch, closures capturing host references, or device FFI into kernels.
4. Use disjoint mutable outputs. For SIMT, prefer a declared launch contract and prepared launch; a contracted module loader still requires an artifact/ABI proof. Raw launch configurations remain unsafe. For Tile, partition each writable output on the host and keep matching launch grids.
5. Document each unsafe boundary with allocation extent, initialization, alignment, address space, aliasing, lifetime, launch geometry, and synchronization facts that actually apply. Prepared geometry does not prove arbitrary kernel indexing or numerical correctness.
6. Preserve ownership until queued GPU work completes. Rust ownership of a handle alone does not establish device completion or cross-stream ordering. Use stream order or explicit events/dependencies and the runtime's retention rules.
7. Validate correctness before changing scheduling or removing checks. Derive signatures from source instead of guessing around compiler errors or disabling safety contracts.

## Finish with evidence

Use checks from the owning workspace and separate host checks, device compilation, GPU execution, sanitizers, and measurements. A host build is insufficient for a Tile kernel because compilation may be deferred until launch. A sanitizer skip is not a pass. On macOS or a host without the required GPU/toolkit, complete source review and host-safe checks, then report exactly which device checks remain unrun; do not fabricate timings.

For an implementation, deliver the change, source/toolchain target, validation actually performed, and remaining hardware limits. For an optimization, include an unchanged numerical contract and comparable baseline/candidate measurements. For a review, lead with actionable correctness or safety findings and their source locations.
