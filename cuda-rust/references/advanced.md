# Advanced GPU and compiler work

## Choose the level of control

Within SIMT, `cuda-device` exposes device primitives, `cute-layout` expresses layout algebra, `cute-rs` provides typed tensor/copy/pipeline/MMA abstractions, and `ctm-rs` exposes typed instruction-level operations. These differ from `cutile`'s Tile IR programming model. Do not infer API compatibility from the word "tile" or from CUDA C++/Python names.

For advanced kernels inspect these snapshot examples and their README/prerequisites before adapting code:

| Goal | Upstream starting point under `cuda-oxide/crates/rustc-codegen-cuda/examples/` |
|---|---|
| Proof-carrying index/tile views | `proof_carrying_views` |
| Generic kernels and captures | `host_closure` |
| Concurrent pipelines | `async_mlp` |
| Cluster/DSMEM communication | `cluster` |
| Blackwell tensor-memory instructions | `tcgen05` |
| Tuned GEMM | `gemm_sol_final` |
| Device C++/CCCL linkage | `device_ffi_test` |
| MathDx device libraries | `mathdx_ffi_test` |
| Tile/SIMT on one stream | `cutile_inter_kernel` |

## Hardware protocols

Warp collectives require a compatible active-lane mask and participation contract. Shared-memory handoff requires the right barriers and memory ordering. TMA adds tensor-map layout, alignment, address-space, barrier initialization/transaction counts, and completion requirements; do not replace those with an ordinary block barrier by analogy.

Cluster operations additionally require target support, a checked cluster launch shape, compatible grid dimensions, and correct distributed-shared-memory lifetimes. Cooperative grid synchronization requires concurrent residency of the whole cooperating grid. Preparation rejects configurations that cannot satisfy this; never work around the rejection by submitting an unchecked launch without a separate proof.

Matrix instructions require compatible operand formats, tile/layout constraints, accumulation semantics, architecture variants, and synchronization. Features named for Hopper/Blackwell are not portable to every higher SM number. Verify suffix-specific targets and the exact intrinsic requirements. `CUDA_OXIDE_DEVICE_ARCH` accepts the documented `sm_<capability>` form with optional `a` at this snapshot; malformed, `compute_`, and family-`f` hints are errors.

Device LTOIR/FFI requires a matching ABI, toolchain/architecture, symbol identity, layouts, address spaces, and linked libraries. `cargo oxide emit-ltoir` is the documented emission path. Verify the generated artifact and an actual linked kernel; a host symbol resolving is insufficient. Do not assume inter-kernel interoperability implies released intra-kernel Tile/SIMT export support.

## Compiler changes

First identify the failing stage using a minimal reproducer and `cargo oxide pipeline`/artifact inspection:

```text
rustc MIR
  -> mir-importer / dialect-mir
  -> MIR transforms and lowering
  -> LLVM/NVVM/PTX dialects
  -> llvm-export and target finalization
  -> PTX or linked device artifact
```

Read the affected crate's instructions and owning workspace. `rustc-codegen-cuda` has an independent workspace; root tests do not cover it. Avoid moving or duplicating an intrinsic lowering into the wrong stage merely to handle one example. Preserve diagnostics and reject unsupported cases explicitly rather than emitting plausible invalid IR.

For intrinsic additions, the normal path is catalog generation. Inspect the nearest admitted family and its reviewed overlay inputs. Sources include `intrinsics/upstream.lock`, `intrinsics/overlay.toml`, `intrinsics/overlay/*.toml`, the append-only `intrinsics/abi-v1.toml` ledger, and reviewed lowering evidence. `intrinsics/catalog.json` and marked generated Rust files are outputs; do not patch them by hand. Read the catalog-generated-intrinsics chapter for the actual generation/check commands at the target revision.

Use hand-written verification/lowering only when the catalog cannot express the semantics. Carry safety/effects, target gates, expected PTX, ABI identity, and tests across the API, importer, dialect verifier, lowering, and export surfaces. A declaration in LLVM is not evidence that the operation is admitted or supported in this stack.

Test the smallest reproducer, wrong-target/shape/type diagnostics, emitted IR/PTX properties, and GPU correctness where the change needs hardware evidence. Use existing differential/fuzzing infrastructure for compiler transformations when appropriate. A codegen assertion or snapshot comparison alone cannot prove numerical equivalence or synchronization correctness.

For upstream contributions, follow current DCO/SPDX and generated-source rules. Preserve existing authorship/notices and push only to the user-authorized repository/branch. The NVIDIA checkout used to research this skill is source material; it does not grant permission to publish changes there.
