# SIMT kernel authoring

## Single-source pipeline

`#[cuda_module]` encloses `#[kernel]` entries and generates host loaders/typed launchers. `cargo oxide build` compiles host code normally and routes device code through Rust MIR → Pliron dialects → LLVM IR → PTX. Plain helpers reachable from a kernel are discovered automatically. `#[device]` is for standalone device compilation roots/device FFI, not a required decoration on every helper.

Begin with `cargo oxide new` or the bundled [starter](../assets/simt-vector-add/src/main.rs). Use the target revision's generated launcher signature. Do not paste a `cuda_std`/NVPTX tutorial into this backend or replace the compiler with plain `cargo build` and assume device artifacts were generated.

## Launch and indexing proof

For elementwise writes, `DisjointSlice<T>` paired with an opaque thread-index witness provides unique mutable element access. Obtain the numeric index before consuming the witness:

```rust
let index = thread::index_1d();
let i = index.get();
if let Some(dst) = output.get_mut(index) {
    *dst = input[i] * alpha;
}
```

The output bounds check does not establish that a shorter input is long enough. Validate relationships on the host. A thread-index witness is kernel-scoped and cannot be copied, cloned, sent, or manufactured from an arbitrary integer.

Declare `#[launch_contract(domain = 1, block = (256, 1, 1))]` when that shape is part of correctness. Use the generated `prepare_<kernel>(LaunchConfig1D::new(...))` to obtain the kernel-branded `PreparedLaunch`, then pass it to the generated safe launch method. Loading a contracted module is still unsafe: prove the embedded artifact was built from the declared module with its ABI and contract. A matching entry name is insufficient.

Preparation checks geometry and resources, not every data-dependent access in the kernel. Reuse a prepared launch only as permitted by its kernel/context contract. `#[launch_bounds]` limits thread count; it does not replace an exact block-shape contract.

Without a contract, launching with raw `cuda_core::simt::LaunchConfig` requires unsafe. Contracted entries expose a raw `<kernel>_unchecked` escape hatch. Do not remove a contract to avoid a preparation error; diagnose geometry, resource, or context mismatch.

`index_1d()` uses only X coordinates. It cannot represent disjoint 2D/3D work by itself. Safe access fails closed for invalid extra axes at this snapshot, but that does not discharge other raw-launch obligations. For 2D data use the documented const-stride or runtime-stride index types and bind the same row width on the host. Prove products/offsets cannot overflow and check byte-versus-element units. Handle `n == 0` before constructing a zero-grid launch; use checked narrowing to CUDA launch integer widths and overflow-safe ceiling division.

## Memory and synchronization

Keep immutable inputs read-only and mutable regions disjoint. For raw pointers or device FFI, document provenance, layout, alignment, extent, aliasing, initialization, and lifetime through completion. `DeviceCopy`/ABI transfer support does not make a host pointer GPU-accessible. `repr(C)` establishes a layout convention, not the complete transfer proof.

Global memory is device storage; shared memory is per-block storage; registers/local values are per-thread. An ordinary Rust reference is not evidence of the intended CUDA address space. Use the stack's memory abstractions and supported conversions.

`SharedArray<T, N, ALIGN>` and `DynamicSharedArray` require explicit synchronization and unsafe access proofs. Initialize every consumed element. Put a block barrier after cooperative writes and before reads, and another before overwriting a tile still in use. Do not return tail threads before a barrier needed by other threads; predicate their memory work while retaining required participation. Warp operations require their documented active masks and participating lanes. A block barrier does not synchronize separate blocks.

Choose atomic ordering and scope from the actual producer/consumer relationship. A relaxed atomic update is not a general publication fence. Grid-wide coordination needs a supported cooperative launch or a separate kernel boundary; do not implement a spin barrier across blocks that may not be concurrently resident.

## Host/device boundary

Use `Result` for host driver/allocation/launch errors. A successful enqueue does not mean the device completed; check synchronization or readback errors. Do not assume panic unwinding, CPU I/O, arbitrary heap allocation, dynamic dispatch, or a CPU standard-library operation is device-supported. Consult `appendix/supported-features.md` and a regression example before introducing one.

Generic kernels and captured closures are supported in specific forms. Prefer small by-value captures for ordinary scalar parameters. Non-move captures can contain host pointers and depend on HMM/accessibility plus extended lifetimes; verify the platform before relying on them. Never fix a lifetime diagnostic by forgetting a GPU future or allocation.

## Build and inspect

From `cuda-oxide/`, upstream examples include:

```bash
cargo oxide run vecadd
cargo oxide inspect vecadd
cargo oxide pipeline vecadd
cargo oxide sanitize vecadd --tool memcheck
```

From a standalone starter, use `cargo oxide run` and the installed subcommand help for selecting that project. Inspect PTX when correctness/performance depends on address spaces, exact block shape, barriers, atomics, register pressure, or spills. Avoid hand-editing generated PTX as a durable Rust-source fix.

For shared memory, warp primitives, clusters, TMA, tensor cores, and layout algebra, continue to [advanced work](advanced.md). For async SIMT use the `cuda_async::simt` surface in [runtime](runtime.md).
