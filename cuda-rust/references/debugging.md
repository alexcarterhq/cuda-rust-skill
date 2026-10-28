# Debugging and validation

## Isolate the failing layer

| Symptom | First evidence and likely boundary |
|---|---|
| Wrong Rust/compiler-private API | Owning directory's toolchain, rustup override, backend revision |
| Two nominally identical CUDA types mismatch | `cargo tree -d`, Git versus registry dependency identities |
| bindgen cannot find `stddef.h` | Clang resource headers/libclang selection, toolkit include paths |
| Missing driver symbol/library | Driver installation and dynamic loading, not kernel arithmetic |
| Tile IR rejected by assembler | Selected `tileiras`, emitted IR version, target/operation registry |
| JIT error only on first execution | Captured Tile AST, specialization, shapes, compile options |
| Prepared launch rejected | Exact block/rank, resources, context, cluster/cooperative limits |
| Illegal memory access or trap | First failing kernel, input lengths, offsets, alignment, lifetimes |
| Hang | Barrier participation, active masks, grid residency, stream dependencies |
| Wrong result only after concurrent execution | Missing event/order edge, aliasing, premature reuse |

Retain the first error and exact command, revision, shape/dtype, architecture, and relevant tool versions. Reproduce on a small deterministic input. Synchronize after suspect boundaries during diagnosis to locate asynchronous faults, then remove unnecessary synchronization once dependencies are correct. Sticky device faults can poison the context; rerun the reproducer in a fresh process rather than treating later failures as independent bugs.

## Correctness contract

Compare against a simple CPU implementation or trusted reference before tuning. Use a fixed seed and inputs that expose indexing mistakes rather than only all-ones data. Cover singleton, tail, non-square, and representative large shapes; include empty behavior where supported. Exercise mismatched lengths/shapes, unsupported dtypes/targets, and resource/launch errors at the host boundary.

For floating point, choose tolerances from dtype, accumulation precision, reduction length, and algorithm. Use a documented absolute-plus-relative criterion and handle NaN/Inf explicitly; `abs(actual - expected) > tolerance` alone can accidentally accept NaN. Check all output elements, shape, and finiteness where required. Do not use bit equality for reordered floating-point reductions unless the contract requires it.

For parallel algorithms, check initialized tails, unique write ownership, repeated runs, stream overlap, readback only after completion, and output reuse. Test optimized and baseline paths on the same numerical cases. Miri can check suitable host Rust components, but does not validate CUDA device execution or driver FFI.

## SIMT tools

From `cuda-oxide/`, the upstream example workflows include:

```bash
cargo oxide doctor
cargo oxide pipeline vecadd
cargo oxide inspect vecadd
cargo oxide sanitize vecadd --tool memcheck
cargo oxide debug vecadd --tui
```

Inspect `cargo oxide sanitize --help` for the installed revision before selecting other tools. For a built application, standard Compute Sanitizer commands include:

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 ./target/release/my_app
compute-sanitizer --tool racecheck --error-exitcode 1 ./target/release/my_app
compute-sanitizer --tool synccheck --error-exitcode 1 ./target/release/my_app
compute-sanitizer --tool initcheck --error-exitcode 1 ./target/release/my_app
```

Choose tools relevant to the kernel; shared-memory race checking is not a complete proof against every possible global-memory race. Match toolkit/tool support to the target and avoid running intentionally device-faulting tests in a shared process.

## Tile tools

`CUTILE_DUMP=ir` or `CUTILE_DUMP=bytecode` emits compiler output. `CUTILE_DUMP_FILTER` uses module matching in this snapshot; a bare function name is not a reliable exclusion filter. `print_ir = true` and `dump_mlir_dir = "..."` entry options provide additional inspection. Not every accepted dump-stage name actually emits a stage.

For optimized profiling with source lines, build with `CUDA_RUST_DEBUG=line cargo build --release`. For control-flow debugging, use `CUDA_RUST_DEBUG=full cargo build` and `cuda-gdb`. Accepted values are `none`, `line`, and `full`; this is a **build-time** default captured by `cutile-compiler`, not a way to change an already-built binary by setting a runtime variable.

Any Cargo profile `debug` level currently enables full device debug by default, even line-tables-only; it can force device optimization to zero. Use the explicit line mode for optimized measurement. Per-launch `CompileOptions::debug_info(...)` can override it. Full debug information currently supports locations/inline frames, not arbitrary Rust variable inspection.

For Tile memcheck, use `CompileOptions::new().sanitize_memcheck(true)` where supported, then run under `compute-sanitizer --tool memcheck`. Instrumentation changes the JIT cache key. Host build success is not a substitute for launching each intended specialization.

## Scope the upstream checks

Run commands in their owning workspace and inspect current recipes first:

| Scope | Starting point | Caveat |
|---|---|---|
| Shared host crates | Selected `cargo check -p ...` / targeted tests at root | Some build dependencies need toolkit headers; some tests need a GPU |
| SIMT | `just --list`, then affected recipes in `cuda-oxide/` | `just check` needs toolkit, Python, cargo-deny; not all CI lanes are covered |
| SIMT CPU-only tests | `just test` | Read package/recipe prerequisites |
| SIMT toolkit tests | `just test-cuda` | Toolkit required; no GPU/driver required by this lane |
| Tile/host CPU and assembler suite | `scripts/run_cpu_tests.sh` in `cutile-rs/` | CPU-only can still require headers and `tileiras` |
| Tile GPU suite | `scripts/run_gpu_tests.sh` in `cutile-rs/` | Supported GPU/toolchain required |

Do not run an expensive full LLVM workspace build for a small unrelated change. Run formatting/lints and targeted tests appropriate to edited packages; broaden after a failure or when a cross-cutting change warrants it. Preserve the distinction between passed, failed, skipped, and not run.

Report the target revision, exact checks performed, whether device code was actually compiled/executed, and any unrun GPU/sanitizer cases. For performance work continue to [performance](performance.md).
