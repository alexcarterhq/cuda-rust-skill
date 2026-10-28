# Setup and compatibility

## Snapshot and workspace selection

These facts describe upstream commit `a4a939a9b5f01f3bf712e92386922e1bc7c434a9`. Inspect the target checkout when versions differ.

| Working directory | Members / purpose | Pinned Rust |
|---|---|---|
| Repository root | `cuda-bindings`, `cuda-core`, `cuda-core-derive`, `cuda-async` | 1.98.0 |
| `cuda-oxide/` | SIMT tools and device/compiler crates | 1.99.1 |
| `cutile-rs/` | Tile compiler, IR, macros, library, examples | 1.98.0 |

The kernel workspaces are excluded from the root workspace. The codegen backend and many SIMT examples also have independent workspaces. Root `cargo test --workspace` does not cover them. There is no root Justfile.

Change into the owning workspace before running Cargo. `--manifest-path` does not select a toolchain by changing the working directory. A rustup override can supersede a pin; inspect it when the compiler version is surprising. Do not run repository-wide GPU tests on a CPU-only worker and interpret the failures as compilation regressions.

## Prerequisites

The documented execution platform is Linux, tested on Ubuntu 24.04. Do not promise native macOS CUDA execution. An ARM Linux host with a supported NVIDIA GPU is a different target from Apple Silicon macOS.

SIMT needs its pinned Rust toolchain including `rust-src`, `rustc-dev`, and `llvm-tools`, CUDA Toolkit 13.0+ including cuRAND headers, a suitable CUDA 13.x driver (the README names R580+), and working Clang/libclang development headers for bindgen. `stddef.h` failures often mean Clang resource headers are missing, even if a libclang shared object exists. LLVM/NVPTX requirements depend on the instructions emitted; the README recommends LLVM 21+ for advanced Hopper/Blackwell paths. Do not infer support merely because a basic kernel compiled with an older tool.

The Tile README gives a language minimum of stable Rust 1.89+, but the repository's CI pin is 1.98.0. Distinguish minimum language support from the tested checkout toolchain. CUDA 13.3 is recommended at this snapshot. Tile compatibility depends on the **emitted Tile IR version**, assembler capabilities, target architecture, and operations, rather than a simple toolkit/driver version comparison.

| Tile target | Minimum emitted Tile IR |
|---|---|
| `sm_80`, `sm_86`, `sm_87`, `sm_88`, `sm_89` | 13.2 |
| `sm_90` | 13.3 |
| `sm_100`, `sm_103`, `sm_110`, `sm_120`, `sm_121` | 13.2 |
| `sm_107` | 13.4 |

Targets below `sm_80` are unsupported by this Tile snapshot. Do not turn this table into a blanket claim about SIMT support. FP4 and scaled matrix operations impose additional IR/type/architecture constraints. Architecture numbers are not a monotonic feature hierarchy. Consult `cutile-rs/cutile-ir/src/requirements.rs` and the compatibility reference for the actual operation.

`nvidia-smi`'s CUDA field describes driver capability; it is not the installed toolkit version. Inspect `nvcc --version` separately. Inspect the selected `tileiras`, not an unrelated copy found first on PATH. Minor-version driver compatibility follows NVIDIA's deployment documentation; avoid inventing a requirement that the driver's reported CUDA version must equal the toolkit.

## Install or scaffold SIMT

On a prepared Linux machine, install the helper at the same source revision as the device crates:

```bash
cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git \
  --rev a4a939a9b5f01f3bf712e92386922e1bc7c434a9 cargo-oxide
cargo oxide new my_kernel
cd my_kernel
cargo oxide doctor
cargo oxide run
```

These are setup/build actions and may download or compile dependencies. The bundled `doctor.py` only inventories; upstream `cargo oxide doctor` is the deeper project-specific check. Consult `cargo oxide --help` at the installed revision when flags differ.

The generated scaffold sets `RUSTC_BOOTSTRAP=1` locally for compiler-private features. This is a stable-pinned custom-backend project, not a reason to force the whole application onto arbitrary nightly. Keep this setting scoped to the relevant project.

At this snapshot, the backend follows the checkout Cargo resolves for the CUDA device crates. Keep `cuda-device`, `cuda-host`, `cuda-core`, and any `cuda-async` dependency on one Git URL and revision, or one coherent path checkout. A crates.io copy plus a Git copy can produce distinct `CudaContext`/`DeviceBuffer` types. Diagnose with `cargo tree -d` and `cargo metadata`; do not paper over it with unsafe casts. Commit an application lockfile after successful dependency resolution.

The bundled `assets/simt-vector-add/` is a small pinned alternative to scaffolding. Copy the entire directory including `.cargo/config.toml`, then run `cargo oxide run` there. Its non-multiple-of-block input exercises tail handling; it skips a zero-size launch on the host.

## Configure and exercise Tile

From a CUDA Rust checkout:

```bash
cd cutile-rs
cargo run -p cutile-examples --example hello_world
cargo run -p cutile-examples --example saxpy
```

For a standalone project, copy `assets/tile-vector-add/` and run `cargo run --release` in that directory. Its conservative shape is divisible by the tile; extend shape coverage before production use. Host compilation alone does not exercise JIT or GPU execution.

Build-header discovery checks `CUDA_TOOLKIT_PATH`, then `CUDA_HOME`; an explicitly empty or invalid value fails instead of falling back. With neither variable set, it probes standard locations for CUDA 13.0+ headers. `CUDA_TOOLKIT_TARGET_DIR` selects an exclusive `targets/<dir>/include` tree. Without that override, upstream checks top-level headers followed by layouts for Cargo's build target. The lightweight inventory defaults to its own host and does not inspect Cargo cross-target configuration.

Tile JIT assembler discovery is separate: a nonempty `CUTILE_TILEIRAS_PATH` selects that binary directly. Otherwise it tries `bin/tileiras` under the first nonempty toolkit variable, then standard installations with CUDA 13.2+ headers and `tileiras`, then PATH. A missing assembler under an explicit toolkit can therefore lead to a different installation; a missing explicit `CUTILE_TILEIRAS_PATH` does not fall back. Inspect the actual probe command in the inventory JSON. Environment path values are literal; use an expanded absolute path rather than a quoted `~`.

`CUTILE_BYTECODE_VERSION` requests a particular supported IR version; an unsupported override fails instead of silently downgrading. A newer driver cannot teach an old assembler a new bytecode format. Prefer a coherent toolkit/assembler installation.

Nix shells live in the component directories. Inspect their actual definitions before relying on a README transcript's Rust version. `cutile-rs/cuda-tile-rs` is omitted from default members because building it downloads and compiles LLVM; `--workspace` includes it unless excluded. Avoid triggering that cost for an unrelated library change.

## Sources

See the pinned paths in [upstream.json](upstream.json): root and component toolchains, root AGENTS, SIMT installation/scaffold/backend source, Tile README and compatibility reference. For driver rules use [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
