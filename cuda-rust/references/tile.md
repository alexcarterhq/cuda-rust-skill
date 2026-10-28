# Tile kernel authoring

## Model and signatures

cuTile kernels are tensor/tile programs, not a manual thread-index DSL. `#[cutile::module]` captures an inline module; `#[cutile::entry()]` marks a launcher entry. Helpers inside the module are device functions. GPU compilation normally occurs at the first execution of a specialization, after the Rust application has already built.

Host `Tensor<T>` owns runtime-shaped device storage. Device `Tensor<E, S>` describes a memory view; `Tile<E, S>` is an immutable compute value with a static shape. Do not interchange host and device types simply because their names match. Kernels write outputs through tensor parameters rather than returning host values.

The [starter](../assets/tile-vector-add/src/main.rs) uses a static output tile `Tensor<f32, {[B]}>` and dynamic full input shapes `Tensor<f32, {[-1]}>`. `load_like(output)` uses the output's coordinates and tile shape. Arithmetic creates tiles; `output.store(tile)` writes the result.

## Ownership, partitioning, and shapes

Partition every writable output on the host before launch. An owned tensor gives `Partition<Tensor<T>>`; `(&mut tensor).partition(shape)` gives a borrowed partition for in-place output. Immutable inputs may be owned, borrowed, or shared with `Arc`. `Arc` alone does not grant concurrent mutable access.

Output partitions infer the logical tile-block grid. Multiple writable outputs must have compatible inferred grids. A tile block may map to many physical threads; never report logical tile blocks as CUDA thread counts. Read-only tensors may be partitioned differently inside the kernel, as needed for GEMM or attention.

Generated launchers return runtime arguments in parameter order, including inputs and scalars. Destructure deliberately or use documented tuple combinators such as `.first()`. Owned output partitions need `.unpartition()` before whole-tensor readback. Borrowed partitions modify the original tensor.

Distinguish full tensor extents from tile dimensions. Use dynamic `-1` input dimensions when varying problem sizes should reuse compiled code. Tile dimensions remain compile-time constants. Validate rank, strides/layout, broadcasting, reductions, and dimension relationships before choosing a tile. Partitioning prevents overlapping writes; it does not prove the algorithm covers every logical element correctly.

For tails, inspect the exact operation's checking, padding, and masking behavior at the target revision. Do not assume every partition operation accepts arbitrary partial tiles. A deliberate host precondition, padded allocation plus valid-region readback, or supported checked/masked access is preferable to unchecked indexing. Test just-below and just-above tile boundaries. Handle empty dimensions explicitly.

## Numerical kernels

For GEMM, tile A as `[BM, BK]`, B as `[BK, BN]`, accumulate into `[BM, BN]`, and verify the K-tail strategy. Avoid integer division that silently drops the final K tile. Verify the operation's supported operand shapes, element types, accumulation type, and architecture. A wider accumulator can be necessary even when inputs and outputs use a low-precision format.

For softmax, subtract a row maximum before exponentiation, reduce the sum on the correct axis, reshape to restore broadcasting, then divide. Apply masks with a mathematically correct neutral element and define all-masked behavior. Attention needs correct scaling, causal/padding masks, online normalization if tiled across K/V, and stable accumulation. Porting syntax from Triton is insufficient to establish these invariants.

Prefer the existing `gemm`, `softmax`, `rms_norm`, `flash_attention`, and persistent-GEMM examples as revision-matched starting points; inspect their prerequisite checks as well as the kernel.

## Checks and JIT behavior

The compiler can discharge bounds checks statically, move them to launch-time shape checks, hoist them out of loops, or leave them at accesses. Inspect placement with `CUTILE_JIT_TIMING=1` and `CUTILE_JIT_LOG=1`. `deny_in_kernel_checks = true` rejects residual device checks. Use current derived shape facts and declared `preconditions`; the older `with_bounds`/`Dim` annotation family is deprecated in this snapshot.

First try simpler loop/index structure and launch-verified preconditions. Do not make `unchecked_accesses = true` a default optimization. It requires an unsafe kernel and a complete external bounds proof, and may not improve performance. Compare actual IR and measurements before accepting it.

Specializations include kernel type/const arguments, target, and compile options. `.const_grid(...)` specializes on grid values; `.grid(...)` does not by itself. Runtime data, scalar values, and dynamic tensor extents need not produce new generic specializations, but shape/stride divisibility hints can still produce variants. Changing tile shape, element type, debug/sanitizer flags, or tuning options can trigger compilation.

Do not assume process restarts preserve the in-memory cache. Optional persistent disk caching is documented in the JIT chapter and has its own validity key. Warm the intended variants before measuring steady state; report first-use latency separately. For supported compile-only APIs, read the target version's JIT chapter rather than inventing an ahead-of-time command.

For streams, futures, graphs, and mixed Tile/SIMT pipelines, see [runtime](runtime.md). For bytecode and hardware requirements, see [setup](setup.md).
