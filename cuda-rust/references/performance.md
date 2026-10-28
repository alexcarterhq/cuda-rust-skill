# Performance engineering

## Establish comparable measurements

Define the optimization target: end-to-end request latency, throughput, transfer time, kernel time, first-use latency, memory footprint, or some combination. Record GPU/SM, driver/toolkit/assembler, source revision, build mode, shape/layout/dtype, numerical tolerance, and baseline. Do not claim a speedup from incomparable targets or debug builds.

Measure three costs separately when relevant: allocation/transfers and orchestration; device execution; compilation/module loading/JIT. Warm the same specializations and buffers used in the measured loop. Keep allocation, logging, host verification, cache population, and compilation outside a steady-state kernel interval unless those costs belong in the stated objective.

Use CUDA events on the relevant stream for device time; CPU wall time around an asynchronous launch measures submission unless completion is included. For multistream end-to-end latency, include all completion dependencies. Use repeated samples, a median and spread, representative shape classes, and randomized or paired baseline/candidate order when thermal/clock drift matters. Report units and sample scope.

Useful dimensional checks:

- Effective bandwidth = accounted bytes moved / seconds. Specify whether traffic is algorithmic or measured DRAM traffic.
- GEMM throughput = `2*M*N*K / seconds`, excluding or including epilogues as stated. Account for any reduced workload.
- Speedup = baseline time / candidate time. Check both versions against the same correctness reference.

Never invent GPU timings on a machine without CUDA execution.

## Follow the bottleneck

Use Nsight Systems for stream overlap, transfers, launch gaps, host scheduling, and synchronization. Use Nsight Compute for kernel memory behavior, instruction mix, occupancy, registers, spills, and tensor-core utilization. Build a standalone binary first so profiler output is about the application rather than Cargo and compiler subprocesses.

```bash
nsys profile -o timeline ./target/release/my_app
ncu --target-processes all -o kernel_profile ./target/release/my_app
```

Profiler replay and instrumentation alter timing. Use them to identify a cause, then measure ordinary execution separately. Optimized source-line information helps attribution; full device debug can invalidate the comparison. Hardware access to counters may be restricted; report unavailable metrics instead of estimating them.

## Useful optimization order

1. Eliminate unnecessary host/device transfers and repeated allocations; keep intermediate tensors resident.
2. Batch small work or use graphs where launch overhead dominates. Avoid adding streams when a single dependency chain has no independent work.
3. Coalesce adjacent lane accesses, use appropriate layouts, and avoid excessive data movement. Consider structure-of-arrays for irregular per-field access only when the actual access pattern benefits.
4. Reuse data through tiles/shared memory when reuse outweighs synchronization and occupancy cost. Avoid bank conflicts and local-memory spills.
5. Tune block/tile sizes, accumulation, pipeline depth, and hardware primitives against measured bottlenecks and resource limits.

Higher occupancy is not itself the objective. Increasing registers or shared memory can improve reuse while reducing residency; the tradeoff must be measured. Do not blindly apply host `target-cpu=native`, fat LTO, or panic/profile settings to device performance.

## SIMT tuning

Inspect emitted PTX and compiler reports for address spaces, integer width, bounds checks, load/store width, shared-memory layout, synchronization, and spills. Start with correct tail handling and disjoint indexing. A larger block is not automatically faster. Tensor-core paths impose operand/layout/alignment and architecture constraints that scalar CUDA cores do not.

Use CuTe/CTM, TMA, warp specialization, clusters, or matrix instructions when the workload and target justify their complexity. Tie each proposed optimization to a measurement and a maintained safety proof. Do not remove bounds/synchronization guarantees merely to make generated code look simpler.

## Tile tuning

Tune a bounded set of `[BM, BN, BK]` shapes and compile options. Large tiles increase reuse and can also increase register pressure or reduce residency. Keep dynamic problem sizes separate from static tile shapes to control JIT variant growth. Compare cold and warm cache behavior when latency matters.

Before disabling checks, inspect `CUTILE_JIT_TIMING` check-placement totals and `CUTILE_JIT_LOG` explanations. Hoisting invariant index arithmetic or expressing launch-verified shape facts may produce equally efficient safe IR. Unchecked access can be slower; measure both implementations.

`cutile::bench` provides device-event timing. `cutile::tune` is gated behind `experimental-tune` in this snapshot. Each candidate needs a correctness gate; validate shape and resource requirements before timing. Bound tuning time and persist configuration provenance (source, shape class, SM, assembler, search space). Do not reuse a winner across architectures or changed kernels without revalidation. Include the shipping configuration in comparisons so an incomplete search cannot silently regress it.

## Accept a change

Keep an optimization when it preserves the numerical/aliasing/ordering contract and improves the stated workload with comparable evidence. Report regressions in other important shape classes. Prefer a clear dispatch condition or fewer variants when gains are narrow. Deliver baseline/candidate metrics, correctness evidence, environment, and remaining uncertainty together.
