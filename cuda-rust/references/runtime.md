# Runtime, ownership, and interoperability

## Choose the API surface deliberately

The shared crates live at the repository root. `cuda-bindings` supplies driver FFI; `cuda-core` supplies contexts, buffers, streams, events, modules, and launch types; `cuda-async` composes device operations. SIMT-specific surfaces are under `cuda_core::simt` and `cuda_async::simt`, with some compatibility re-exports. Do not assume similarly named root and SIMT types/methods are interchangeable.

| Model | Typical host surface |
|---|---|
| SIMT explicit stream | `CudaContext`, `DeviceBuffer`, generated module methods |
| SIMT lazy operation | `cuda_async::simt::device_operation::DeviceOperation`, generated `<kernel>_async` |
| Tile/shared lazy operation | `cuda_async::device_operation::DeviceOp`, `Device`, tensors, generated entry launchers |

Inspect the matching example for initialization, trait imports, launch return types, and error types. `DeviceOp` and SIMT `DeviceOperation` are distinct interfaces. Do not rename one into the other while porting code.

## Allocation and transfers

Prefer typed RAII device allocations with an explicit stream/context. Allocate persistent buffers outside repeated hot-path work. Check shape products and element-to-byte multiplication for overflow before allocating or passing an FFI size. Confirm source and destination contexts and device ordinals match the intended operation.

Use zeroed storage when the algorithm consumes prior values. Uninitialized output is valid only when every subsequently observed element is written before readback; tails, error paths, and graph replay also count. Device-transfer traits must exclude invalid bit patterns and host-owned resources as specified by the actual trait contract.

Pinned host buffers support predictable asynchronous DMA. Keep them allocated and untouched for the required transfer lifetime. Pageable host storage may introduce staging or synchronization; measure rather than claiming overlap from an asynchronous API name. Overlap requires independent buffers, capable hardware, appropriate streams, and explicit dependency management.

## Submission and completion

Kernel construction and enqueueing are different from completion. A SIMT typed synchronous launcher enqueues immediately on the supplied stream; the host can continue while the GPU runs. Tile constructors and `DeviceOp`s are lazy. `.sync()`/`.sync_on()` submit and block to completion; `.await` submits when polled and resolves through the runtime's completion mechanism. Merely creating an operation or future is not an executed kernel.

Within Tile/shared composition, `.then()` chains dependent work on one stream; a combinator combining independent operations can allow overlap through the scheduler. Different streams have no automatic global ordering. Use explicit events or documented operation dependencies before consuming another stream's writes. Do not add device-wide synchronization to every kernel as a substitute for a correct dependency graph.

The shared runtime's `.then()` closure runs under an execution lock. Return the next lazy operation from the closure; calling nested `.sync()`, `.sync_on()`, or executing a nested await there is an error. `then_unchecked` is an unsafe escape hatch with additional ordering obligations, not the routine fix.

Borrowed tensors are suitable for scoped operations; moving them into a detached task does not make borrowed data `'static`. For independently spawned work, use owned or supported shared inputs and exclusive output ownership. Inspect `Send` requirements of the actual future and executor; do not assume all CUDA handles are transferable between host threads.

## Drop, cancellation, and custom operations

At this snapshot, shared `cuda-async` execution retains storage owners and access leases while submitted work is in flight, including relevant error/panic paths. Dropping an operation before submission and dropping a submitted future have different effects. Read the current implementation before relying on exact drop timing or cancellation semantics. Do not present dropping a future as cancelling already-enqueued GPU commands.

`mem::forget(future)` deliberately leaks retained owners/leases; it is not a cancellation technique. Same-stream ordering does not authorize conflicting access from another stream. On failure, determine whether work was partially submitted before reclaiming or reusing buffers.

Custom shared `KernelInputStored`/`KernelOutputStored` implementations must satisfy `retain`; custom `DeviceOp`s must register owned resources through `ExecutionContext::retain` before enqueueing. A wrapper holding fields or returning buffers is not, by itself, the complete in-flight retention proof. Follow current implementation/tests, especially `drop_in_flight`, instead of copying an older interop snippet. Raw async escape hatches leave lifetime and ordering obligations to their caller.

SIMT has a separate operation/reclamation implementation. Inspect its contracts rather than assuming every shared-runtime drop detail transfers unchanged.

## CUDA graphs

Capture a supported device-operation graph only after understanding required allocation/JIT behavior. Recording is not evidence that recorded kernels executed. Keep graph-referenced storage and modules alive. Pointers are baked into the captured graph; vary contents through supported updates into stable allocations or rebuild the graph when topology/resources change.

For shared `CudaGraph`, `.graph_on(stream)` records; `.launch()` returns a lazy operation needing execution. `graph.update(op)` accepts supported unit-output graph nodes and orders them on the graph's stream. A replay scheduled elsewhere still needs dependencies. No nested execution inside capture. Validate the first replay before timing many replays, and include capture/instantiation costs separately when they matter to the workload.

## External kernels and borrowed handles

Tile and SIMT kernels can share streams and buffers. Use the upstream inter-kernel example or an external PTX/CUBIN wrapper to avoid unnecessary host transfers. PTX compatibility depends on PTX version, target, and driver support; it is not universally portable. CUBIN compatibility follows NVIDIA's architecture rules; do not assume an arbitrary architecture's binary will load.

For raw launches, prove ABI argument order/width, pointer address spaces, context, memory extent, alignment, aliasing, lifetime, launch configuration, and completion. `AsyncKernelLaunch::push_arg` handles supported scalar values; device pointer insertion is unsafe. A safe wrapper should express exclusive/read-only ownership and retain owners, then confine raw FFI to a small documented boundary.

`Device::borrow_raw`, `Stream::borrow_raw`, and corresponding module/function constructors wrap foreign handles without transferring destruction ownership. The original owner must outlive every wrapper and in-flight operation, and must not destroy them concurrently. Matching raw C ABI representations does not prove matching device/context ownership. Never transmute framework buffer types to silence a dependency identity mismatch.

See pinned `cuda-async/README.md`, `cuda-async/src/simt/`, Tile interoperability, `cudarc_interop`, and the SIMT `cutile_inter_kernel` example in [sources](sources.md).
