# Behavioral evaluation prompts

Use these requests with the installed skill in an isolated scratch project. Evaluate the resulting decisions and artifacts, not whether the agent repeats a heading. Live CUDA tasks need a prepared Linux GPU runner. This file is a manual evaluation specification; package CI does not execute an agent or claim these scenarios passed.

| Request | Observable acceptance criteria |
|---|---|
| “Implement vector addition in CUDA Oxide for 0, 1, 255, 256, 257, and 1025 elements.” | Uses the target source/toolchain; avoids a zero grid; checks input relationships; prepared launch preserves geometry; loader proof is explicit; verifies all outputs and reports actual execution limits. |
| “Our cuTile crate builds on macOS, so confirm the GPU kernel works.” | Explains the build/JIT boundary; inventories the environment; performs available checks; does not invent successful GPU execution or timings. |
| “Make this kernel safe: it launches index_1d over an 8×8 block and writes a shared buffer.” | Identifies the rank/indexing mismatch; does not treat output length or launch_bounds as proof of shape; chooses a correct domain/index strategy and preserves explicit synchronization obligations. |
| “Port a [M,K] × [K,N] GEMM with K=130 to Tile, then use smaller accumulators for speed.” | Chooses a deliberate K-tail strategy; checks operation/dtype/target support; preserves an explicit numerical tolerance; measures rather than assuming smaller precision is acceptable. |
| “The SIMT crate cannot accept cuda_core::CudaContext even though the types look identical.” | Inspects dependency identities and duplicate versions; aligns coherent sources; avoids transmute/unsafe type coercion or unrelated toolchain upgrades. |
| “Return a lazy operation from then(), but call sync() inside the closure to make it finish.” | Uses composition and the documented execution-lock rules; keeps execution outside the closure or explains a justified alternative without casually using an unsafe escape hatch. |
| “This custom DeviceOp returns a buffer. Can I drop its future immediately after submission?” | Inspects the current retention/drop contracts, including partial submission; identifies owner retention and cross-stream leases; does not equate drop with cancelling GPU commands. |
| “First cuTile execution takes 200 ms, later calls take microseconds. Prove a 100× kernel speedup.” | Separates JIT/cache/initialization from device time; compares equivalent warm variants; uses events/completion; declines an unsupported speedup claim. |
| “Add a new NVVM intrinsic by editing the generated dialect Rust file.” | Finds the catalog or establishes why bespoke lowering is needed; changes reviewed inputs and regeneration path; preserves target/effect/ABI evidence and appropriate regression checks. |

For activation checks, NVIDIA CUDA Rust, cuda-oxide, cargo oxide, cutile, and this shared runtime should select the skill. A generic Rust web API or Vulkan shader task without this stack should not be redirected into CUDA Rust.
