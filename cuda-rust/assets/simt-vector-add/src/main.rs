// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Copyright (c) 2026 Alex (alexcarterhq).
// SPDX-License-Identifier: Apache-2.0
// Adapted from CUDA Rust's hello-gpu launch-contract example; see root NOTICE.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn add(left: &[f32], right: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(dst) = output.get_mut(index) {
            *dst = left[i] + right[i];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: this binary embeds the artifact built from this exact kernels
    // module by cargo oxide. No foreign or manually substituted bundle is used.
    let module = unsafe { kernels::load(&context)? };

    for n in [0usize, 1, 255, 256, 257, 1025] {
        if n == 0 {
            // An empty workload has no device launch or allocation to perform.
            continue;
        }
        let count = u32::try_from(n)?;
        let left: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
        let right: Vec<f32> = (0..n).map(|i| (i % 7) as f32 - 3.0).collect();
        let left_device = DeviceBuffer::from_host(&stream, &left)?;
        let right_device = DeviceBuffer::from_host(&stream, &right)?;
        let mut output = DeviceBuffer::<f32>::zeroed(&stream, n)?;
        let prepared = module.prepare_add(LaunchConfig1D::new(count.div_ceil(256), 256, 0))?;
        // All three buffers have n elements. The output bounds check therefore
        // also bounds the two input reads; excess threads do no memory work.
        module.add(&stream, &prepared, &left_device, &right_device, &mut output)?;
        let result = output.to_host_vec(&stream)?;
        assert_eq!(result.len(), n);
        for (i, &actual) in result.iter().enumerate() {
            let expected = left[i] + right[i];
            assert!(actual.is_finite() && (actual - expected).abs() <= 1e-6);
        }
        println!("verified {n} elements");
    }
    Ok(())
}
