// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Copyright (c) 2026 Alex (alexcarterhq).
// SPDX-License-Identifier: Apache-2.0
// Adapted from CUDA Rust's cuTile vector-add patterns; see root NOTICE.

use cutile::prelude::*;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const B: i32>(
        output: &mut Tensor<f32, { [B] }>,
        left: &Tensor<f32, { [-1] }>,
        right: &Tensor<f32, { [-1] }>,
    ) {
        let a = left.load_like(output);
        let b = right.load_like(output);
        output.store(a + b);
    }
}

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    // These sizes deliberately satisfy the tile divisibility precondition.
    // Add an explicit tail/empty policy before accepting arbitrary user sizes.
    for n in [128usize, 256, 1024] {
        let left = api::arange::<f32>(n).sync_on(&stream)?;
        let right = api::ones::<f32>(&[n]).sync_on(&stream)?;
        let output = api::zeros::<f32>(&[n]).partition([128]);
        let (output, _left, _right) = kernels::add(output, left, right).sync_on(&stream)?;
        let result = output.unpartition().to_host_vec().sync_on(&stream)?;
        assert_eq!(result.len(), n);
        for (i, &actual) in result.iter().enumerate() {
            let expected = i as f32 + 1.0;
            assert!(actual.is_finite() && (actual - expected).abs() <= 1e-6);
        }
        println!("verified {n} elements");
    }
    Ok(())
}
