use anyhow::Result;
use candle::{DType, Device, Tensor};

fn to_vec3_round(t: Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec3::<f32>()?;
    let t = t
        .iter()
        .map(|t| {
            t.iter()
                .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                .collect()
        })
        .collect();
    Ok(t)
}

#[test]
fn reshape_and_cache() -> Result<()> {
    let device = Device::new_cuda(0)?;

    let block_size = 4;
    let element_size = std::mem::size_of::<f32>();
    let x = block_size / element_size;

    let num_blocks = 1;
    let num_heads = 2;
    let head_size = 8;

    let key_cache = Tensor::randn(
        0.0,
        1.0,
        (num_blocks, num_heads, head_size / x, block_size, x),
        &device,
    )?
    .to_dtype(DType::F32)?;
    let value_cache = Tensor::randn(
        0.0,
        1.0,
        (num_blocks, num_heads, head_size, block_size),
        &device,
    )?
    .to_dtype(DType::F32)?;

    let key = Tensor::randn(0.0, 1.0, (4, num_heads, head_size), &device)?.to_dtype(DType::F32)?;
    let value =
        Tensor::randn(0.0, 1.0, (4, num_heads, head_size), &device)?.to_dtype(DType::F32)?;
    let slot_mapping = Tensor::arange(0u32, 4, &device)?.to_dtype(DType::I64)?;

    candle_paged_attention::reshape_and_cache(
        &key,
        &value,
        &key_cache,
        &value_cache,
        &slot_mapping,
    )?;

    let value_cache = value_cache.permute((0, 3, 1, 2))?.reshape((
        block_size * num_blocks,
        num_heads,
        head_size,
    ))?;

    assert_eq!(to_vec3_round(value, 6)?, to_vec3_round(value_cache, 6)?);

    Ok(())
}
