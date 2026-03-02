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
fn paged_attention() -> Result<()> {
    let device = Device::new_cuda(0)?;

    let block_size = 16;
    let element_size = std::mem::size_of::<f32>();
    let x = block_size / element_size;

    let num_blocks = 1;
    let num_heads = 3;
    let head_size = 64;

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

    let q = Tensor::arange(0u32, 384, &device)?
        .to_dtype(DType::F32)?
        .reshape((num_heads, 2, head_size))?;
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;

    let q = q.transpose(0, 1)?;
    let k = k.transpose(0, 1)?;
    let v = v.transpose(0, 1)?;

    let slot_mapping = Tensor::arange(0u32, 2, &device)?.to_dtype(DType::I64)?;

    candle_paged_attention::reshape_and_cache(&k, &v, &key_cache, &value_cache, &slot_mapping)?;

    let block_tables = Tensor::new(&[0u32, 0u32], &device)?.reshape((2, 1))?;
    let context_lens = Tensor::new(&[1u32, 2u32], &device)?;

    let ys = candle_paged_attention::paged_attention(
        &q,
        &key_cache,
        &value_cache,
        &block_tables,
        &context_lens,
        2,
        0.5,
    )?
    .transpose(0, 1)?;

    assert_eq!(ys.dims(), &[num_heads, 2, head_size]);
    Ok(())
}
