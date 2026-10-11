use crate::{
    backend::BackendStorage, CpuStorage, DType, Device, Result, Shape, Storage, Tensor, D,
};
use k_quants::*;
use std::borrow::Cow;

#[cfg(target_feature = "avx2")]
pub mod avx;
mod dummy_cuda;
mod dummy_metal;
pub mod ggml_file;
pub mod gguf_file;
pub mod imatrix_file;
pub mod k_quants;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(not(target_arch = "wasm32"))]
pub mod tokenizer;
#[cfg(not(feature = "metal"))]
mod metal {
    pub use super::dummy_metal::*;
}
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(not(feature = "cuda"))]
mod cuda {
    pub use super::dummy_cuda::*;
}

#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(target_feature = "simd128")]
pub mod simd128;
pub mod utils;
use half::{bf16, f16};

pub use k_quants::GgmlType;

fn as_t_slice<T>(data: Cow<'_, [u8]>) -> &[T] {
    let size = std::mem::size_of::<T>();
    assert_eq!(
        data.len() % size,
        0,
        "Data length must be a multiple of T's size"
    );
    let ptr = data.as_ptr();
    assert_eq!(
        (ptr as usize) % std::mem::align_of::<T>(),
        0,
        "Data pointer must be aligned to T's alignment"
    );
    unsafe { std::slice::from_raw_parts(ptr as *const T, data.len() / size) }
}

pub struct QTensor {
    storage: QStorage,
    shape: Shape,
}

impl Device {
    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<QStorage> {
        match self {
            Device::Cpu => {
                let storage = dtype.cpu_zeros(elem_count);
                Ok(QStorage::Cpu(storage))
            }
            Device::Metal(metal) => {
                let storage = metal::QMetalStorage::zeros(metal, elem_count, dtype)?;
                Ok(QStorage::Metal(storage))
            }
            Device::Cuda(cuda) => {
                let storage = cuda::QCudaStorage::zeros(cuda, elem_count, dtype)?;
                Ok(QStorage::Cuda(storage))
            }
        }
    }
}

pub enum QStorage {
    Cpu(Box<dyn QuantizedType>),
    Metal(metal::QMetalStorage),
    Cuda(cuda::QCudaStorage),
}

impl QStorage {
    pub fn from_data(data: Cow<'_, [u8]>, device: &Device, dtype: GgmlDType) -> Result<Self> {
        match device {
            Device::Cpu => Ok(Self::Cpu(dtype.from_data(data))),
            Device::Metal(d) => match dtype {
                GgmlDType::F32 => metal::load_quantized(d, as_t_slice::<f32>(data)),
                GgmlDType::F16 => metal::load_quantized(d, as_t_slice::<f16>(data)),
                GgmlDType::Q4_0 => metal::load_quantized(d, as_t_slice::<BlockQ4_0>(data)),
                GgmlDType::Q4_1 => metal::load_quantized(d, as_t_slice::<BlockQ4_1>(data)),
                GgmlDType::Q5_0 => metal::load_quantized(d, as_t_slice::<BlockQ5_0>(data)),
                GgmlDType::Q5_1 => metal::load_quantized(d, as_t_slice::<BlockQ5_1>(data)),
                GgmlDType::Q8_0 => metal::load_quantized(d, as_t_slice::<BlockQ8_0>(data)),
                GgmlDType::Q8_1 => metal::load_quantized(d, as_t_slice::<BlockQ8_1>(data)),
                GgmlDType::Q2K => metal::load_quantized(d, as_t_slice::<BlockQ2K>(data)),
                GgmlDType::Q3K => metal::load_quantized(d, as_t_slice::<BlockQ3K>(data)),
                GgmlDType::Q4K => metal::load_quantized(d, as_t_slice::<BlockQ4K>(data)),
                GgmlDType::Q5K => metal::load_quantized(d, as_t_slice::<BlockQ5K>(data)),
                GgmlDType::Q6K => metal::load_quantized(d, as_t_slice::<BlockQ6K>(data)),
                GgmlDType::Q8K => metal::load_quantized(d, as_t_slice::<BlockQ8K>(data)),
                GgmlDType::BF16 => metal::load_quantized(d, as_t_slice::<bf16>(data)),
            },
            Device::Cuda(d) => match dtype {
                GgmlDType::F32 => cuda::load_quantized(d, as_t_slice::<f32>(data)),
                GgmlDType::F16 => cuda::load_quantized(d, as_t_slice::<f16>(data)),
                GgmlDType::Q4_0 => cuda::load_quantized(d, as_t_slice::<BlockQ4_0>(data)),
                GgmlDType::Q4_1 => cuda::load_quantized(d, as_t_slice::<BlockQ4_1>(data)),
                GgmlDType::Q5_0 => cuda::load_quantized(d, as_t_slice::<BlockQ5_0>(data)),
                GgmlDType::Q5_1 => cuda::load_quantized(d, as_t_slice::<BlockQ5_1>(data)),
                GgmlDType::Q8_0 => cuda::load_quantized(d, as_t_slice::<BlockQ8_0>(data)),
                GgmlDType::Q8_1 => cuda::load_quantized(d, as_t_slice::<BlockQ8_1>(data)),
                GgmlDType::Q2K => cuda::load_quantized(d, as_t_slice::<BlockQ2K>(data)),
                GgmlDType::Q3K => cuda::load_quantized(d, as_t_slice::<BlockQ3K>(data)),
                GgmlDType::Q4K => cuda::load_quantized(d, as_t_slice::<BlockQ4K>(data)),
                GgmlDType::Q5K => cuda::load_quantized(d, as_t_slice::<BlockQ5K>(data)),
                GgmlDType::Q6K => cuda::load_quantized(d, as_t_slice::<BlockQ6K>(data)),
                GgmlDType::Q8K => cuda::load_quantized(d, as_t_slice::<BlockQ8K>(data)),
                GgmlDType::BF16 => cuda::load_quantized(d, as_t_slice::<bf16>(data)),
            },
        }
    }

    fn block_size(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.block_size(),
            QStorage::Metal(storage) => storage.dtype().block_size(),
            QStorage::Cuda(storage) => storage.dtype().block_size(),
        }
    }

    fn dtype(&self) -> GgmlDType {
        match self {
            QStorage::Cpu(storage) => storage.dtype(),
            QStorage::Metal(storage) => storage.dtype(),
            QStorage::Cuda(storage) => storage.dtype(),
        }
    }

    fn device(&self) -> Device {
        match self {
            QStorage::Cpu(_storage) => Device::Cpu,
            QStorage::Metal(storage) => Device::Metal(storage.device().clone()),
            QStorage::Cuda(storage) => Device::Cuda(storage.device().clone()),
        }
    }

    fn size_in_bytes(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.storage_size_in_bytes(),
            QStorage::Metal(storage) => storage.storage_size_in_bytes(),
            QStorage::Cuda(storage) => storage.storage_size_in_bytes(),
        }
    }

    fn quantize(&mut self, src: &Storage) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float(src.as_slice::<f32>()?);
            }
            (QStorage::Metal(storage), Storage::Metal(src)) => storage.quantize(src)?,
            (QStorage::Cuda(storage), Storage::Cuda(src)) => storage.quantize(src)?,
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn quantize_imatrix(
        &mut self,
        src: &Storage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
            }
            (QStorage::Metal(storage), Storage::Metal(src)) => {
                storage.quantize_imatrix(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Cuda(storage), Storage::Cuda(src)) => {
                storage.quantize_imatrix(src, imatrix_weights, n_per_row)?
            }
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn quantize_onto(&mut self, src: &Storage) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float(src.as_slice::<f32>()?);
            }
            (QStorage::Metal(storage), Storage::Cpu(src)) => storage.quantize_onto(src)?,
            (QStorage::Cuda(storage), Storage::Cpu(src)) => storage.quantize_onto(src)?,
            _ => crate::bail!("Invalid quantize source storage locations: not on cpu"),
        }
        Ok(())
    }

    fn quantize_imatrix_onto(
        &mut self,
        src: &Storage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
            }
            (QStorage::Metal(storage), Storage::Cpu(src)) => {
                storage.quantize_imatrix_onto(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Cuda(storage), Storage::Cpu(src)) => {
                storage.quantize_imatrix_onto(src, imatrix_weights, n_per_row)?
            }
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn dequantize(&self, elem_count: usize) -> Result<Storage> {
        match self {
            QStorage::Cpu(storage) => Ok(Storage::Cpu(storage.dequantize(elem_count)?)),
            QStorage::Metal(storage) => Ok(Storage::Metal(storage.dequantize(elem_count)?)),
            QStorage::Cuda(storage) => Ok(Storage::Cuda(storage.dequantize(elem_count)?)),
        }
    }

    fn data(&self) -> Result<Cow<'_, [u8]>> {
        match self {
            QStorage::Cpu(storage) => {
                let data_ptr = storage.as_ptr();
                let size_in_bytes = storage.storage_size_in_bytes();
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
                Ok(Cow::from(data))
            }
            QStorage::Cuda(storage) => Ok(Cow::from(storage.data()?)),
            QStorage::Metal(storage) => Ok(Cow::from(storage.data()?)),
        }
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        match self {
            QStorage::Cuda(storage) => storage.device_ptr(),
            QStorage::Metal(_) | QStorage::Cpu(_) => {
                crate::bail!("not implemented");
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl GgmlDType {
    pub(crate) fn from_u32(u: u32) -> Result<Self> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => Self::BF16,
            _ => crate::bail!("unknown dtype for tensor {u}"),
        };
        Ok(dtype)
    }

    pub(crate) fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            Self::BF16 => 30,
        }
    }

    /// The block dtype
    pub fn cpu_zeros(&self, elem_count: usize) -> Box<dyn QuantizedType> {
        match self {
            Self::F32 => Box::new(vec![f32::zeros(); elem_count]),
            Self::F16 => Box::new(vec![f16::zeros(); elem_count]),
            Self::Q4_0 => Box::new(vec![BlockQ4_0::zeros(); elem_count / BlockQ4_0::BLCK_SIZE]),
            Self::Q4_1 => Box::new(vec![BlockQ4_1::zeros(); elem_count / BlockQ4_1::BLCK_SIZE]),
            Self::Q5_0 => Box::new(vec![BlockQ5_0::zeros(); elem_count / BlockQ5_0::BLCK_SIZE]),
            Self::Q5_1 => Box::new(vec![BlockQ5_1::zeros(); elem_count / BlockQ5_1::BLCK_SIZE]),
            Self::Q8_0 => Box::new(vec![BlockQ8_0::zeros(); elem_count / BlockQ8_0::BLCK_SIZE]),
            Self::Q8_1 => Box::new(vec![BlockQ8_1::zeros(); elem_count / BlockQ8_1::BLCK_SIZE]),
            Self::Q2K => Box::new(vec![BlockQ2K::zeros(); elem_count / BlockQ2K::BLCK_SIZE]),
            Self::Q3K => Box::new(vec![BlockQ3K::zeros(); elem_count / BlockQ3K::BLCK_SIZE]),
            Self::Q4K => Box::new(vec![BlockQ4K::zeros(); elem_count / BlockQ4K::BLCK_SIZE]),
            Self::Q5K => Box::new(vec![BlockQ5K::zeros(); elem_count / BlockQ5K::BLCK_SIZE]),
            Self::Q6K => Box::new(vec![BlockQ6K::zeros(); elem_count / BlockQ6K::BLCK_SIZE]),
            Self::Q8K => Box::new(vec![BlockQ8K::zeros(); elem_count / BlockQ8K::BLCK_SIZE]),
            Self::BF16 => Box::new(vec![bf16::zeros(); elem_count]),
        }
    }

    pub fn from_data(&self, data: Cow<'_, [u8]>) -> Box<dyn QuantizedType> {
        match self {
            Self::F32 => Box::new(as_t_slice::<f32>(data).to_vec()),
            Self::F16 => Box::new(as_t_slice::<f16>(data).to_vec()),
            Self::Q4_0 => Box::new(as_t_slice::<BlockQ4_0>(data).to_vec()),
            Self::Q4_1 => Box::new(as_t_slice::<BlockQ4_1>(data).to_vec()),
            Self::Q5_0 => Box::new(as_t_slice::<BlockQ5_0>(data).to_vec()),
            Self::Q5_1 => Box::new(as_t_slice::<BlockQ5_1>(data).to_vec()),
            Self::Q8_0 => Box::new(as_t_slice::<BlockQ8_0>(data).to_vec()),
            Self::Q8_1 => Box::new(as_t_slice::<BlockQ8_1>(data).to_vec()),
            Self::Q2K => Box::new(as_t_slice::<BlockQ2K>(data).to_vec()),
            Self::Q3K => Box::new(as_t_slice::<BlockQ3K>(data).to_vec()),
            Self::Q4K => Box::new(as_t_slice::<BlockQ4K>(data).to_vec()),
            Self::Q5K => Box::new(as_t_slice::<BlockQ5K>(data).to_vec()),
            Self::Q6K => Box::new(as_t_slice::<BlockQ6K>(data).to_vec()),
            Self::Q8K => Box::new(as_t_slice::<BlockQ8K>(data).to_vec()),
            Self::BF16 => Box::new(as_t_slice::<bf16>(data).to_vec()),
        }
    }

    /// The type size for blocks in bytes.
    pub fn type_size(&self) -> usize {
        use k_quants::*;
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6K => std::mem::size_of::<BlockQ6K>(),
            Self::Q8K => std::mem::size_of::<BlockQ8K>(),
        }
    }

    /// The block size, i.e. the number of elements stored in each block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 | Self::BF16 => 1,
            Self::Q4_0 => k_quants::QK4_0,
            Self::Q4_1 => k_quants::QK4_1,
            Self::Q5_0 => k_quants::QK5_0,
            Self::Q5_1 => k_quants::QK5_1,
            Self::Q8_0 => k_quants::QK8_0,
            Self::Q8_1 => k_quants::QK8_1,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => k_quants::QK_K,
        }
    }
}

// A version of GgmlType without `vec_dot` so that it can be dyn boxed.
pub trait QuantizedType: Send + Sync {
    fn dtype(&self) -> GgmlDType;
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()>;
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()>;
    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage>;
    fn storage_size_in_bytes(&self) -> usize;
    fn as_ptr(&self) -> *const u8;
    fn block_size(&self) -> usize;
    #[allow(clippy::wrong_self_convention)]
    fn from_float(&mut self, xs: &[f32]);
    #[allow(clippy::wrong_self_convention)]
    fn from_float_imatrix(&mut self, xs: &[f32], imatrix_weights: &[f32], n_per_row: usize);
    fn size(&self) -> usize;
    /// SwarmLLM patch: the block type an activation is quantized to before a
    /// product with this weight (`GgmlType::VecDotType`).
    fn vec_dot_dtype(&self) -> GgmlDType;
    /// SwarmLLM patch: activation rows (`xs`, rows of `k`) quantized for this
    /// weight type, as `k_quants::matmul` quantizes its input.
    fn quantize_rows(&self, xs: &[f32], k: usize) -> Box<dyn QuantizedBlocks>;
    /// SwarmLLM patch: weight rows `first..` against activation rows
    /// `lhs_rows` of `lhs` (from `quantize_rows` of a weight with the same
    /// `vec_dot_dtype`), column-major: weight row `first + j` times activation
    /// row `lhs_rows.start + r` lands at `dst[j * lhs_rows.len() + r]`.
    fn matmul_rows(
        &self,
        k: usize,
        lhs: &(dyn std::any::Any + Send + Sync),
        lhs_rows: std::ops::Range<usize>,
        first: usize,
        dst: &mut [f32],
    ) -> Result<()>;
}

/// SwarmLLM patch: a type-erased buffer of quantized activation rows (a
/// `Vec<GgmlType::VecDotType>`), able to copy out some of its rows.
pub trait QuantizedBlocks: std::any::Any + Send + Sync {
    fn as_any(&self) -> &(dyn std::any::Any + Send + Sync);
    /// The rows `rows`, each `row_blocks` blocks long, in that order.
    fn gather(&self, row_blocks: usize, rows: &[usize]) -> Box<dyn QuantizedBlocks>;
}

impl<B: Clone + Send + Sync + 'static> QuantizedBlocks for Vec<B> {
    fn as_any(&self) -> &(dyn std::any::Any + Send + Sync) {
        self
    }

    fn gather(&self, row_blocks: usize, rows: &[usize]) -> Box<dyn QuantizedBlocks> {
        let mut out = Vec::with_capacity(rows.len() * row_blocks);
        for &r in rows {
            out.extend_from_slice(&self[r * row_blocks..(r + 1) * row_blocks]);
        }
        Box::new(out)
    }
}

/// Activation rows per `vec_dot_rows` call of [`QuantizedType::matmul_rows`]
/// — `k_quants::matmul`'s `ROW_BLOCK`, which keeps them in L2.
const MATMUL_ROW_BLOCK: usize = 128;

/// Fewest rows [`QuantizedType::quantize_rows`] quantizes in parallel: a
/// decoded token's few rows are not worth a fork. Either way each row is the
/// same `from_float`, so the bits do not depend on it.
const QUANTIZE_ROWS_IN_PARALLEL: usize = 16;

impl<T: k_quants::GgmlType + Send + Sync> QuantizedType for Vec<T>
where
    T::VecDotType: 'static,
{
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        k_quants::matmul(mkn, lhs, self.as_slice(), dst)
    }

    fn vec_dot_dtype(&self) -> GgmlDType {
        T::VecDotType::DTYPE
    }

    // The quantization of `k_quants::matmul`, kept apart so that one quantized
    // input can serve several weights: same zeroed blocks, same direct copy or
    // one `from_float` per row.
    fn quantize_rows(&self, xs: &[f32], k: usize) -> Box<dyn QuantizedBlocks> {
        let k_in_blocks = k.div_ceil(T::BLCK_SIZE);
        let rows = xs.len() / k;
        let mut blocks = vec![T::VecDotType::zeros(); rows * k_in_blocks];
        if T::DIRECT_COPY {
            T::VecDotType::direct_copy(xs, &mut blocks);
        } else if rows < QUANTIZE_ROWS_IN_PARALLEL {
            for (b, x) in blocks.chunks_mut(k_in_blocks).zip(xs.chunks(k)) {
                T::VecDotType::from_float(x, b);
            }
        } else {
            use rayon::prelude::*;
            blocks
                .par_chunks_mut(k_in_blocks)
                .zip(xs.par_chunks(k))
                .for_each(|(b, x)| T::VecDotType::from_float(x, b));
        }
        Box::new(blocks)
    }

    // The loops of `k_quants::matmul` over a range of its columns: one
    // `T::vec_dot` per output for one row, `T::vec_dot_rows` over blocks of
    // `MATMUL_ROW_BLOCK` rows for more — the same operands, so the same bits.
    fn matmul_rows(
        &self,
        k: usize,
        lhs: &(dyn std::any::Any + Send + Sync),
        lhs_rows: std::ops::Range<usize>,
        first: usize,
        dst: &mut [f32],
    ) -> Result<()> {
        let Some(lhs) = lhs.downcast_ref::<Vec<T::VecDotType>>() else {
            crate::bail!(
                "matmul_rows: the activation was not quantized for {:?}",
                T::DTYPE
            )
        };
        let k_in_blocks = k.div_ceil(T::BLCK_SIZE);
        let rows = lhs_rows.len();
        if rows == 0
            || lhs_rows.end * k_in_blocks > lhs.len()
            || !dst.len().is_multiple_of(rows)
            || (first + dst.len() / rows) * k_in_blocks > self.len()
        {
            crate::bail!(
                "matmul_rows: rows {lhs_rows:?} of {} blocks against weight rows {first}..{} \
                 of {k} wide in {} blocks",
                lhs.len(),
                first + dst.len() / rows.max(1),
                self.len()
            )
        }
        let lhs = &lhs[lhs_rows.start * k_in_blocks..lhs_rows.end * k_in_blocks];
        let col = |j: usize| &self[(first + j) * k_in_blocks..(first + j + 1) * k_in_blocks];
        if rows == 1 {
            for (j, out) in dst.iter_mut().enumerate() {
                *out = T::vec_dot(k, col(j), lhs);
            }
            return Ok(());
        }
        let mut base = 0;
        while base < rows {
            let n = MATMUL_ROW_BLOCK.min(rows - base);
            let block = &lhs[base * k_in_blocks..(base + n) * k_in_blocks];
            for (j, out) in dst.chunks_mut(rows).enumerate() {
                k_quants::vec_dot_rows_chosen::<T>(k, col(j), block, n, &mut out[base..base + n]);
            }
            base += n;
        }
        Ok(())
    }
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()> {
        k_quants::matmul_f16(mkn, lhs, self.as_slice(), dst)
    }

    fn size(&self) -> usize {
        self.len() * core::mem::size_of::<T>()
    }

    fn from_float(&mut self, xs: &[f32]) {
        T::from_float(xs, self)
    }

    fn from_float_imatrix(&mut self, xs: &[f32], imatrix_weights: &[f32], n_per_row: usize) {
        T::from_float_imatrix(xs, self, imatrix_weights, n_per_row)
    }

    fn dtype(&self) -> GgmlDType {
        T::DTYPE
    }

    fn block_size(&self) -> usize {
        T::BLCK_SIZE
    }

    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage> {
        let mut ys = vec![0.0f32; elem_count];
        T::to_float(self.as_slice(), &mut ys);
        Ok(CpuStorage::F32(ys))
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    fn as_ptr(&self) -> *const u8 {
        self.as_ptr() as *const u8
    }
}

/// SwarmLLM patch: activation rows, quantized ONCE for a weight type
/// (`QTensor::quantize_rows`) and usable against any processor weight with
/// the same `vec_dot_dtype` — gate and up of every expert, usually.
pub struct QuantizedRows {
    blocks: Box<dyn QuantizedBlocks>,
    vec_dot_dtype: GgmlDType,
    k: usize,
    rows: usize,
}

impl QuantizedRows {
    /// Can these rows be multiplied by `weight` as they are?
    pub fn fits(&self, weight: &QTensor) -> bool {
        match (&weight.storage, weight.shape.dims2()) {
            (QStorage::Cpu(w), Ok((_, k))) => w.vec_dot_dtype() == self.vec_dot_dtype && k == self.k,
            _ => false,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The rows `rows`, in that order, as rows of their own — an expert's
    /// tokens out of a prompt quantized once. A copy of blocks, so each row
    /// keeps its exact bits.
    pub fn gather(&self, rows: &[usize]) -> Result<Self> {
        if rows.is_empty() || rows.iter().any(|&r| r >= self.rows) {
            crate::bail!("gather: rows {rows:?} of {}", self.rows)
        }
        let row_blocks = self.k.div_ceil(self.vec_dot_dtype.block_size());
        Ok(Self {
            blocks: self.blocks.gather(row_blocks, rows),
            vec_dot_dtype: self.vec_dot_dtype,
            k: self.k,
            rows: rows.len(),
        })
    }
}

/// SwarmLLM patch: one job of [`matmul_many`] — `weight` times the rows
/// `lhs_rows` of `lhs`, COLUMN-major: output `j` of row `r` at
/// `dst[j * lhs_rows.len() + r]` (one row: the plain output vector).
pub struct MatMulJob<'a> {
    pub weight: &'a QTensor,
    pub lhs: &'a QuantizedRows,
    pub lhs_rows: std::ops::Range<usize>,
    pub dst: &'a mut [f32],
}

impl QTensor {
    /// SwarmLLM patch: `xs` (rows as wide as this weight) quantized as
    /// `forward` would quantize them. Processor weights only.
    pub fn quantize_rows(&self, xs: &[f32]) -> Result<QuantizedRows> {
        let QStorage::Cpu(w) = &self.storage else {
            crate::bail!("quantize_rows: processor weights only")
        };
        let (_, k) = self.shape.dims2()?;
        if k == 0 || xs.is_empty() || !xs.len().is_multiple_of(k) {
            crate::bail!("quantize_rows: {} values for a weight {k} wide", xs.len())
        }
        Ok(QuantizedRows {
            blocks: w.quantize_rows(xs, k),
            vec_dot_dtype: w.vec_dot_dtype(),
            k,
            rows: xs.len() / k,
        })
    }
}

/// Weight rows per task of [`matmul_many`] — llama.cpp's chunk for a product
/// with one activation row (`ggml_compute_forward_mul_mat_id`, `chunk_size =
/// 64` when `nr1 == 1`) and `k_quants::matmul`'s `COL_BLOCK`. Fixed, so no
/// result depends on the thread count.
const MATMUL_COLS_PER_TASK: usize = 64;

/// SwarmLLM patch: many products against processor weights in ONE parallel
/// region over 64-row chunks of every job's weight.
///
/// A mixture-of-experts layer multiplies each token by its chosen experts'
/// small matrices — 8 of 128 on Qwen3-30B-A3B, three each. Each expert as its
/// own `QMatMul::forward` is a parallel region of a few hundred rows with
/// tensor plumbing around it — for a 681-token prompt, 384 regions a layer of
/// ~12 tasks each; llama.cpp's `mul_mat_id` instead runs every chosen expert
/// of one projection in ONE thread region, its threads walking the experts'
/// row chunks with no barrier between experts. This is that shape (FUTURE_WORK
/// #138), each task applying its 64 weight rows to all of its job's
/// activation rows, as `k_quants::matmul` does.
///
/// Each output is the `T::vec_dot` (one row) or `T::vec_dot_rows` (several)
/// over the operands `forward` uses, so the result is bit-identical to
/// calling it per job.
pub fn matmul_many(jobs: Vec<MatMulJob<'_>>) -> Result<()> {
    use rayon::prelude::*;
    let mut tasks = Vec::new();
    for MatMulJob {
        weight,
        lhs,
        lhs_rows,
        dst,
    } in jobs
    {
        let QStorage::Cpu(w) = &weight.storage else {
            crate::bail!("matmul_many: processor weights only")
        };
        let (n, k) = weight.shape.dims2()?;
        let rows = lhs_rows.len();
        if rows == 0 || lhs_rows.end > lhs.rows || dst.len() != n * rows || !lhs.fits(weight) {
            crate::bail!(
                "matmul_many: {n} x {k} weight ({:?}), {} outputs, rows {lhs_rows:?} of {} \
                 {} wide for {:?}",
                weight.dtype(),
                dst.len(),
                lhs.rows,
                lhs.k,
                lhs.vec_dot_dtype
            )
        }
        for (chunk, out) in dst.chunks_mut(MATMUL_COLS_PER_TASK * rows).enumerate() {
            let first = chunk * MATMUL_COLS_PER_TASK;
            tasks.push((w.as_ref(), lhs.blocks.as_any(), k, lhs_rows.clone(), first, out));
        }
    }
    tasks
        .into_par_iter()
        .try_for_each(|(w, lhs, k, rows, first, out)| w.matmul_rows(k, lhs, rows, first, out))
}

impl std::fmt::Debug for QTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "QTensor[{:?}; {:?}]", self.shape, self.dtype())
    }
}

fn check_shape(shape: &Shape, block_size: usize) -> Result<()> {
    let dims = shape.dims();
    if dims.is_empty() {
        crate::bail!("scalar tensor cannot be quantized {shape:?}")
    }
    if !dims[dims.len() - 1].is_multiple_of(block_size) {
        crate::bail!(
            "quantized tensor must have their last dim divisible by block size {shape:?} {}",
            block_size
        )
    }
    Ok(())
}

impl QTensor {
    pub fn new<S: Into<Shape>>(storage: QStorage, shape: S) -> Result<Self> {
        let shape = shape.into();
        check_shape(&shape, storage.block_size())?;
        Ok(Self { storage, shape })
    }

    pub fn quantize(src: &Tensor, dtype: GgmlDType) -> Result<Self> {
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize(&src.storage())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    pub fn quantize_imatrix(
        src: &Tensor,
        imatrix_weights: &[f32],
        dtype: GgmlDType,
    ) -> Result<Self> {
        // (n_per_row/QK_K-1)*QK_K+(QK_K/32-1)*32+32=n_per_row
        // Size of imatrix == last dim of tensor
        let n_per_row = src.dim(D::Minus1)?;
        if imatrix_weights.len() != n_per_row {
            crate::bail!(
                "imatrix weights must have the same length {} as the last dim of src {}",
                imatrix_weights.len(),
                src.dim(D::Minus1)?
            );
        }

        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            );
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize_imatrix(&src.storage(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`
    pub fn quantize_imatrix_onto(
        src: &Tensor,
        imatrix_weights: &[f32],
        dtype: GgmlDType,
        dev: &Device,
    ) -> Result<Self> {
        if !src.device().is_cpu() {
            crate::bail!(
                "`quantize_onto` expects a `src` to be on the cpu, got {:?}.",
                src.device()
            )
        }
        // (n_per_row/QK_K-1)*QK_K+(QK_K/32-1)*32+32=n_per_row
        // Size of imatrix == last dim of tensor
        let n_per_row = src.dim(D::Minus1)?;
        if imatrix_weights.len() != n_per_row {
            crate::bail!(
                "imatrix weights must have the same length {} as the last dim of src {}",
                imatrix_weights.len(),
                src.dim(D::Minus1)?
            );
        }
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        // storage is on the `dev`, src is on `cpu`
        let mut storage = dev.qzeros(elem_count, dtype)?;
        storage.quantize_imatrix_onto(&src.storage(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`
    pub fn quantize_onto(src: &Tensor, dtype: GgmlDType, dev: &Device) -> Result<Self> {
        if !src.device().is_cpu() {
            crate::bail!(
                "`quantize_onto` expects a `src` to be on the cpu, got {:?}.",
                src.device()
            )
        }
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        // storage is on the `dev`, src is on `cpu`
        let mut storage = dev.qzeros(elem_count, dtype)?;
        storage.quantize_onto(&src.storage())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let storage = self.storage.dequantize(self.shape.elem_count())?;
        let none = crate::op::BackpropOp::none();
        crate::tensor::from_storage(storage, self.shape.clone(), none, false).to_device(device)
    }

    pub fn dequantize_f16(&self, device: &Device) -> Result<Tensor> {
        // In the CUDA case, we have a specialized kernel as this can be useful for volta
        // architectures. https://github.com/huggingface/candle/issues/2136
        match &self.storage {
            QStorage::Cuda(s) => {
                let s = s.dequantize_f16(self.shape.elem_count())?;
                let none = crate::op::BackpropOp::none();
                crate::tensor::from_storage(Storage::Cuda(s), self.shape.clone(), none, false)
                    .to_device(device)
            }
            _ => {
                let s = self.dequantize(device)?.to_dtype(crate::DType::F16)?;
                Ok(s)
            }
        }
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.storage.size_in_bytes()
    }

    pub fn data(&self) -> Result<Cow<'_, [u8]>> {
        self.storage.data()
    }

    /// SwarmLLM patch: rows of a `[vocab, hidden]` quantized tensor, gathered
    /// on whatever device the tensor lives on, returned still quantized.
    ///
    /// The caller dequantizes the (small) result, so an embedding lookup costs
    /// `n_ids` rows instead of a resident copy of the whole table. See
    /// `cuda::QCudaStorage::gather_rows` for why the CUDA half cannot simply
    /// reuse the CPU one.
    ///
    /// `ids` must be U32 and on the same device. Rows must be a whole number of
    /// quantization blocks — checked here, because a row that straddles a block
    /// cannot be addressed and reading across the boundary would silently
    /// return the wrong embedding.
    pub fn gather_rows(&self, ids: &Tensor) -> Result<Self> {
        let (vocab, hidden) = match *self.shape.dims() {
            [v, h] => (v, h),
            _ => crate::bail!("gather_rows expects a 2-D [vocab, hidden] tensor"),
        };
        let block = self.dtype().block_size();
        if block == 0 || !hidden.is_multiple_of(block) {
            crate::bail!(
                "gather_rows: hidden {hidden} is not a whole number of {block}-element blocks"
            )
        }
        let row_bytes = hidden / block * self.dtype().type_size();
        let n_ids = ids.elem_count();
        if ids.dtype() != crate::DType::U32 {
            crate::bail!("gather_rows: ids must be u32, got {:?}", ids.dtype())
        }
        match &self.storage {
            #[cfg(feature = "cuda")]
            QStorage::Cuda(s) => {
                let ids_storage = ids.storage();
                let gathered = match &*ids_storage {
                    Storage::Cuda(c) => match &c.slice {
                        crate::cuda_backend::CudaStorageSlice::U32(sl) => {
                            s.gather_rows(sl, n_ids, vocab, row_bytes)?
                        }
                        _ => crate::bail!("gather_rows: ids must be u32"),
                    },
                    _ => crate::bail!("gather_rows: ids must be on the same CUDA device"),
                };
                QTensor::new(QStorage::Cuda(gathered), (n_ids, hidden))
            }
            #[cfg(not(feature = "cuda"))]
            QStorage::Cuda(_) => crate::bail!("gather_rows: not compiled with CUDA support"),
            QStorage::Cpu(_) => {
                // Zero-copy on CPU, so the rows can simply be sliced out.
                let bytes = self.data()?;
                let ids: Vec<u32> = ids.flatten_all()?.to_vec1()?;
                let mut gathered = Vec::with_capacity(n_ids * row_bytes);
                for &id in &ids {
                    let id = id as usize;
                    if id >= vocab {
                        crate::bail!("gather_rows: id {id} out of range for vocabulary {vocab}")
                    }
                    let start = id * row_bytes;
                    gathered.extend_from_slice(&bytes[start..start + row_bytes]);
                }
                // NOTE: `Cow::Borrowed`. `as_t_slice` takes the `Cow` by value
                // and returns a slice borrowed from it, so an owned buffer is
                // dropped before the read behind it happens.
                let storage = QStorage::from_data(
                    Cow::Borrowed(gathered.as_slice()),
                    &Device::Cpu,
                    self.dtype(),
                )?;
                QTensor::new(storage, (n_ids, hidden))
            }
            QStorage::Metal(_) => crate::bail!("gather_rows is not implemented for Metal"),
        }
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match &self.storage {
            QStorage::Cuda(s) => match (&*x.storage(), &*ids.storage()) {
                (Storage::Cuda(x_storage), Storage::Cuda(ids_storage)) => {
                    let (storage, out_shape) = s.indexed_moe_forward(
                        self.shape(),
                        x_storage,
                        x.layout(),
                        ids_storage,
                        ids.layout(),
                    )?;
                    Ok(crate::tensor::from_storage(
                        Storage::Cuda(storage),
                        out_shape,
                        crate::op::BackpropOp::none(),
                        false,
                    ))
                }
                _ => {
                    panic!("Non-cuda indexed_moe_forward is not implemented!");
                }
            },
            _ => {
                panic!("indexed_moe_forward is not implemented in this platform!");
            }
        }
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        match &self.storage {
            QStorage::Cuda(storage) => storage.device_ptr(),
            QStorage::Metal(_) | QStorage::Cpu(_) => {
                crate::bail!("not implemented");
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum QMatMul {
    QTensor(std::sync::Arc<QTensor>),
    Tensor(Tensor),
    TensorF16(Tensor),
}

thread_local! {
    static DEQUANTIZE_ALL: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

thread_local! {
    static DEQUANTIZE_ALL_F16: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL_F16") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

impl QMatMul {
    pub fn from_arc(qtensor: std::sync::Arc<QTensor>) -> Result<Self> {
        let dequantize = match qtensor.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => true,
            _ => DEQUANTIZE_ALL.with(|b| *b),
        };
        let t = if dequantize {
            let tensor = qtensor.dequantize(&qtensor.device())?;
            Self::Tensor(tensor)
        } else if DEQUANTIZE_ALL_F16.with(|b| *b) {
            let tensor = qtensor.dequantize_f16(&qtensor.device())?;
            Self::TensorF16(tensor)
        } else {
            Self::QTensor(qtensor)
        };
        Ok(t)
    }

    pub fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        Self::from_arc(std::sync::Arc::new(qtensor))
    }

    /// Keep this weight's products in f32 where a card's prompt pass would
    /// accumulate them in f16 (llama.cpp's `GGML_PREC_F32`; SwarmLLM
    /// FUTURE_WORK #147). Only a quantized weight on a card takes that path, so
    /// any other is answered `true` unchanged; `false` only for a quantized
    /// weight someone else also holds, which cannot be changed from here.
    pub fn accumulate_in_f32(&mut self) -> bool {
        match self {
            Self::QTensor(t) => match std::sync::Arc::get_mut(t) {
                Some(t) => {
                    if let QStorage::Cuda(s) = &mut t.storage {
                        s.accumulate_in_f32();
                    }
                    true
                }
                None => false,
            },
            Self::Tensor(_) | Self::TensorF16(_) => true,
        }
    }

    pub fn dequantize_f16(&self) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.dequantize_f16(&t.device()),
            Self::Tensor(t) => t.to_dtype(DType::F16),
            Self::TensorF16(t) => Ok(t.clone()),
        }
    }

    pub fn forward_via_f16(&self, xs: &Tensor) -> Result<Tensor> {
        let w = self.dequantize_f16()?;
        let in_dtype = xs.dtype();
        let w = match *xs.dims() {
            [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
            _ => w.t()?,
        };
        xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
    }

    /// SwarmLLM patch: run several weight matrices against ONE activation,
    /// quantizing that activation once instead of once per matmul.
    ///
    /// `q`/`k`/`v` share the post-attention-norm hidden state and the FFN's
    /// `gate`/`up` share the post-FFN-norm one, but each matmul quantized its
    /// input independently — 7 `quantize_q8_1` launches per layer for 7 matmuls
    /// when only 4 inputs were distinct. On a machine where a decoded token is
    /// bound by how many times the CPU talks to the driver, 3 needless launches
    /// per layer is ~11% of them.
    ///
    /// **This owns the loop on purpose.** The sharing window must be closed to
    /// anything but these matmuls: a cache keyed on the input's device pointer
    /// would be unsound, because CUDA recycles freed addresses and a stale hit
    /// is a silently wrong reply; and a guard the caller opens would leave a
    /// window an unrelated matmul (LoRA, for one) could hit. Here `xs` is
    /// borrowed for the whole call, so its storage cannot be freed, and nothing
    /// else runs inside.
    ///
    /// Falls back to a plain loop off CUDA, for dequantized weights, and for a
    /// single weight — so it is always correct to call and never *required*.
    ///
    /// → `docs/invariants/inference.md` § "A decode token is bound by GPU
    ///   submission COUNT, not bandwidth"
    pub fn forward_shared(xs: &Tensor, ws: &[&Self]) -> Result<Vec<Tensor>> {
        #[cfg(feature = "cuda")]
        {
            // Only the quantized CUDA path shares anything: a dequantized
            // weight goes through cuBLAS and never calls `quantize_q8_1`.
            let shareable = xs.device().is_cuda()
                && ws.len() > 1
                && ws.iter().all(|w| matches!(w, Self::QTensor(_)));
            if shareable {
                return cuda::with_shared_activation(|| {
                    ws.iter()
                        .map(|w| crate::Module::forward(*w, xs))
                        .collect()
                });
            }
        }
        ws.iter()
            .map(|w| crate::Module::forward(*w, xs))
            .collect()
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.indexed_moe_forward(x, ids),
            _ => {
                panic!("Not implemented!")
            }
        }
    }
}

impl crate::CustomOp1 for QTensor {
    fn name(&self) -> &'static str {
        "qmatmul"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        let (n, k) = self.shape.dims2()?;
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self.shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        #[allow(clippy::infallible_destructuring_match)]
        let self_storage = match &self.storage {
            QStorage::Cpu(storage) => storage,
            QStorage::Metal(_) | QStorage::Cuda(_) => crate::bail!("Invalid storage"),
        };
        match storage.dtype() {
            DType::F32 => {
                let slice = storage.as_slice::<f32>()?;
                let slice =
                    &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                let mut dst_storage = vec![0f32; dst_shape.elem_count()];
                self_storage.matmul_t(
                    (dst_shape.elem_count() / n, k, n),
                    slice,
                    &mut dst_storage,
                )?;
                Ok((crate::CpuStorage::F32(dst_storage), dst_shape))
            }
            DType::F16 => {
                let slice = storage.as_slice::<f16>()?;
                let slice =
                    &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                let mut dst_storage = vec![f16::ZERO; dst_shape.elem_count()];
                self_storage.matmul_t_f16(
                    (dst_shape.elem_count() / n, k, n),
                    slice,
                    &mut dst_storage,
                )?;
                Ok((crate::CpuStorage::F16(dst_storage), dst_shape))
            }
            _ => crate::bail!("Expected f32/f16"),
        }
    }

    fn metal_fwd(
        &self,
        storage: &crate::MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Metal(metal) => metal,
            _ => unreachable!("Cannot call metal matmul on non metal QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }

    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Cuda(cuda) => cuda,
            _ => unreachable!("Cannot call cuda matmul on non cuda QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }
}

impl crate::Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => xs.apply_op1_no_bwd(t.as_ref()),
            Self::Tensor(w) => {
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.matmul(&w)
            }
            Self::TensorF16(w) => {
                let in_dtype = xs.dtype();
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
            }
        }
    }
}
