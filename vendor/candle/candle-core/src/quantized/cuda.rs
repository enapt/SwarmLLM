use super::{GgmlDType, QStorage};
use crate::quantized::k_quants::GgmlType;
use crate::{
    backend::{BackendDevice, BackendStorage},
    cuda_backend::WrapErr,
};
use crate::{builder_arg as barg, CudaDevice, CudaStorage, Result};
use half::f16;

use cudarc::driver::{CudaSlice, CudaView, PushKernelArg};

#[derive(Clone, Debug)]
struct PaddedCudaSlice {
    inner: CudaSlice<u8>,
    len: usize,
}

#[derive(Clone, Debug)]
pub struct QCudaStorage {
    data: PaddedCudaSlice,
    dtype: GgmlDType,
    device: CudaDevice,
    /// This weight's products accumulate in f32 on the f16 cuBLAS path, where
    /// every other weight's accumulate in f16 — llama.cpp's `GGML_PREC_F32`
    /// mark, set by [`Self::accumulate_in_f32`] (SwarmLLM FUTURE_WORK #147).
    accumulate_f32: bool,
}

static FORCE_DMMV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_force_dmmv(f: bool) {
    FORCE_DMMV.store(f, std::sync::atomic::Ordering::Relaxed)
}

pub const WARP_SIZE: usize = 32;
pub const MMQ_X_Q4_0_AMPERE: usize = 4;
pub const MMQ_Y_Q4_0_AMPERE: usize = 32;
pub const NWARPS_Q4_0_AMPERE: usize = 4;
pub const GGML_CUDA_MMV_X: usize = 32;
pub const GGML_CUDA_MMV_Y: usize = 1;
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const CUDA_DEQUANTIZE_BLOCK_SIZE: usize = 256;
pub const MATRIX_ROW_PADDING: usize = 512;

fn ceil_div(p: usize, q: usize) -> usize {
    p.div_ceil(q)
}

fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

fn quantize_q8_1(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);

    let total_rows = ky;
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = kx_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;

    const CHUNK_SIZE: usize = 65535; // gridDim.y limit
    let func = dev.get_or_load_func("quantize_q8_1", &candle_kernels::QUANTIZED)?;

    let mut rows_processed = 0;
    while rows_processed < total_rows {
        // --- calculate the number of rows for this chunk ---
        let remaining_rows = total_rows - rows_processed;
        // This is our gridDim.y, now <= 65535
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        // --- slice the source (f32) tensor by elements ---
        let src_start_elem = rows_processed * k;
        let src_num_elems = rows_in_chunk * k;
        let src_chunk = src.slice(src_start_elem..(src_start_elem + src_num_elems));

        // --- slice the destination (u8) tensor by bytes ---
        let dst_start_byte = rows_processed * dst_row_size_bytes;
        let dst_num_bytes = rows_in_chunk * dst_row_size_bytes;
        let dst_chunk = dst.slice(dst_start_byte..(dst_start_byte + dst_num_bytes));

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, rows_in_chunk as u32, 1),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&src_chunk);
        builder.arg(&dst_chunk);
        barg!(builder, k as i32, kx_padded as i32);
        unsafe { builder.launch(cfg) }.w()?;

        rows_processed += rows_in_chunk;
    }

    Ok(())
}

fn dequantize_f32(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f32", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f32", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f32", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f32", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f32", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f32", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f32", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f32", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f32", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// SwarmLLM patch: the dtypes that have a CUDA dequantize kernel.
///
/// `dequantize_f32` and `dequantize_f16` below are generated from the same
/// list of block types, so a dtype outside it — an *unquantized* GGUF, i.e.
/// `F16` / `BF16` / `F32` — has no kernel on either side and must be handled
/// on the host. Both `QCudaStorage::dequantize` and
/// `QCudaStorage::dequantize_f16` gate on this ONE predicate so they cannot
/// disagree about which dtypes they accept: they used to, and the f16 side
/// was the one missing the fallback.
fn has_dequantize_kernel(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
            | GgmlDType::Q8K
    )
}

fn dequantize_f16(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let dst = dequantize_f16_slice(data, dtype, elem_count, dev)?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// SwarmLLM patch: [`dequantize_f16`]'s kernel launch, handing back the raw
/// slice so [`mul_mat_via_f16_cublas`] can pass it straight to cuBLAS.
fn dequantize_f16_slice(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaSlice<f16>> {
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f16", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f16", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f16", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f16", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f16", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f16", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f16", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f16", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f16", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f16>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(dst)
}

/// SwarmLLM patch: the fewest activation rows for which a quantized matmul
/// dequantizes the weight to f16 and hands it to cuBLAS instead of running
/// the MMQ kernel.
///
/// **llama.cpp's own rule, for OUR kind of MMQ.** The MMQ kernels vendored here
/// are llama.cpp's older dp4a ones (`mul_mat_q4_K` and friends: `__dp4a`, no
/// tensor cores). llama.cpp keeps a dp4a MMQ only below
/// `MMQ_DP4A_MAX_BATCH_SIZE` = 64 rows when the card has fp16 tensor cores —
/// "Max. batch size to use for dp4a MMQ kernels when FP16 tensor cores are
/// available" (`ggml-cuda/mmq.cuh`); above it `ggml_cuda_should_use_mmq`
/// returns false and the weight is dequantized to f16 for a cuBLAS GEMM
/// (`ggml_cuda_op_mul_mat_cublas`). Its NEWER MMQ uses int8 tensor-core `mma`
/// instructions, which is why current llama.cpp keeps MMQ on Ampere; ours has
/// no such kernel, so the dp4a rule is the one that applies to us.
///
/// Decode (one row per slot, eight at most) never reaches it; a prompt pass
/// does, which is where time-to-first-token goes. `SWARMLLM_QMATMUL_CUBLAS=0`
/// keeps MMQ for every batch (the A/B in one binary);
/// `SWARMLLM_QMATMUL_CUBLAS_MIN_ROWS` moves the threshold for a measurement.
const CUBLAS_MIN_ROWS: usize = 64;

/// Rows at which [`mul_mat_via_f16_cublas`] takes over, or `None` when it
/// never does (switched off, or a card without fp16 tensor cores). Read once.
fn cublas_min_rows(dev: &CudaDevice) -> Option<usize> {
    static MIN_ROWS: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *MIN_ROWS.get_or_init(|| {
        if matches!(
            std::env::var("SWARMLLM_QMATMUL_CUBLAS").ok().as_deref(),
            Some("0") | Some("off") | Some("false")
        ) {
            return None;
        }
        // fp16 tensor cores arrived with Volta (7.0). Below it cuBLAS runs the
        // f16 GEMM on the ordinary cores and the dp4a MMQ is the faster path —
        // llama.cpp's `fp16_mma_hardware_available` makes the same cut.
        use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
        let major = dev
            .cuda_stream()
            .context()
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(0);
        if major < 7 {
            return None;
        }
        Some(
            std::env::var("SWARMLLM_QMATMUL_CUBLAS_MIN_ROWS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(CUBLAS_MIN_ROWS),
        )
    })
}

/// SwarmLLM patch: `dst[y_cols, x_rows] = y[y_cols, x_cols] · Wᵀ` for a
/// quantized `W[x_rows, x_cols]`, as llama.cpp's cuBLAS path computes it: the
/// weight dequantized to f16, the activation cast to f16, one `cublasGemmEx`
/// on the tensor cores accumulating in f32.
///
/// Same layout as [`mul_mat_via_q8_1`]'s result (row-major `[y_cols, x_rows]`),
/// so a caller cannot tell which ran — except by precision: MMQ rounds the
/// activation to 8 bits (`q8_1`), this path to f16, so replies may move by a
/// near-tie between the two; judge them against a reference, not each other.
///
/// Temporary memory: the f16 weight (`x_rows · x_cols · 2` bytes — 117 MB for
/// an 8B's feed-forward) and the f16 activation, both freed on return into the
/// card's pool, which keeps them for the next matmul (`inference::cuda_pool`).
fn mul_mat_via_f16_cublas(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_cols: usize,
    f16_accumulate: bool,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    use cudarc::cublas::sys;
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    let k = x_cols;
    if y.len() != k * y_cols {
        crate::bail!("unexpected y size {}, {k} {y_cols}", y.len())
    }
    let w = dequantize_f16_slice(data, dtype, x_rows * k, dev)?;

    let el = y.len();
    let mut y16 = dev.alloc_fully_overwritten::<f16>(el)?;
    {
        // candle's own cast kernel; a null layout means contiguous, and it
        // assigns every one of the `el` elements.
        let func = dev.get_or_load_func("cast_f32_f16", &candle_kernels::CAST)?;
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(el as u32);
        let mut builder = func.builder();
        barg!(builder, el, 1usize, 0usize);
        builder.arg(y);
        builder.arg(&mut y16);
        unsafe { builder.launch(cfg) }.w()?;
    }

    // cuBLAS is column-major: with `W` row-major [x_rows, k] read as a k×x_rows
    // column-major matrix, op(A) = Aᵀ is the x_rows×k weight; `y` row-major
    // [y_cols, k] is k×y_cols column-major, op(B) = B. C is x_rows×y_cols
    // column-major, i.e. row-major [y_cols, x_rows] — MMQ's layout.
    let mut dst = dev.alloc_fully_overwritten::<f32>(x_rows * y_cols)?;
    let stream = dst.stream().clone();
    let blas = dev.cublas_handle();
    let (a, _ga) = w.device_ptr(&stream);
    let (b, _gb) = y16.device_ptr(&stream);
    if f16_accumulate {
        // llama.cpp's own default on NVIDIA (`ggml_cuda_mul_mat_cublas`): f16
        // accumulate into an f16 result, then widened — twice the tensor rate
        // of an f32 accumulate on GeForce cards.
        let mut dst16 = dev.alloc_fully_overwritten::<f16>(x_rows * y_cols)?;
        let alpha = f16::ONE;
        let beta = f16::ZERO;
        {
            let (c, _gc) = dst16.device_ptr_mut(&stream);
            unsafe {
                cudarc::cublas::result::gemm_ex(
                    *blas.handle(),
                    sys::cublasOperation_t::CUBLAS_OP_T,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    x_rows as i32,
                    y_cols as i32,
                    k as i32,
                    (&alpha) as *const f16 as *const _,
                    a as *const _,
                    sys::cudaDataType_t::CUDA_R_16F,
                    k as i32,
                    b as *const _,
                    sys::cudaDataType_t::CUDA_R_16F,
                    k as i32,
                    (&beta) as *const f16 as *const _,
                    c as *mut _,
                    sys::cudaDataType_t::CUDA_R_16F,
                    x_rows as i32,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
            }
            .w()?;
        }
        let el = x_rows * y_cols;
        let func = dev.get_or_load_func("cast_f16_f32", &candle_kernels::CAST)?;
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(el as u32);
        let mut builder = func.builder();
        barg!(builder, el, 1usize, 0usize);
        builder.arg(&dst16);
        builder.arg(&mut dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let (c, _gc) = dst.device_ptr_mut(&stream);
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *blas.handle(),
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                x_rows as i32,
                y_cols as i32,
                k as i32,
                (&alpha) as *const f32 as *const _,
                a as *const _,
                sys::cudaDataType_t::CUDA_R_16F,
                k as i32,
                b as *const _,
                sys::cudaDataType_t::CUDA_R_16F,
                k as i32,
                (&beta) as *const f32 as *const _,
                c as *mut _,
                sys::cudaDataType_t::CUDA_R_32F,
                x_rows as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        }
        .w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// [`mul_mat_via_f16_cublas`] accumulates in f16, as llama.cpp's cuBLAS path
/// does for a quantized weight on any card with fast f16 — except a weight
/// marked [`QCudaStorage::accumulate_in_f32`]. `SWARMLLM_QMATMUL_CUBLAS_ACC=32`
/// accumulates every weight in f32 (the A/B arm; the default until
/// 2026-10-09). Read once.
///
/// Judged 2026-10-09 (SwarmLLM FUTURE_WORK #147) on eight families at a
/// 4,096-token context on an RTX 3070, the prompt's last 128 positions'
/// logits against llama.cpp's: the median cosine moved by at most 2e-5 and
/// every argmax the two arms disagreed on was a near-tie (a gap of ≤ 0.29).
fn cublas_accumulates_in_f16() -> bool {
    static F16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F16.get_or_init(|| {
        std::env::var("SWARMLLM_QMATMUL_CUBLAS_ACC").ok().as_deref() != Some("32")
    })
}

fn dequantize_mul_mat_vec(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "dequantize_mul_mat_vec_q4_0_cuda",
        GgmlDType::Q4_1 => "dequantize_mul_mat_vec_q4_1_cuda",
        GgmlDType::Q5_0 => "dequantize_mul_mat_vec_q5_0_cuda",
        GgmlDType::Q5_1 => "dequantize_mul_mat_vec_q5_1_cuda",
        GgmlDType::Q8_0 => "dequantize_mul_mat_vec_q8_0_cuda",
        GgmlDType::Q2K => "dequantize_mul_mat_vec_q2_k",
        GgmlDType::Q3K => "dequantize_mul_mat_vec_q3_k",
        GgmlDType::Q4K => "dequantize_mul_mat_vec_q4_k",
        GgmlDType::Q5K => "dequantize_mul_mat_vec_q5_k",
        GgmlDType::Q6K => "dequantize_mul_mat_vec_q6_k",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(nrows)? };
    let block_num_y = ceil_div(nrows, GGML_CUDA_MMV_Y);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (block_num_y as u32, 1, 1),
        block_dim: (WARP_SIZE as u32, GGML_CUDA_MMV_Y as u32, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(y);
    builder.arg(&dst);
    barg!(builder, ncols as i32, nrows as i32);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn mul_mat_vec_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols * b_size {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    if b_size == 0 || b_size > 8 {
        crate::bail!("only bsize between 1 and 8 are supported, got {b_size}")
    }
    // Start by quantizing y
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        b_size * ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    // SwarmLLM patch: uninitialized, not zeroed. `alloc_zeros` is
    // `alloc` + a `cuMemsetD8Async`, and that memset is a full GPU submission
    // costing as much as a kernel launch (~10 us measured on WSL2) for a
    // buffer `quantize_q8_1` then overwrites in its entirety: its grid covers
    // `kx_padded`, and the padding tail is written explicitly
    // (`ix < kx ? x[...] : 0.0f`). That ternary only exists because the
    // reference implementation does not pre-zero either — upstream llama.cpp
    // hands `quantize_row_q8_1_cuda` pool memory.
    //
    // Measured: a decode token on a 22-layer model issued 323 memsets, ~14.7
    // per layer, i.e. two per quantized matmul (this buffer and `dst` below),
    // for 3.8 of 23 ms. See `docs/invariants/inference.md` § "A decode token
    // is bound by GPU submission count, not bandwidth".
    //
    // SwarmLLM patch: reuse ONE quantization across weight matrices that share
    // this activation — see `shared_activation`. Outside such a group this is
    // exactly the old behaviour.
    let y_q8_1: std::sync::Arc<CudaSlice<u8>> = match shared_take(ncols, b_size, Q8_1_ROWS) {
        Some(existing) => existing,
        None => {
            let mut buf = dev.alloc_fully_overwritten::<u8>(y_size_in_bytes)?;
            quantize_q8_1(y, &mut buf, ncols, b_size, dev)?;
            let buf = std::sync::Arc::new(buf);
            shared_put(ncols, b_size, Q8_1_ROWS, &buf);
            buf
        }
    };

    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "mul_mat_vec_q4_0_q8_1_cuda",
        GgmlDType::Q4_1 => "mul_mat_vec_q4_1_q8_1_cuda",
        GgmlDType::Q5_0 => "mul_mat_vec_q5_0_q8_1_cuda",
        GgmlDType::Q5_1 => "mul_mat_vec_q5_1_q8_1_cuda",
        GgmlDType::Q8_0 => "mul_mat_vec_q8_0_q8_1_cuda",
        GgmlDType::Q2K => "mul_mat_vec_q2_K_q8_1_cuda",
        GgmlDType::Q3K => "mul_mat_vec_q3_K_q8_1_cuda",
        GgmlDType::Q4K => "mul_mat_vec_q4_K_q8_1_cuda",
        GgmlDType::Q5K => "mul_mat_vec_q5_K_q8_1_cuda",
        GgmlDType::Q6K => "mul_mat_vec_q6_K_q8_1_cuda",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let kernel_name = format!("{kernel_name}{b_size}");
    let func = dev.get_or_load_func(&kernel_name, &candle_kernels::QUANTIZED)?;
    // SwarmLLM patch: uninitialized — see the note on `y_q8_1` above.
    // `mul_mat_vec_q` ASSIGNS every element it is responsible for
    // (`dst[j*nrows_dst + row0 + threadIdx.x] = tmp[...]`, not `+=`), and the
    // grid covers every row: `nblocks` is `nrows` at b_size 1 and
    // `ceil_div(nrows, 2)` with `rows_per_cuda_block == 2` above it. Nothing
    // reads `dst` before that write, so the zeroing is dead work.
    // ⚠ A kernel that ever ACCUMULATES into `dst` would need the zeroing back.
    let dst = dev.alloc_fully_overwritten::<f32>(nrows * b_size)?;
    // https://github.com/ggerganov/llama.cpp/blob/facb8b56f8fd3bb10a693bf0943ae9d69d0828ef/ggml-cuda/mmvq.cu#L98
    let (nblocks, nwarps) = match b_size {
        1 => (nrows as u32, 4),
        2..=4 => ((nrows as u32).div_ceil(2), 4),
        5..=8 => ((nrows as u32).div_ceil(2), 2),
        _ => crate::bail!("unexpected bsize {b_size}"),
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nblocks, 1, 1),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(&*y_q8_1);
    builder.arg(&dst);
    barg!(
        builder,
        /* ncols_x */ ncols as i32,
        /* nrows_x */ nrows as i32,
        /* nrows_y */ ncols_padded as i32,
        /* nrows_dst */ nrows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// SwarmLLM patch: one Q8_1 quantization of an activation, reused by every
/// weight matrix that shares it.
///
/// **What it fixes.** `q`, `k` and `v` are three matmuls against the SAME
/// post-attention-norm hidden state, and the FFN's `gate` and `up` are two
/// against the same post-FFN-norm one. Each quantized its input independently,
/// so a layer ran **7 `quantize_q8_1` launches for 7 matmuls when only 4 of the
/// inputs were distinct** — measured at 7.05 quantize launches per layer
/// against 7.04 matmuls on tinyllama, with 3 per layer recomputing a buffer
/// byte-for-byte identical to one just built.
///
/// **Why this shape and not a cache.** A cache keyed on the input's device
/// pointer is unsound here: `cuMemAllocAsync` recycles addresses, so a freed
/// buffer's pointer reappears attached to different data, and a stale hit would
/// be a silently wrong reply rather than a failure. [`shared_activation`]
/// instead owns the whole window — it borrows `xs` for the duration, so the
/// storage cannot be freed, and it runs the matmuls itself, so no unrelated
/// matmul can observe the slot. That is "make the wrong call unrepresentable"
/// rather than "document that forgetting it is the bug".
///
/// The `(ncols, b_size)` check is a belt-and-braces assertion, not the
/// correctness argument: within one group every weight shares the activation's
/// `k` by construction.
///
/// → `docs/invariants/inference.md` § "A decode token is bound by GPU
///   submission COUNT, not bandwidth"
struct SharedQ8 {
    ncols: usize,
    b_size: usize,
    /// Which arrangement `buf` is in: [`Q8_1_ROWS`] for candle's own kernels,
    /// else the `block_q8_1_mmq` scale layout llama.cpp's MMQ reads
    /// ([`mul_mat_via_mmq_mma`]). A Q4_K weight and a Q6_K one read the same
    /// activation in different layouts, so the layout is part of the key.
    layout: i32,
    buf: std::sync::Arc<CudaSlice<u8>>,
}

/// [`SharedQ8::layout`] of the plain `block_q8_1` rows candle's kernels read.
const Q8_1_ROWS: i32 = -1;

std::thread_local! {
    /// `Some` only while [`shared_activation`] is on the stack.
    static SHARED_Q8: std::cell::RefCell<Option<Option<SharedQ8>>> =
        const { std::cell::RefCell::new(None) };
}

/// The quantized activation for this group, if one has been built and matches.
fn shared_take(
    ncols: usize,
    b_size: usize,
    layout: i32,
) -> Option<std::sync::Arc<CudaSlice<u8>>> {
    SHARED_Q8.with(|s| {
        let slot = s.borrow();
        match &*slot {
            Some(Some(sh)) if sh.ncols == ncols && sh.b_size == b_size && sh.layout == layout => {
                Some(sh.buf.clone())
            }
            _ => None,
        }
    })
}

/// Record the quantization so the rest of the group can reuse it. A no-op
/// unless a group is active, which is what keeps the non-grouped path identical.
fn shared_put(ncols: usize, b_size: usize, layout: i32, buf: &std::sync::Arc<CudaSlice<u8>>) {
    SHARED_Q8.with(|s| {
        let mut slot = s.borrow_mut();
        if let Some(inner @ None) = slot.as_mut() {
            *inner = Some(SharedQ8 {
                ncols,
                b_size,
                layout,
                buf: buf.clone(),
            });
        }
    })
}

/// Run `f`, letting the quantized-matmul path quantize the shared activation
/// once and reuse it.
///
/// `f` must only run matmuls whose activation is the one the caller intends to
/// share; [`crate::quantized::QMatMul::forward_shared`] is the only public way
/// in, and it holds that invariant by owning the loop.
pub(crate) fn with_shared_activation<T>(f: impl FnOnce() -> T) -> T {
    /// Clears the slot however the group ends. **Not cosmetic**: on a panic an
    /// early return would leave a quantized activation visible to the next
    /// matmul on this thread, and a stale reuse is a silently wrong reply.
    struct Clear;
    impl Drop for Clear {
        fn drop(&mut self) {
            SHARED_Q8.with(|s| *s.borrow_mut() = None);
        }
    }
    SHARED_Q8.with(|s| *s.borrow_mut() = Some(None));
    let _clear = Clear;
    f()
}

/// SwarmLLM patch: whether a product of `rows` activation rows against a
/// weight of `dtype` goes to llama.cpp's int8 tensor-core MMQ
/// ([`mul_mat_via_mmq_mma`]), and if so in which activation layout.
///
/// **llama.cpp's rule**: on a card with int8 tensor cores
/// (`turing_mma_available`) every batch above the vector kernel's eight rows
/// takes MMQ — `ggml_cuda_should_use_mmq` returns true before it asks about
/// batch size. This repo's [`CUBLAS_MIN_ROWS`] rule was llama.cpp's for the
/// OLD dp4a MMQ candle vendored, and now applies only where this kernel does
/// not run.
///
/// Measured before it (RTX 3070, Qwen2.5 Coder 7B Q4_K_M whole, 300-token
/// depth; SwarmLLM FUTURE_WORK #325): 16 rows 83.3 ms here against 22.9 in
/// llama.cpp, 32 rows 84.4 vs 25.8, 64 rows 107.8 (cuBLAS) vs 37.4.
///
/// `SWARMLLM_QMATMUL_MMQ=dp4a` keeps the old kernels (the A/B in one
/// binary); `SWARMLLM_QMATMUL_MMQ_MAX_ROWS` caps the rows it takes, for a
/// measurement. After a launch fails it is off for the process: the old
/// kernels answer, once with a line on stderr. A card the library has no
/// image for is told no before any launch (`swarm_mmq_available` probes for
/// one). The library carries its floor's PTX, so a card newer than the
/// build's CUDA_COMPUTE_CAP has one (candle-kernels `build.rs`).
fn mmq_mma_layout(dev: &CudaDevice, dtype: GgmlDType, rows: usize) -> Option<i32> {
    static ON: std::sync::OnceLock<(bool, usize)> = std::sync::OnceLock::new();
    let (on, max_rows) = *ON.get_or_init(|| {
        let off = matches!(
            std::env::var("SWARMLLM_QMATMUL_MMQ").ok().as_deref(),
            Some("dp4a") | Some("0") | Some("off") | Some("false")
        );
        let max_rows = std::env::var("SWARMLLM_QMATMUL_MMQ_MAX_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(usize::MAX);
        // The runtime API asks the CURRENT context's device: bind this one.
        let bound = dev.cuda_stream().context().bind_to_thread().is_ok();
        // SAFETY: a query of the current device; no pointers.
        let available =
            !off && bound && unsafe { candle_kernels::ffi::swarm_mmq_available() } == 1;
        (available, max_rows)
    });
    if !on || rows > max_rows || MMQ_MMA_FAILED.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    // SAFETY: a table lookup.
    let layout = unsafe { candle_kernels::ffi::swarm_mmq_q8_1_layout(dtype.to_u32() as i32) };
    if layout < 0 {
        return None;
    }
    mmq_prepare_few_rows_once(dev, dtype);
    Some(layout)
}

/// The fewest rows below the vector kernel's limit (8) that llama.cpp's MMQ
/// takes instead, per weight type, where [`mmq_takes_few_rows`] allows it.
///
/// A k-quant costs more to decode and the vector kernel redoes that for every
/// row, so MMQ wins sooner — llama.cpp's own reasoning in
/// `ggml_cuda_should_use_mmvq`, whose per-type tables cover Volta, Ada,
/// Blackwell and Orin and keep every type on the vector kernel to 8 rows
/// elsewhere, this card's generation (Ampere) included. Ours sit lower because
/// our vector kernel is candle's copy of an older llama.cpp one, which at 8
/// rows costs ~1.5x its 7. Measured (RTX 3070 Laptop, whole models, decode
/// steps captured as graphs, two passes, 2026-10-22; SwarmLLM
/// `docs/invariants/inference.md` § "A card multiplies several rows"): Q4_K_M
/// files (Qwen2.5-Coder-7B, Llama-3.2-3B) gain from 4 rows and split at 3;
/// Q8_0 files (Qwen2.5-0.5B, Qwen2.5-Coder-1.5B, Llama-3.2-1B, Qwen3-1.7B)
/// lose at 5, split at 6 and gain from 7. The other 32-value formats decode as
/// simply as Q8_0 and take its line, unmeasured.
fn mmq_fewest_rows_default(dtype: GgmlDType) -> usize {
    match dtype {
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => 4,
        _ => 7,
    }
}

/// The smallest weight, in bytes, whose product of a few rows MMQ takes
/// ([`mmq_takes_few_rows`]). Below it the vector kernel is faster: a weight
/// of under ~7 tiles of 128 rows runs MMQ as stream-k plus a fixup pass, and a
/// small weight is quick to read anyway. At 8 rows on an RTX 3070 Laptop
/// (nsys, graph nodes traced, 2026-10-22): a Qwen2.5-0.5B's 128 × 896 Q8_0 key
/// (0.12 MB) 3.7 µs on the vector kernel against 8.7 + 4.8 (fixup) on MMQ; its
/// 4,864 × 896 feed-forward (4.6 MB) 43.5 against 24.7, its 151,936-row head
/// (145 MB) 1,190 against 560. llama.cpp draws its line by rows alone (per
/// card and type, [`mmq_fewest_rows_default`]).
const MMQ_FEW_ROWS_LEAST_BYTES: usize = 2 << 20;

/// SwarmLLM patch: whether a product of `rows` ≤ 8 activation rows against a
/// weight of `weight_bytes` goes to llama.cpp's MMQ rather than the vector
/// kernel — from the rows of [`mmq_few_rows_line`] (per type;
/// `SWARMLLM_QMATMUL_MMQ_MIN_ROWS` sets one line for all, above 8 = never) and
/// its bytes (`SWARMLLM_QMATMUL_MMQ_FEW_ROWS_BYTES`, 0 = any weight). While
/// the device's stream is being captured, only once the kernel such a product
/// picks is ready for it ([`mmq_few_rows_ready`]).
///
/// A CUDA graph records forwards of up to 8 positions — a drafter's tree level
/// of 8 rows among them since 2026-10-22 (`SplitModel::tree_steps_may_be_captured`)
/// — and at 8 rows the vector kernel reads a Qwen2.5-0.5B's weights at about a
/// quarter of the card's bandwidth: 43.5 µs for a feed-forward up (896 × 4,864)
/// where one row takes 13.3, 1.19 ms for the 151,936-row head where one row
/// takes 0.36 (RTX 3070 Laptop, nsys with graph nodes traced; SwarmLLM
/// `docs/invariants/inference.md` § "A card multiplies several rows").
fn mmq_takes_few_rows(
    dev: &CudaDevice,
    dtype: GgmlDType,
    rows: usize,
    weight_bytes: usize,
) -> bool {
    let (fewest, least_bytes) = mmq_few_rows_line(dtype);
    if rows < fewest.max(2) || weight_bytes < least_bytes {
        return false;
    }
    // Checked BEFORE `mmq_mma_layout`, whose first call asks the driver about
    // the card: when the kernel is ready, that call has been made.
    if stream_is_capturing(dev) && !(mmq_may_be_captured() && mmq_few_rows_ready(dev, dtype)) {
        return false;
    }
    mmq_mma_layout(dev, dtype, rows).is_some()
}

/// `SWARMLLM_QMATMUL_MMQ_CAPTURED=0` keeps every CAPTURED product of ≤ 8 rows
/// on the vector kernel, as before 2026-10-22 — the A/B in one binary.
fn mmq_may_be_captured() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_QMATMUL_MMQ_CAPTURED").ok().as_deref(),
            Some("0") | Some("off") | Some("false")
        )
    })
}

/// SwarmLLM patch: `(fewest rows, fewest weight bytes)` of a weight of `dtype`
/// whose product of ≤ 8 rows MMQ takes in this process — the one answer
/// [`mmq_takes_few_rows`] reads, public for the card tests:
/// [`mmq_fewest_rows_default`] unless `SWARMLLM_QMATMUL_MMQ_MIN_ROWS` sets one
/// line for every type, [`MMQ_FEW_ROWS_LEAST_BYTES`] unless
/// `SWARMLLM_QMATMUL_MMQ_FEW_ROWS_BYTES` moves it. The environment is read once.
pub fn mmq_few_rows_line(dtype: GgmlDType) -> (usize, usize) {
    static ENV: std::sync::OnceLock<(Option<usize>, usize)> = std::sync::OnceLock::new();
    let (rows, bytes) = *ENV.get_or_init(|| {
        let read = |name: &str| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
        };
        (
            read("SWARMLLM_QMATMUL_MMQ_MIN_ROWS"),
            read("SWARMLLM_QMATMUL_MMQ_FEW_ROWS_BYTES").unwrap_or(MMQ_FEW_ROWS_LEAST_BYTES),
        )
    });
    (rows.unwrap_or_else(|| mmq_fewest_rows_default(dtype)), bytes)
}

/// Whether the device's stream is recording a CUDA graph.
fn stream_is_capturing(dev: &CudaDevice) -> bool {
    !matches!(
        dev.cuda_stream().capture_status(),
        Ok(cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE)
    )
}

/// Per device ordinal (the first 16), one bit per weight type
/// (`GgmlDType::to_u32`): the few-rows kernel's shared-memory limit is raised.
static MMQ_FEW_ROWS_READY: [std::sync::atomic::AtomicU64; 16] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 16];

/// The same, for a type whose raising was tried, so a refusal is not retried
/// on every product.
static MMQ_FEW_ROWS_TRIED: [std::sync::atomic::AtomicU64; 16] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 16];

/// The slot and bit of `dtype` on this device, or `None` past the 16th card.
fn mmq_few_rows_slot(dev: &CudaDevice, dtype: GgmlDType) -> Option<(usize, u64)> {
    let ordinal = dev.cuda_stream().context().ordinal();
    let bit = 1u64.checked_shl(dtype.to_u32())?;
    (ordinal < MMQ_FEW_ROWS_READY.len()).then_some((ordinal, bit))
}

/// Whether a CAPTURED product of ≤ 8 rows against a weight of `dtype` may take
/// MMQ: its kernel's shared-memory limit was raised outside a capture. That
/// limit is raised by MMQ's first launch of each tile width, a driver call a
/// capture must not see (gotcha #841's class); with it raised, a product
/// enqueues kernels and stream-ordered allocations only. An atomic read —
/// nothing here touches the driver.
fn mmq_few_rows_ready(dev: &CudaDevice, dtype: GgmlDType) -> bool {
    mmq_few_rows_slot(dev, dtype).is_some_and(|(slot, bit)| {
        MMQ_FEW_ROWS_READY[slot].load(std::sync::atomic::Ordering::Acquire) & bit != 0
    })
}

/// Raises the few-rows kernel's limit for `dtype` on this device the first
/// time MMQ answers for that type outside a capture — the prompt a drafter
/// reads before its first tree level, a tree check of 9+ rows, an uncaptured
/// step of a few rows — so a later captured level can take it.
fn mmq_prepare_few_rows_once(dev: &CudaDevice, dtype: GgmlDType) {
    use std::sync::atomic::Ordering;
    let Some((slot, bit)) = mmq_few_rows_slot(dev, dtype) else {
        return;
    };
    if MMQ_FEW_ROWS_TRIED[slot].load(Ordering::Acquire) & bit != 0 || stream_is_capturing(dev) {
        return;
    }
    MMQ_FEW_ROWS_TRIED[slot].fetch_or(bit, Ordering::AcqRel);
    if dev.cuda_stream().context().bind_to_thread().is_err() {
        return;
    }
    // SAFETY: plans two products and launches nothing; no pointer is read.
    if unsafe { candle_kernels::ffi::swarm_mmq_prepare_few_rows(dtype.to_u32() as i32) } == 0 {
        MMQ_FEW_ROWS_READY[slot].fetch_or(bit, Ordering::AcqRel);
    }
}

/// Set after an MMQ launch fails; see [`mmq_mma_layout`].
static MMQ_MMA_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Products [`mul_mat_via_mmq_mma`] has enqueued in this process.
static MMQ_MMA_PRODUCTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// SwarmLLM patch: how many quantized products llama.cpp's tensor-core MMQ
/// has run in this process — the check that it fired, for a test or a bench
/// (SwarmLLM's diagnosis rule 4), never a decision.
pub fn mmq_mma_products() -> usize {
    MMQ_MMA_PRODUCTS.load(std::sync::atomic::Ordering::Relaxed)
}

/// SwarmLLM patch: why [`mul_mat_via_mmq_mma`] ran no product.
enum MmqFailure {
    /// The kernels refused or failed — a status, a launch, a shape they do not
    /// take. MMQ is not tried again in this process ([`MMQ_MMA_FAILED`]).
    Refused(crate::Error),
    /// This call found no card memory for one of its buffers. The older
    /// kernels are tried for this product and MMQ stays on: a card full of a
    /// split's layers fails an allocation now and then, and turning MMQ off for
    /// good on one left every later check 2-3x slower until a restart (the
    /// v0.3.247 pre-release review; gotcha #647's recovery half — a latch one
    /// transient condition must not set for the process's life).
    NoRoom(crate::Error),
}

/// Products [`mul_mat_via_mmq_mma`] found no card memory for, in this process.
static MMQ_MMA_NO_ROOM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// SwarmLLM patch: how many products found no card memory for MMQ's buffers
/// and went to the older kernels instead ([`MmqFailure::NoRoom`]) — for a test
/// or a bench, never a decision.
pub fn mmq_mma_no_room() -> usize {
    MMQ_MMA_NO_ROOM.load(std::sync::atomic::Ordering::Relaxed)
}

/// Set by [`fail_the_next_mmq_allocation`]; taken by the next product.
static MMQ_FAULT_NO_ROOM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// SwarmLLM patch, a TEST hook: the next MMQ product's first allocation
/// fails as if the card had no room. Nothing in a running node calls it.
pub fn fail_the_next_mmq_allocation() {
    MMQ_FAULT_NO_ROOM.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// An allocation for MMQ: a failure is [`MmqFailure::NoRoom`].
fn mmq_alloc<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    dev: &CudaDevice,
    len: usize,
) -> std::result::Result<CudaSlice<T>, MmqFailure> {
    if MMQ_FAULT_NO_ROOM.swap(false, std::sync::atomic::Ordering::Relaxed) {
        return Err(MmqFailure::NoRoom(crate::Error::Msg(
            "an injected allocation failure (fail_the_next_mmq_allocation)".into(),
        )));
    }
    dev.alloc_fully_overwritten::<T>(len).map_err(MmqFailure::NoRoom)
}

/// SwarmLLM patch: `dst[y_cols][x_rows] = y · xᵀ` with llama.cpp's int8
/// tensor-core MMQ (candle-kernels `src/ggml_mmq`, its `mmq.cuh` verbatim,
/// launched on this device's stream). Same result layout as
/// [`mul_mat_via_q8_1`]. The activation is quantized ONCE per group of
/// weights reading it in the same `layout` ([`shared_take`]).
///
/// Every buffer is uninitialised, as llama.cpp's pool hands them out: the
/// quantizer writes every row it is given (the tail past the last row is read
/// only into tile rows the write-back skips), the kernel assigns every element
/// of `dst`, and each fixup slot is written by the block that leaves a tile
/// unfinished before the fixup kernel reads it.
#[allow(clippy::too_many_arguments)]
fn mul_mat_via_mmq_mma(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    layout: i32,
    x_rows: usize,
    x_cols: usize,
    y_cols: usize,
    dev: &CudaDevice,
) -> std::result::Result<CudaStorage, MmqFailure> {
    use cudarc::driver::DevicePtr;
    let refused = |e: String| MmqFailure::Refused(crate::Error::Msg(e));
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < x_rows * x_cols {
        return Err(refused(format!(
            "unexpected lhs size {data_elems}, {x_rows} {x_cols}"
        )));
    }
    if y.len() != x_cols * y_cols {
        return Err(refused(format!(
            "unexpected y size {}, {x_cols} {y_cols}",
            y.len()
        )));
    }
    let stream = dev.cuda_stream();
    stream
        .context()
        .bind_to_thread()
        .w()
        .map_err(MmqFailure::Refused)?;
    let raw_stream = stream.cu_stream() as i64;
    let ggml_type = dtype.to_u32() as i32;
    let status = |what: &str, code: i32| -> std::result::Result<(), MmqFailure> {
        if code == 0 {
            Ok(())
        } else {
            Err(refused(format!(
                "llama.cpp MMQ: {what} failed with status {code}"
            )))
        }
    };

    let y_mmq: std::sync::Arc<CudaSlice<u8>> = match shared_take(x_cols, y_cols, layout) {
        Some(existing) => existing,
        None => {
            // SAFETY: a size computation.
            let bytes =
                unsafe { candle_kernels::ffi::swarm_mmq_q8_1_bytes(x_cols as i64, y_cols as i64) };
            let buf = mmq_alloc::<u8>(dev, bytes)?;
            {
                let (y_ptr, _y_guard) = y.device_ptr(&stream);
                let (buf_ptr, _buf_guard) = buf.device_ptr(&stream);
                // SAFETY: `y` holds `y_cols` rows of `x_cols` floats and `buf`
                // the bytes `swarm_mmq_q8_1_bytes` asked for, both on `stream`.
                let code = unsafe {
                    candle_kernels::ffi::swarm_mmq_quantize(
                        y_ptr as *const f32,
                        buf_ptr as *mut core::ffi::c_void,
                        ggml_type,
                        x_cols as i64,
                        y_cols as i64,
                        x_cols as i64,
                        raw_stream,
                    )
                };
                status("quantizing the activation", code)?;
            }
            let buf = std::sync::Arc::new(buf);
            shared_put(x_cols, y_cols, layout, &buf);
            buf
        }
    };

    let (x_ptr, _x_guard) = data.inner.device_ptr(&stream);
    let (y_ptr, _y_guard) = y_mmq.device_ptr(&stream);
    let mut need: usize = 0;
    // SAFETY: `launch == 0` only computes the fixup size; nothing is read.
    let code = unsafe {
        candle_kernels::ffi::swarm_mmq_matmul(
            ggml_type,
            x_ptr as *const core::ffi::c_void,
            y_ptr as *const core::ffi::c_void,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            &mut need,
            x_cols as i64,
            x_rows as i64,
            y_cols as i64,
            0,
            raw_stream,
        )
    };
    status("planning the product", code)?;
    let dst = mmq_alloc::<f32>(dev, x_rows * y_cols)?;
    let fixup = if need > 0 {
        Some(mmq_alloc::<f32>(dev, need)?)
    } else {
        None
    };
    {
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
        let fixup_ptr = fixup
            .as_ref()
            .map(|f| f.device_ptr(&stream).0 as *mut f32)
            .unwrap_or(std::ptr::null_mut());
        // SAFETY: every pointer is a live allocation on `stream` of the size
        // the planning call and the shapes above give; the kernels are
        // enqueued on `stream`, which orders the frees of `fixup` and of a
        // dropped `y_mmq` after them (stream-ordered allocation).
        let code = unsafe {
            candle_kernels::ffi::swarm_mmq_matmul(
                ggml_type,
                x_ptr as *const core::ffi::c_void,
                y_ptr as *const core::ffi::c_void,
                dst_ptr as *mut f32,
                fixup_ptr,
                need,
                &mut need,
                x_cols as i64,
                x_rows as i64,
                y_cols as i64,
                1,
                raw_stream,
            )
        };
        status("the product", code)?;
    }
    drop(fixup);
    MMQ_MMA_PRODUCTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

#[allow(clippy::too_many_arguments)]
fn mul_mat_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_rows: usize,
    y_cols: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < x_rows * x_cols {
        crate::bail!("unexpected lhs size {}, {x_rows} {x_cols}", data_elems)
    }
    if y.len() != y_rows * y_cols {
        crate::bail!("unexpected y size {}, {y_rows} {y_cols}", y.len())
    }
    if x_cols != y_rows {
        crate::bail!("unexpected x/y size {x_rows} {x_cols} {y_rows} {y_cols}")
    }
    let k = x_cols;
    // Start by quantizing y
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        k_padded * y_cols * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    // SwarmLLM patch: uninitialized, and shared across the group — same kernel
    // and the same reasoning as the `mul_mat_vec_q8_1` path above.
    //
    // ⚠ **This path had to be fixed too, and nearly was not.** Sharing landed
    // on the vec path first, and a prefill still showed 7.05 `quantize_q8_1`
    // launches per layer where decode showed 4.05 — because a prompt takes the
    // MMQ kernel and not the vec one. That is
    // `.claude/rules/architecture.md` § "One invariant, N paths", this
    // project's most-repeated defect, and only the per-kernel measurement
    // caught it.
    //
    // It matters MORE here than at decode: the redundant quantization covers
    // every row of the prompt chunk rather than one position, so it is real
    // work and not a bare submission. It pays into time-to-first-token.
    let y_q8_1: std::sync::Arc<CudaSlice<u8>> = match shared_take(k, y_cols, Q8_1_ROWS) {
        Some(existing) => existing,
        None => {
            let mut buf = dev.alloc_fully_overwritten::<u8>(y_size_in_bytes)?;
            quantize_q8_1(y, &mut buf, k, y_cols, dev)?;
            let buf = std::sync::Arc::new(buf);
            shared_put(k, y_cols, Q8_1_ROWS, &buf);
            buf
        }
    };

    let (kernel_name, mmq_x, mmq_y) = match dtype {
        GgmlDType::Q4_0 => ("mul_mat_q4_0", 64, 128),
        GgmlDType::Q4_1 => ("mul_mat_q4_1", 64, 128),
        GgmlDType::Q5_0 => ("mul_mat_q5_0", 128, 64),
        GgmlDType::Q5_1 => ("mul_mat_q5_1", 128, 64),
        GgmlDType::Q8_0 => ("mul_mat_q8_0", 128, 64),
        GgmlDType::Q2K => ("mul_mat_q2_K", 64, 128),
        GgmlDType::Q3K => ("mul_mat_q3_K", 128, 128),
        GgmlDType::Q4K => ("mul_mat_q4_K", 64, 128),
        GgmlDType::Q5K => ("mul_mat_q5_K", 64, 128),
        GgmlDType::Q6K => ("mul_mat_q6_K", 64, 64),
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    // SwarmLLM patch: uninitialized. The MMQ epilogue assigns
    // `dst[col_dst*nrows_dst + row_dst]` for every in-range element, skipping
    // only what is out of range (`col_dst >= ncols_dst` returns,
    // `row_dst >= nrows_dst` continues), and the grid is `ceil_div` of both
    // dimensions, so each in-range element is covered exactly once.
    let dst = dev.alloc_fully_overwritten::<f32>(x_rows * y_cols)?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (
            ceil_div(x_rows, mmq_y) as u32,
            ceil_div(y_cols, mmq_x) as u32,
            1,
        ),
        block_dim: (WARP_SIZE as u32, 4, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(/* vx */ &data.inner);
    builder.arg(/* vy */ &*y_q8_1);
    builder.arg(/* dst */ &dst);
    barg!(
        builder,
        /* ncols_x */ x_cols as i32,
        /* nrows_x */ x_rows as i32,
        /* ncols_y */ y_cols as i32,
        /* nrows_y */ k_padded as i32,
        /* nrows_dst */ x_rows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

#[allow(clippy::too_many_arguments)]
fn indexed_moe_forward_fused_q8_1_input(
    weight: &CudaView<u8>,
    w_shape: &crate::Shape, //[num_experts, n, k]
    w_dtype: GgmlDType,
    input: &CudaSlice<f32>,
    in_shape: &crate::Shape, //[batch, topk or 1, k]
    ids: &CudaView<u32>,
    idx_shape: &crate::Shape, //[batch, topk]
    dev: &CudaDevice,
) -> Result<(CudaStorage, crate::Shape)> {
    let (_, n, k) = w_shape.dims3()?;
    let batch = in_shape.dims()[0];
    let input_dim1 = in_shape.dims()[1];

    let topk = idx_shape.dims()[1];
    assert!(batch == idx_shape.dims()[0], "batch dim not match!");

    // Quantize input into q8_1.
    let total_rows = batch * input_dim1;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = k_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;
    let y_size_in_bytes = total_rows * dst_row_size_bytes;
    // SwarmLLM patch: uninitialized — `quantize_q8_1` overwrites all of it,
    // as above. The `out` buffer below is deliberately left zeroed: the
    // `indexed_moe_forward_*` kernels' coverage of it has not been read, and
    // an unverified partial write would leak garbage into a reply.
    let mut input_quant = dev.alloc_fully_overwritten::<u8>(y_size_in_bytes)?;

    let input_view = input.slice(0..);
    quantize_q8_1(&input_view, &mut input_quant, k, total_rows, dev)?;

    // output buffer
    let outsize = batch * topk * n;
    let out = dev.alloc_zeros::<f32>(outsize)?;

    let kernel_name = match w_dtype {
        GgmlDType::Q2K => "indexed_moe_forward_q2k_q8_1",
        GgmlDType::Q3K => "indexed_moe_forward_q3k_q8_1",
        GgmlDType::Q4K => "indexed_moe_forward_q4k_q8_1",
        GgmlDType::Q5K => "indexed_moe_forward_q5k_q8_1",
        GgmlDType::Q6K => "indexed_moe_forward_q6k_q8_1",
        GgmlDType::Q8_0 => "indexed_moe_forward_q8_0_q8_1",
        _ => crate::bail!("unsupported dtype for indexed_moe_forward {w_dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let (nblocks, nwarps) = (n as u32, 4);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nblocks, batch as u32, topk as u32),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(weight);
    builder.arg(&input_quant);
    builder.arg(ids);
    builder.arg(&out);

    barg!(
        builder,
        n as i32,
        k as i32,
        batch as i32,
        topk as i32,
        k_padded as i32,
        input_dim1 as i32
    );
    unsafe { builder.launch(cfg) }.w()?;

    let mut out_shape = in_shape.dims().to_vec();
    out_shape.pop();
    out_shape.push(n);
    out_shape[1] = topk;
    Ok((
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        out_shape.into(),
    ))
}

impl QCudaStorage {
    pub fn indexed_moe_forward(
        &self,
        self_shape: &crate::Shape, //[num_experts, n, k]
        input: &CudaStorage,       //[batch, topk or 1, k]
        input_l: &crate::Layout,
        ids: &CudaStorage, //[batch, topk]
        ids_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        if matches!(
            self.dtype(),
            GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        ) {
            let input_storage = input.as_cuda_slice::<f32>()?;
            let ids_storage = ids.as_cuda_slice::<u32>()?;
            indexed_moe_forward_fused_q8_1_input(
                &self.data.inner.slice(0..),
                self_shape, //[num_experts, n, k]
                self.dtype(),
                input_storage,
                input_l.shape(), //[batch, topk or 1, k]
                &ids_storage.slice(0..),
                ids_l.shape(), //[batch, topk]
                &self.device,
            )
        } else {
            crate::bail!(
                "The given quantized dtype {:?} is not supported for indexed_moe_forward!",
                self.dtype()
            );
        }
    }

    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let padded_size_in_bytes =
            ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let inner = device.alloc_zeros::<u8>(padded_size_in_bytes)?;
        Ok(QCudaStorage {
            data: PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            },
            device: device.clone(),
            dtype,
            accumulate_f32: false,
        })
    }

    /// Keep this weight's products in f32 on the f16 cuBLAS path (llama.cpp's
    /// `ggml_prec_set_acc(.., GGML_PREC_F32)`), for the weights an
    /// architecture's numbers do not survive an f16 accumulator in.
    pub fn accumulate_in_f32(&mut self) {
        self.accumulate_f32 = true;
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<CudaStorage> {
        fn deq<T: GgmlType>(buffer: &[u8], n: usize, dst: &mut [f32]) {
            let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, n) };
            let vec = slice.to_vec();
            T::to_float(&vec, dst)
        }

        if has_dequantize_kernel(self.dtype) {
            return dequantize_f32(&self.data, self.dtype, elem_count, self.device());
        }
        // Run the dequantization on cpu.

        let buffer = self
            .device
            .clone_dtoh(&self.data.inner.slice(..self.data.len))?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => deq::<f32>(&buffer, block_len, &mut out),
            GgmlDType::F16 => deq::<half::f16>(&buffer, block_len, &mut out),
            GgmlDType::BF16 => deq::<half::bf16>(&buffer, block_len, &mut out),
            GgmlDType::Q4_0 => deq::<crate::quantized::BlockQ4_0>(&buffer, block_len, &mut out),
            GgmlDType::Q4_1 => deq::<crate::quantized::BlockQ4_1>(&buffer, block_len, &mut out),
            GgmlDType::Q5_0 => deq::<crate::quantized::BlockQ5_0>(&buffer, block_len, &mut out),
            GgmlDType::Q5_1 => deq::<crate::quantized::BlockQ5_1>(&buffer, block_len, &mut out),
            GgmlDType::Q8_0 => deq::<crate::quantized::BlockQ8_0>(&buffer, block_len, &mut out),
            GgmlDType::Q8_1 => deq::<crate::quantized::BlockQ8_1>(&buffer, block_len, &mut out),
            GgmlDType::Q2K => deq::<crate::quantized::BlockQ2K>(&buffer, block_len, &mut out),
            GgmlDType::Q3K => deq::<crate::quantized::BlockQ3K>(&buffer, block_len, &mut out),
            GgmlDType::Q4K => deq::<crate::quantized::BlockQ4K>(&buffer, block_len, &mut out),
            GgmlDType::Q5K => deq::<crate::quantized::BlockQ5K>(&buffer, block_len, &mut out),
            GgmlDType::Q6K => deq::<crate::quantized::BlockQ6K>(&buffer, block_len, &mut out),
            GgmlDType::Q8K => deq::<crate::quantized::BlockQ8K>(&buffer, block_len, &mut out),
        }

        self.device
            .storage_from_cpu_storage(&crate::CpuStorage::F32(out))
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<CudaStorage> {
        if has_dequantize_kernel(self.dtype) {
            return dequantize_f16(&self.data, self.dtype, elem_count, self.device());
        }
        // SwarmLLM patch: an unquantized GGUF (F16 / BF16 / F32) has no
        // dequantize kernel, and this entry point used to `bail!` on it while
        // its `dequantize` sibling quietly fell back to the host. Nothing
        // upstream of here knows the difference, so the model LOADED on a CUDA
        // node and then failed every single request with "unsupported dtype
        // for dequantize F16" — while the same file served correctly on CPU.
        //
        // Go through the sibling (which owns the host fallback) and cast on
        // the device, so there is exactly one place that knows how to handle a
        // kernel-less dtype. This costs a transient f32 buffer, which is
        // acceptable for what is a one-shot load-time conversion.
        let storage = self.dequantize(elem_count)?;
        storage.to_dtype(&crate::Layout::contiguous(elem_count), crate::DType::F16)
    }

    /// SwarmLLM patch: gather whole rows out of this quantized buffer, on the
    /// device, returning a new quantized buffer holding just those rows.
    ///
    /// **Why this has to exist here.** An embedding lookup reads at most
    /// `seq_len` rows of a table that can be a gigabyte dequantized, so the
    /// table is better kept quantized and read a row at a time. On CPU that
    /// needs no new API — `QTensor::data()` is a zero-copy borrow, so the rows
    /// can be sliced out and handed back through `QStorage::from_data`. On CUDA
    /// `data()` copies the WHOLE table device-to-host, so the same approach
    /// would move a gigabyte across PCIe on every decoded token. llama.cpp hit
    /// exactly that shape: with the lookup off-device a Qwen3-1.7B decoded at
    /// 6.18 ms/token against 1.72 once `k_get_rows_kq` existed.
    ///
    /// It needs no new kernel. Rows are contiguous and block-aligned, the
    /// buffer is a flat `CudaSlice<u8>` whose padding is only ever TRAILING, so
    /// gathering rows is `index_select` over a `[vocab, row_bytes]` view of the
    /// bytes — and `is_u32_u8` is already instantiated in `candle-kernels`.
    ///
    /// `row_bytes` must be `hidden / block_size * type_size`; the caller owns
    /// that arithmetic because it also owns the block-alignment check that makes
    /// a row addressable at all.
    pub fn gather_rows(
        &self,
        ids: &cudarc::driver::CudaSlice<u32>,
        n_ids: usize,
        vocab: usize,
        row_bytes: usize,
    ) -> Result<Self> {
        if row_bytes == 0 {
            crate::bail!("gather_rows: row_bytes must be non-zero")
        }
        if self.data.len < vocab * row_bytes {
            crate::bail!(
                "gather_rows: buffer holds {} bytes, {vocab} rows of {row_bytes} need {}",
                self.data.len,
                vocab * row_bytes
            )
        }
        let dev = &self.device;
        let out_bytes = n_ids * row_bytes;
        // Same trailing padding every quantized buffer here carries, so the
        // result can be fed to the dequantize kernels unchanged.
        let padded_len = out_bytes + MATRIX_ROW_PADDING * self.dtype.type_size()
            / self.dtype.block_size();
        let out = unsafe { dev.alloc::<u8>(padded_len)? };

        // `index_select` over the bytes: one "element" is one byte, the
        // selected dimension is the row, and `right_size` is the row width.
        //
        // No layout is uploaded. The kernel reads `info` for one thing — is
        // the SOURCE contiguous? — and with zero dims `is_contiguous` answers
        // yes without a read, which is true: the source is a flat run of
        // bytes. The `[n_ids] / [1]` layout this passed before was contiguous
        // too, so every lookup takes the same branch as before. The upload was
        // a host→device copy on every decoded token, and a copy inside a CUDA
        // graph capture is replayed from a freed host address (see
        // `cuda_backend::device::HTOD_COPIES`).
        let func = dev.get_or_load_func("is_u32_u8", &candle_kernels::INDEXING)?;
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(out_bytes as u32);
        let src = self.data.inner.slice(..self.data.len);
        let mut builder = func.builder();
        barg!(builder, out_bytes);
        barg!(builder, 0usize); // num_dims
        barg!(builder, 0u64); // info: a null `const size_t*`, never read with zero dims
        builder.arg(ids);
        builder.arg(&src);
        builder.arg(&out);
        barg!(builder, 1usize); // left_size — nothing left of the row dimension
        barg!(builder, vocab); // src_dim_size
        barg!(builder, n_ids); // ids_dim_size
        barg!(builder, row_bytes); // right_size — bytes per row
        unsafe { builder.launch(cfg) }.w()?;

        Ok(Self {
            data: PaddedCudaSlice {
                inner: out,
                len: out_bytes,
            },
            dtype: self.dtype,
            device: dev.clone(),
            // Rows of this weight keep its precision rule.
            accumulate_f32: self.accumulate_f32,
        })
    }

    pub fn quantize(&mut self, src: &CudaStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &CudaStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len
    }

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let max_bm = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            1
        } else {
            8
        };
        let rows = match layout.shape().dims() {
            [b, m, _k] => b * m,
            [b, _k] => *b,
            _ => usize::MAX,
        };
        let use_vec_kernel =
            rows <= max_bm && !mmq_takes_few_rows(self.device(), self.dtype, rows, self.data.len);
        if use_vec_kernel {
            self.dequantize_matmul_vec(self_shape, storage, layout)
        } else {
            self.dequantize_matmul(self_shape, storage, layout)
        }
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.data.len];
        self.device
            .memcpy_dtoh(&self.data.inner.slice(..self.data.len), &mut out)?;
        Ok(out)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        use cudarc::driver::DevicePtr;
        Ok(self.data.inner.device_ptr(self.data.inner.stream()).0 as *const u8)
    }
}

impl QCudaStorage {
    fn dequantize_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let rhs = rhs.as_cuda_slice::<f32>()?;
        let rhs = match rhs_l.contiguous_offsets() {
            Some((o1, o2)) => rhs.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "dmmv" }.bt())?,
        };
        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!("unexpected rhs shape in dmmv {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            dequantize_mul_mat_vec(&self.data, &rhs, self.dtype, ncols, nrows, self.device())?
        } else {
            mul_mat_vec_via_q8_1(
                &self.data,
                &rhs,
                self.dtype,
                ncols,
                nrows,
                b_size,
                self.device(),
            )?
        };
        let mut out_shape = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        Ok((out, out_shape.into()))
    }

    fn dequantize_matmul(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            let data_f32 = self.dequantize(n * k)?;
            let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
            storage.matmul(&data_f32, (b, m, n, k), layout, &rhs_l)?
        } else {
            let storage = storage.as_cuda_slice::<f32>()?;
            let storage = match layout.contiguous_offsets() {
                Some((o1, o2)) => storage.slice(o1..o2),
                None => Err(crate::Error::RequiresContiguous {
                    op: "quantized-matmul",
                }
                .bt())?,
            };
            // SwarmLLM patch: llama.cpp's tensor-core MMQ first, where it runs
            // ([`mmq_mma_layout`]); a refused or failed launch turns it off and
            // falls through to the old kernels for this and every later call,
            // a call that found no card memory for its buffers only for itself
            // ([`MmqFailure`]).
            let mma = match mmq_mma_layout(self.device(), self.dtype, b * m) {
                Some(layout) => match mul_mat_via_mmq_mma(
                    &self.data,
                    &storage,
                    self.dtype,
                    layout,
                    /* x_rows */ n,
                    /* x_cols */ k,
                    /* y_cols */ b * m,
                    self.device(),
                ) {
                    Ok(out) => Some(out),
                    Err(MmqFailure::NoRoom(e)) => {
                        if MMQ_MMA_NO_ROOM.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0
                        {
                            eprintln!(
                                "llama.cpp MMQ found no card memory for one product; the older \
                                 kernels ran it and MMQ stays on: {e}"
                            );
                        }
                        None
                    }
                    Err(MmqFailure::Refused(e)) => {
                        if !MMQ_MMA_FAILED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            eprintln!(
                                "llama.cpp MMQ unavailable, using the older kernels from now on: {e}"
                            );
                        }
                        None
                    }
                },
                None => None,
            };
            if let Some(out) = mma {
                out
            } else if cublas_min_rows(self.device()).is_some_and(|min| b * m >= min) {
                mul_mat_via_f16_cublas(
                    &self.data,
                    &storage,
                    self.dtype,
                    /* x_rows */ n,
                    /* x_cols */ k,
                    /* y_cols */ b * m,
                    cublas_accumulates_in_f16() && !self.accumulate_f32,
                    self.device(),
                )?
            } else {
                mul_mat_via_q8_1(
                    &self.data,
                    &storage,
                    self.dtype,
                    /* x_rows */ n,
                    /* x_cols */ k,
                    /* y_rows */ k,
                    /* y_cols */ b * m,
                    self.device(),
                )?
            }
        };
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &CudaDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let dtype = T::DTYPE;
    let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let mut inner = device.alloc_zeros::<u8>(padded_len)?;
    device.memcpy_htod(data, &mut inner.slice_mut(..data.len()))?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: PaddedCudaSlice {
            inner,
            len: data.len(),
        },
        device: device.clone(),
        dtype,
        accumulate_f32: false,
    }))
}

/// SwarmLLM patch: [`load_quantized`] from a weight's bytes, copied on the
/// device's copy stream (`CudaDevice::clone_htod_padded_beside`) and padded as
/// [`load_quantized`] pads.
pub fn load_quantized_beside(
    device: &CudaDevice,
    data: &[u8],
    dtype: GgmlDType,
) -> Result<super::QStorage> {
    let padding = MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let inner = device.clone_htod_padded_beside(data, padding)?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: PaddedCudaSlice {
            inner,
            len: data.len(),
        },
        device: device.clone(),
        dtype,
        accumulate_f32: false,
    }))
}

/// SwarmLLM patch: a quantized stack of `entries` slots of `entry_bytes` each
/// on the card, holding only the slots `parts` names — `(slot, bytes)`, each
/// copied to `slot * entry_bytes`. Every other slot stays zero and must never be
/// read. A processor-held mixture-of-experts layer sends a prompt chunk's
/// chosen experts this way, one copy per expert straight from where the
/// processor keeps it (llama.cpp copies only the used experts too:
/// `ggml_backend_sched_compute_splits`, SwarmLLM FUTURE_WORK #138). The stack
/// is padded as [`load_quantized`] pads one, and the copies run on the
/// device's one stream, ordered with the kernels that read them.
pub fn load_quantized_entries(
    device: &CudaDevice,
    dtype: GgmlDType,
    entries: usize,
    entry_bytes: usize,
    parts: &[(usize, &[u8])],
) -> Result<super::QStorage> {
    let len = entries * entry_bytes;
    let padded_len = len + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let mut inner = device.alloc_zeros::<u8>(padded_len)?;
    for &(slot, bytes) in parts {
        if slot >= entries || bytes.len() != entry_bytes {
            crate::bail!(
                "load_quantized_entries: slot {slot} of {entries} with {} bytes, expected {entry_bytes}",
                bytes.len()
            )
        }
        let at = slot * entry_bytes;
        device.memcpy_htod(bytes, &mut inner.slice_mut(at..at + entry_bytes))?;
    }
    Ok(QStorage::Cuda(QCudaStorage {
        data: PaddedCudaSlice { inner, len },
        device: device.clone(),
        dtype,
        accumulate_f32: false,
    }))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn cuda_quantize_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let el = 256;
        let el_padded = pad(el, MATRIX_ROW_PADDING);
        let y_size_in_bytes =
            el_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut y_q8_1 = dev.alloc_fully_overwritten::<u8>(y_size_in_bytes)?;
        let vs: Vec<f32> = (0..el).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        quantize_q8_1(&y.as_view(), &mut y_q8_1, el, 1, &dev)?;
        Ok(())
    }

    #[test]
    fn cuda_mmv_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_vec_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            /* b_size */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        // for n = 255, n.(n+1).(2n+1) / 6 = 5559680
        // Q8 means 1/256 precision.
        assert_eq!(vs[0], 5561664.5);

        let cuda_storage = dequantize_mul_mat_vec(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0], 5561851.0);
        Ok(())
    }

    #[test]
    fn cuda_mm_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols * 4).map(|v| v as f32 / 4.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * 4, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ 4,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ 4,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;

        /*
           x = torch.tensor([float(v) for v in range(1024)]).reshape(4, 256)
           x @ x.t() / 16
        tensor([[  347480.0000,   869720.0000,  1391960.0000,  1914200.0000],
                [  869720.0000,  2440536.0000,  4011352.0000,  5582166.5000],
                [ 1391960.0000,  4011352.0000,  6630742.0000,  9250132.0000],
                [ 1914200.0000,  5582166.5000,  9250132.0000, 12918099.0000]])
                */
        assert_eq!(vs.len(), 16);
        assert_eq!(vs[0], 347604.0);
        assert_eq!(vs[1], 888153.06);
        assert_eq!(vs[4], 869780.7);
        assert_eq!(vs[5], 2483145.0);
        assert_eq!(vs[11], 9407368.0);
        assert_eq!(vs[14], 9470856.0);
        assert_eq!(vs[15], 13138824.0);
        Ok(())
    }

    // The following test used to fail under compute-sanitizer until #2526.
    #[test]
    fn cuda_mm_q8_1_pad() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (x_rows, ncols, y_cols) = (4, 16, 2048);
        let vs: Vec<f32> = (0..ncols * y_cols).map(|v| v as f32 / 256.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ x_rows,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ y_cols,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let _vs = dev.clone_dtoh(&vs.as_view())?;
        Ok(())
    }
}
