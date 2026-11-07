//! Support for the GGML file format.

use super::{k_quants, GgmlDType, QStorage};
use crate::{Device, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;

// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.h#L37
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Magic {
    Ggjt,
    Ggla,
    Ggmf,
    Ggml,
    Ggsn,
}

impl TryFrom<u32> for Magic {
    type Error = crate::Error;
    fn try_from(value: u32) -> Result<Self> {
        let magic = match value {
            0x67676a74 => Self::Ggjt,
            0x67676c61 => Self::Ggla,
            0x67676d66 => Self::Ggmf,
            0x67676d6c => Self::Ggml,
            0x6767736e => Self::Ggsn,
            _ => crate::bail!("unknown magic {value:08x}"),
        };
        Ok(magic)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgmlUnversioned,
    GgmfV1,
    GgjtV1,
    GgjtV2,
    GgjtV3,
}

impl VersionedMagic {
    fn read<R: std::io::Read>(reader: &mut R) -> Result<Self> {
        let magic = reader.read_u32::<LittleEndian>()?;
        let magic = Magic::try_from(magic)?;
        if magic == Magic::Ggml {
            return Ok(Self::GgmlUnversioned);
        }
        let version = reader.read_u32::<LittleEndian>()?;
        let versioned_magic = match (magic, version) {
            (Magic::Ggmf, 1) => Self::GgmfV1,
            (Magic::Ggjt, 1) => Self::GgjtV1,
            (Magic::Ggjt, 2) => Self::GgjtV2,
            (Magic::Ggjt, 3) => Self::GgjtV3,
            _ => crate::bail!("ggml: unsupported magic/version {magic:?}/{version}"),
        };
        Ok(versioned_magic)
    }

    fn align32(&self) -> bool {
        match self {
            Self::GgmlUnversioned | Self::GgmfV1 => false,
            Self::GgjtV1 | Self::GgjtV2 | Self::GgjtV3 => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HParams {
    pub n_vocab: u32,
    pub n_embd: u32,
    pub n_mult: u32,
    pub n_head: u32,
    pub n_layer: u32,
    pub n_rot: u32,
    pub ftype: u32,
}

impl HParams {
    fn read<R: std::io::Read>(reader: &mut R) -> Result<Self> {
        let n_vocab = reader.read_u32::<LittleEndian>()?;
        let n_embd = reader.read_u32::<LittleEndian>()?;
        let n_mult = reader.read_u32::<LittleEndian>()?;
        let n_head = reader.read_u32::<LittleEndian>()?;
        let n_layer = reader.read_u32::<LittleEndian>()?;
        let n_rot = reader.read_u32::<LittleEndian>()?;
        let ftype = reader.read_u32::<LittleEndian>()?;
        Ok(Self {
            n_vocab,
            n_embd,
            n_mult,
            n_head,
            n_layer,
            n_rot,
            ftype,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Vocab {
    pub token_score_pairs: Vec<(Vec<u8>, f32)>,
}

impl Vocab {
    fn read<R: std::io::Read>(reader: &mut R, n_vocab: usize) -> Result<Self> {
        // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.cpp#L556
        let mut token_score_pairs = Vec::with_capacity(n_vocab);
        for _index in 0..n_vocab {
            let len = reader.read_u32::<LittleEndian>()? as usize;
            let mut word = vec![0u8; len];
            reader.read_exact(&mut word)?;
            let score = reader.read_f32::<LittleEndian>()?;
            token_score_pairs.push((word, score))
        }
        Ok(Self { token_score_pairs })
    }
}

fn from_raw_data<T: super::GgmlType + Send + Sync + 'static>(
    raw_data: &[u8],
    size_in_bytes: usize,
    dims: Vec<usize>,
    device: &Device,
) -> Result<super::QTensor> {
    let raw_data_ptr = raw_data.as_ptr();
    let n_blocks = size_in_bytes / std::mem::size_of::<T>();
    let data = unsafe { std::slice::from_raw_parts(raw_data_ptr as *const T, n_blocks) };
    let data: QStorage = match device {
        Device::Cpu => QStorage::Cpu(Box::new(data.to_vec())),
        Device::Metal(metal) => super::metal::load_quantized(metal, data)?,
        Device::Cuda(cuda) => super::cuda::load_quantized(cuda, data)?,
    };
    super::QTensor::new(data, dims)
}

/// Creates a [Tensor] from a raw GGML tensor.
pub fn qtensor_from_ggml(
    ggml_dtype: GgmlDType,
    raw_data: &[u8],
    dims: Vec<usize>,
    device: &Device,
) -> Result<super::QTensor> {
    let tensor_elems = dims.iter().product::<usize>();
    let block_size = ggml_dtype.block_size();
    if tensor_elems % block_size != 0 {
        crate::bail!(
            "the number of elements {tensor_elems} is not divisible by the block size {block_size}"
        )
    }
    let size_in_bytes = tensor_elems / block_size * ggml_dtype.type_size();

    match ggml_dtype {
        GgmlDType::F32 => from_raw_data::<f32>(raw_data, size_in_bytes, dims, device),
        GgmlDType::F16 => from_raw_data::<half::f16>(raw_data, size_in_bytes, dims, device),
        GgmlDType::BF16 => from_raw_data::<half::bf16>(raw_data, size_in_bytes, dims, device),
        GgmlDType::Q4_0 => {
            from_raw_data::<k_quants::BlockQ4_0>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q4_1 => {
            from_raw_data::<k_quants::BlockQ4_1>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q5_0 => {
            from_raw_data::<k_quants::BlockQ5_0>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q5_1 => {
            from_raw_data::<k_quants::BlockQ5_1>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q8_0 => {
            from_raw_data::<k_quants::BlockQ8_0>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q2K => {
            from_raw_data::<k_quants::BlockQ2K>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q3K => {
            from_raw_data::<k_quants::BlockQ3K>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q4K => {
            from_raw_data::<k_quants::BlockQ4K>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q5K => {
            from_raw_data::<k_quants::BlockQ5K>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Q6K => {
            from_raw_data::<k_quants::BlockQ6K>(raw_data, size_in_bytes, dims, device)
        }
        // SwarmLLM patch: gpt-oss's experts.
        GgmlDType::Mxfp4 => {
            from_raw_data::<k_quants::BlockMxfp4>(raw_data, size_in_bytes, dims, device)
        }
        // SwarmLLM patch: IQ4_NL / IQ4_XS (#404).
        GgmlDType::Iq4Nl => {
            from_raw_data::<k_quants::BlockIq4Nl>(raw_data, size_in_bytes, dims, device)
        }
        GgmlDType::Iq4Xs => {
            from_raw_data::<k_quants::BlockIq4Xs>(raw_data, size_in_bytes, dims, device)
        }
        _ => crate::bail!("quantized type {ggml_dtype:?} is not supported yet"),
    }
}

/// SwarmLLM patch (#424 step 3): a GGML tensor read off `reader` straight into
/// the storage it will live in — [`qtensor_from_ggml`] without the buffer of
/// the tensor's whole size it is handed.
///
/// That buffer was fresh memory for every tensor, and a large one is fresh
/// pages from the OS: a page fault per 4 KB while the file is read into it
/// (1.23 GB/s here against 3.5 reading into memory already touched, the same
/// cached files — 287 K faults a GB). On the processor the blocks were then
/// copied into a SECOND fresh allocation (`to_vec`), faulting again. Now the
/// processor reads into the blocks themselves, and a card through one
/// staging buffer each loading thread reuses (`cuda::load_quantized_read`).
///
/// Prior art: llama.cpp's `llama_model_loader::load_all_data` reads a tensor
/// bound for a device through 4 REUSED pinned staging buffers (1 MiB, or
/// 64 MiB with direct I/O) and uploads each asynchronously behind an event
/// (`src/llama-model-loader.cpp`, read 2026-11-07); one in host memory is read
/// in place. The reuse is taken here; the overlap of a read with the previous
/// chunk's copy is not (measured first, #424 step 3).
pub fn qtensor_read<R: std::io::Read>(
    ggml_dtype: GgmlDType,
    reader: &mut R,
    dims: Vec<usize>,
    device: &Device,
) -> Result<super::QTensor> {
    let tensor_elems = dims.iter().product::<usize>();
    let block_size = ggml_dtype.block_size();
    if tensor_elems % block_size != 0 {
        crate::bail!(
            "the number of elements {tensor_elems} is not divisible by the block size {block_size}"
        )
    }
    let size = tensor_elems / block_size * ggml_dtype.type_size();
    use k_quants::*;
    match ggml_dtype {
        GgmlDType::F32 => read_blocks::<f32, R>(reader, size, dims, device),
        GgmlDType::F16 => read_blocks::<half::f16, R>(reader, size, dims, device),
        GgmlDType::BF16 => read_blocks::<half::bf16, R>(reader, size, dims, device),
        GgmlDType::Q4_0 => read_blocks::<BlockQ4_0, R>(reader, size, dims, device),
        GgmlDType::Q4_1 => read_blocks::<BlockQ4_1, R>(reader, size, dims, device),
        GgmlDType::Q5_0 => read_blocks::<BlockQ5_0, R>(reader, size, dims, device),
        GgmlDType::Q5_1 => read_blocks::<BlockQ5_1, R>(reader, size, dims, device),
        GgmlDType::Q8_0 => read_blocks::<BlockQ8_0, R>(reader, size, dims, device),
        GgmlDType::Q2K => read_blocks::<BlockQ2K, R>(reader, size, dims, device),
        GgmlDType::Q3K => read_blocks::<BlockQ3K, R>(reader, size, dims, device),
        GgmlDType::Q4K => read_blocks::<BlockQ4K, R>(reader, size, dims, device),
        GgmlDType::Q5K => read_blocks::<BlockQ5K, R>(reader, size, dims, device),
        GgmlDType::Q6K => read_blocks::<BlockQ6K, R>(reader, size, dims, device),
        GgmlDType::Mxfp4 => read_blocks::<BlockMxfp4, R>(reader, size, dims, device),
        GgmlDType::Iq4Nl => read_blocks::<BlockIq4Nl, R>(reader, size, dims, device),
        GgmlDType::Iq4Xs => read_blocks::<BlockIq4Xs, R>(reader, size, dims, device),
        _ => crate::bail!("quantized type {ggml_dtype:?} is not supported yet"),
    }
}

/// [`qtensor_read`] for one block type: the bytes [`from_raw_data`] would be
/// handed, read where they will stay.
fn read_blocks<T: super::GgmlType + Send + Sync + 'static, R: std::io::Read>(
    reader: &mut R,
    size_in_bytes: usize,
    dims: Vec<usize>,
    device: &Device,
) -> Result<super::QTensor> {
    let n_blocks = size_in_bytes / std::mem::size_of::<T>();
    let bytes = n_blocks * std::mem::size_of::<T>();
    let data: QStorage = match device {
        Device::Cpu => {
            let mut blocks = zeroed_blocks::<T>(n_blocks)?;
            // SAFETY: a GGML block is plain numbers (`GgmlType::zeros` builds one
            // from zeroed memory), so its bytes may be written as bytes.
            let raw = unsafe { std::slice::from_raw_parts_mut(blocks.as_mut_ptr() as *mut u8, bytes) };
            reader.read_exact(raw)?;
            QStorage::Cpu(Box::new(blocks))
        }
        Device::Cuda(cuda) => super::cuda::load_quantized_read(cuda, reader, bytes, T::DTYPE)?,
        Device::Metal(metal) => {
            let mut raw_data = vec![0u8; bytes];
            reader.read_exact(&mut raw_data)?;
            // SAFETY: as `from_raw_data` — the bytes of `n_blocks` plain blocks.
            let blocks =
                unsafe { std::slice::from_raw_parts(raw_data.as_ptr() as *const T, n_blocks) };
            super::metal::load_quantized(metal, blocks)?
        }
    };
    super::QTensor::new(data, dims)
}

/// `n` zeroed blocks in memory the allocator hands over already zero — a large
/// one fresh from the OS, its pages faulted once, by the read that fills them.
fn zeroed_blocks<T: super::GgmlType>(n: usize) -> Result<Vec<T>> {
    if n == 0 || std::mem::size_of::<T>() == 0 {
        return Ok(Vec::new());
    }
    let layout = std::alloc::Layout::array::<T>(n)
        .map_err(|e| crate::Error::Msg(format!("{n} blocks: {e}")))?;
    // SAFETY: a layout of non-zero size; a null answer is handled below.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) } as *mut T;
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    // SAFETY: allocated by the global allocator with `T`'s layout for `n`
    // elements, every one initialised: all-zero bytes are a block.
    Ok(unsafe { Vec::from_raw_parts(ptr, n, n) })
}

fn read_one_tensor<R: std::io::Seek + std::io::Read>(
    reader: &mut R,
    magic: VersionedMagic,
    device: &Device,
) -> Result<(String, super::QTensor)> {
    let n_dims = reader.read_u32::<LittleEndian>()?;
    let name_len = reader.read_u32::<LittleEndian>()?;
    let ggml_dtype = reader.read_u32::<LittleEndian>()?;
    let ggml_dtype = GgmlDType::from_u32(ggml_dtype)?;
    let mut dims = vec![0u32; n_dims as usize];
    reader.read_u32_into::<LittleEndian>(&mut dims)?;
    // The dimensions are stored in reverse order, see for example:
    // https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/convert.py#L969
    dims.reverse();
    let mut name = vec![0u8; name_len as usize];
    reader.read_exact(&mut name)?;
    let name = String::from_utf8_lossy(&name).into_owned();

    if magic.align32() {
        let pos = reader.stream_position()?;
        reader.seek(std::io::SeekFrom::Current(((32 - pos % 32) % 32) as i64))?;
    }
    let dims = dims.iter().map(|&u| u as usize).collect::<Vec<_>>();
    let tensor_elems = dims.iter().product::<usize>();
    let size_in_bytes = tensor_elems * ggml_dtype.type_size() / ggml_dtype.block_size();
    // TODO: Mmap version to avoid copying the data around?
    let mut raw_data = vec![0u8; size_in_bytes];
    reader.read_exact(&mut raw_data)?;
    match qtensor_from_ggml(ggml_dtype, &raw_data, dims, device) {
        Ok(tensor) => Ok((name, tensor)),
        Err(e) => crate::bail!("Error creating tensor {name}: {e}"),
    }
}

pub struct Content {
    pub magic: VersionedMagic,
    pub hparams: HParams,
    pub vocab: Vocab,
    pub tensors: HashMap<String, super::QTensor>,
    pub device: Device,
}

impl Content {
    pub fn read<R: std::io::Seek + std::io::Read>(
        reader: &mut R,
        device: &Device,
    ) -> Result<Content> {
        // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.cpp#L505
        let last_position = reader.seek(std::io::SeekFrom::End(0))?;
        reader.seek(std::io::SeekFrom::Start(0))?;
        let magic = VersionedMagic::read(reader)?;
        let hparams = HParams::read(reader)?;
        let vocab = Vocab::read(reader, hparams.n_vocab as usize)?;
        let mut tensors = HashMap::new();

        while reader.stream_position()? != last_position {
            let (name, tensor) = read_one_tensor(reader, magic, device)?;
            tensors.insert(name, tensor);
        }
        let device = device.clone();
        Ok(Self {
            magic,
            hparams,
            vocab,
            tensors,
            device,
        })
    }

    pub fn remove(&mut self, name: &str) -> Result<super::QTensor> {
        match self.tensors.remove(name) {
            None => crate::bail!("cannot find tensor with name '{name}'"),
            Some(tensor) => Ok(tensor),
        }
    }
}
