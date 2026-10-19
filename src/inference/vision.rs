//! Vision Language Model (VLM) support for multimodal inference.
//!
//! This module provides:
//! - Image preprocessing: resize, normalize, convert to candle tensors
//! - Vision encoder (ViT-style): patch embedding, transformer blocks, projection
//! - Multimodal projection: map vision features into the LLM embedding space
//! - Projectors, chosen by the file's `clip.projector_type`: LLaVA's MLP
//!   (`mlp`, CLIP ViT) and Gemma 3's (`gemma3`, SigLIP + 4×4 pool + RMS norm +
//!   linear); any other kind is refused by name
//!
//! The vision encoder can run as a separate shard — the node encoding an
//! image sends its rows beside the prompt, and the first segment puts them
//! where the prompt marks the image ([`MEDIA_MARKER`], [`splice_image_rows`]).

use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{Linear, Module};

use crate::error::SwarmError;
use crate::types::{ImageData, VisionConfig};

// ── Image Preprocessing ──

// Default CLIP normalization constants (ImageNet). These are the canonical
// published values from OpenAI's CLIP repo; the extra precision (>7 sig figs)
// exceeds f32's range and clippy flags it, but truncating would silently drift
// from upstream — the allow is the documented exception.
#[allow(clippy::excessive_precision)]
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
#[allow(clippy::excessive_precision)]
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// How an image is made the encoder's square, read off the projector the way
/// llama.cpp's `mtmd_image_preprocessor_fixed_size` does: scaled to fit inside
/// the square, keeping its shape, and centred on `pad` — `PAD_CEIL`, so the
/// scaled sides are rounded UP and the offsets are integer halves of what is
/// left. Neither family stretches a picture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Preprocess {
    /// The side of the square the encoder reads.
    pub size: u32,
    /// Per-channel mean and spread (`clip.vision.image_mean` / `image_std`).
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// The colour around a picture that is not square: LLaVA's `expand2square`
    /// fills with the mean (llama.cpp's `{122, 116, 104}` for CLIP's), Gemma 3
    /// with black.
    pub pad: [u8; 3],
    pub resample: Resample,
}

/// The filter a picture is resized with — both antialiased when shrinking, as
/// Pillow's are (llama.cpp's `resize_pillow`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resample {
    /// Gemma 3 (`RESIZE_ALGO_BILINEAR`).
    Bilinear,
    /// LLaVA (llama.cpp's default, `RESIZE_ALGO_BICUBIC`): Pillow's bicubic is
    /// Keys' cubic at a = -0.5, which is Catmull-Rom.
    Bicubic,
}

impl Resample {
    fn filter(self) -> image::imageops::FilterType {
        match self {
            Self::Bilinear => image::imageops::FilterType::Triangle,
            Self::Bicubic => image::imageops::FilterType::CatmullRom,
        }
    }
}

/// LLaVA's padding colour: the mean as bytes, truncated — LLaVA's
/// `expand2square(image, tuple(int(x * 255) for x in image_mean))`.
fn mean_as_bytes(mean: [f32; 3]) -> [u8; 3] {
    mean.map(|m| (m * 255.0).clamp(0.0, 255.0) as u8)
}

/// The scaled size and offset of an image letterboxed into a `size` square —
/// llama.cpp's `img_tool::resize` with `PAD_CEIL`.
fn letterbox_geometry(width: u32, height: u32, size: u32) -> (u32, u32, u32, u32) {
    let scale = (size as f32 / width as f32).min(size as f32 / height as f32);
    let w = ((width as f32 * scale).ceil() as u32).clamp(1, size);
    let h = ((height as f32 * scale).ceil() as u32).clamp(1, size);
    (w, h, (size - w) / 2, (size - h) / 2)
}

/// Preprocess an image for the vision encoder.
///
/// 1. Letterbox into `pre.size x pre.size` on `pre.pad`, resized with
///    `pre.resample`
/// 2. Convert to f32 [0, 1]
/// 3. Normalize with the projector's mean/std
/// 4. Return tensor of shape (3, H, W)
fn preprocess_image(
    image: &ImageData,
    pre: &Preprocess,
    device: &Device,
) -> Result<Tensor, SwarmError> {
    // Reject images that would exhaust memory when decoded to float tensors
    const MAX_IMAGE_PIXELS: u64 = 4096 * 4096; // 16M pixels max
    let pixel_count = image.width as u64 * image.height as u64;
    if pixel_count > MAX_IMAGE_PIXELS || pixel_count == 0 {
        return Err(SwarmError::Inference(format!(
            "Image dimensions {}x{} exceed maximum {} pixels",
            image.width, image.height, MAX_IMAGE_PIXELS
        )));
    }

    // Use the image crate to resize
    let img_buf = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
        image.width,
        image.height,
        image.rgb_bytes.clone(),
    )
    .ok_or_else(|| SwarmError::Inference("Invalid image dimensions".into()))?;

    let size = pre.size;
    let resized = {
        let (w, h, x0, y0) = letterbox_geometry(image.width, image.height, size);
        let scaled = image::imageops::resize(&img_buf, w, h, pre.resample.filter());
        let mut canvas = image::ImageBuffer::from_pixel(size, size, image::Rgb(pre.pad));
        image::imageops::replace(&mut canvas, &scaled, x0 as i64, y0 as i64);
        canvas
    };

    let (w, h) = (resized.width() as usize, resized.height() as usize);

    // Convert to CHW float tensor, normalize
    let mut data = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let pixel = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                let normalized = (val - pre.mean[c]) / pre.std[c];
                data[c * h * w + y * w + x] = normalized;
            }
        }
    }

    Tensor::from_vec(data, &[3, h, w], device)
        .map_err(|e| SwarmError::Inference(format!("Image tensor creation failed: {e}")))
}

/// Preprocess a batch of images into a single (B, 3, H, W) tensor.
pub fn preprocess_images(
    images: &[ImageData],
    pre: &Preprocess,
    device: &Device,
) -> Result<Tensor, SwarmError> {
    if images.is_empty() {
        return Err(SwarmError::Inference("No images to preprocess".into()));
    }

    let tensors: Vec<Tensor> = images
        .iter()
        .map(|img| preprocess_image(img, pre, device))
        .collect::<Result<Vec<_>, _>>()?;

    // Stack into batch: each is (3, H, W) → (B, 3, H, W)
    Tensor::stack(&tensors, 0)
        .map_err(|e| SwarmError::Inference(format!("Image batch stacking failed: {e}")))
}

// ── Vision Encoder (ViT) ──

/// Patch embedding: converts image patches into a sequence of embeddings.
///
/// Input: (B, 3, H, W) image tensor
/// Output: (B, num_patches + 1, hidden_dim) with a `[CLS]` token (CLIP), or
/// (B, num_patches, hidden_dim) without one (SigLIP, Gemma 3's encoder)
pub struct PatchEmbedding {
    /// Conv2D projection: (hidden_dim, 3, patch_size, patch_size)
    proj_weight: Tensor,
    proj_bias: Tensor,
    /// `[CLS]` token embedding: (1, 1, hidden_dim) — `v.class_embd`, which a
    /// SigLIP encoder does not have.
    cls_token: Option<Tensor>,
    /// Positional embeddings: (1, num_patches (+ 1), hidden_dim)
    position_embedding: Tensor,
    patch_size: usize,
}

impl PatchEmbedding {
    pub fn new(
        proj_weight: Tensor,
        proj_bias: Tensor,
        cls_token: Option<Tensor>,
        position_embedding: Tensor,
        patch_size: usize,
    ) -> Self {
        Self {
            proj_weight,
            proj_bias,
            cls_token,
            position_embedding,
            patch_size,
        }
    }

    /// Forward: (B, 3, H, W) → (B, num_patches (+ 1), hidden_dim)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, SwarmError> {
        let (b, _c, h, w) = x
            .dims4()
            .map_err(|e| SwarmError::Inference(e.to_string()))?;
        let ps = self.patch_size;
        let num_patches_h = h / ps;
        let num_patches_w = w / ps;
        let num_patches = num_patches_h * num_patches_w;
        let hidden_dim = self
            .proj_weight
            .dim(0)
            .map_err(|e| SwarmError::Inference(e.to_string()))?;

        // Extract patches and project: manual unfold + matmul
        // Reshape image into patches: (B, 3, nH, ps, nW, ps)
        // → transpose → (B, nH*nW, 3*ps*ps) → matmul with weight
        let x_reshaped = x
            .reshape(&[b, 3, num_patches_h, ps, num_patches_w, ps])
            .map_err(|e| SwarmError::Inference(format!("Patch reshape: {e}")))?;

        // Permute to (B, nH, nW, 3, ps, ps) then flatten patches
        let x_perm = x_reshaped
            .permute([0, 2, 4, 1, 3, 5])
            .map_err(|e| SwarmError::Inference(format!("Patch permute: {e}")))?;

        let patch_dim = 3 * ps * ps;
        let patches = x_perm
            .reshape(&[b, num_patches, patch_dim])
            .map_err(|e| SwarmError::Inference(format!("Patch flatten: {e}")))?;

        // Project: (B, num_patches, patch_dim) @ (patch_dim, hidden_dim) → (B, num_patches, hidden_dim)
        let weight_t = self
            .proj_weight
            .reshape(&[hidden_dim, patch_dim])
            .map_err(|e| SwarmError::Inference(format!("Weight reshape: {e}")))?
            .t()
            .map_err(|e| SwarmError::Inference(format!("Weight transpose: {e}")))?;

        let projected = patches
            .broadcast_matmul(&weight_t)
            .map_err(|e| SwarmError::Inference(format!("Patch projection: {e}")))?;

        let projected = projected
            .broadcast_add(&self.proj_bias)
            .map_err(|e| SwarmError::Inference(format!("Patch bias: {e}")))?;

        // Prepend CLS token, where the encoder has one
        let sequence = match &self.cls_token {
            Some(cls) => {
                let cls_expanded = cls
                    .broadcast_as(&[b, 1, hidden_dim])
                    .map_err(|e| SwarmError::Inference(format!("CLS broadcast: {e}")))?;
                Tensor::cat(&[&cls_expanded, &projected], 1)
                    .map_err(|e| SwarmError::Inference(format!("CLS concat: {e}")))?
            }
            None => projected,
        };
        let positions = num_patches + usize::from(self.cls_token.is_some());

        // Add positional embeddings
        let pos_embed = if self.position_embedding.dim(1).unwrap_or(0) == positions {
            self.position_embedding.clone()
        } else {
            // Interpolate position embeddings if needed (image size mismatch)
            self.position_embedding
                .i((.., ..positions, ..))
                .map_err(|e| SwarmError::Inference(format!("Pos embed slice: {e}")))?
        };

        sequence
            .broadcast_add(&pos_embed)
            .map_err(|e| SwarmError::Inference(format!("Pos embed add: {e}")))
    }
}

/// The activation of an encoder's feed-forward, read off the file the way
/// llama.cpp does (`clip.cpp`, `KEY_USE_GELU` / `KEY_USE_SILU`) by
/// [`EncoderActivation::from_keys`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderActivation {
    /// The exact, erf form — no file names it; the default of a block built
    /// in code.
    GeluErf,
    /// `clip.use_gelu = true`: `ggml_gelu`, the TANH approximation (Gemma 3's
    /// SigLIP, HF's `gelu_pytorch_tanh`).
    GeluTanh,
    /// Neither key set: `x · sigmoid(1.702 x)` — CLIP's `quick_gelu`, LLaVA's
    /// encoder (#310: it ran the erf form here before).
    GeluQuick,
    /// `clip.use_silu = true`.
    Silu,
}

impl EncoderActivation {
    /// llama.cpp's reading: `use_gelu` → tanh GELU, `use_silu` → SiLU, neither
    /// → quick GELU, both → refused.
    pub fn from_keys(use_gelu: bool, use_silu: bool) -> Result<Self, SwarmError> {
        match (use_gelu, use_silu) {
            (true, true) => Err(SwarmError::Validation(
                "This model's image reader sets both clip.use_gelu and clip.use_silu; \
                 it cannot use both."
                    .into(),
            )),
            (true, false) => Ok(Self::GeluTanh),
            (false, true) => Ok(Self::Silu),
            (false, false) => Ok(Self::GeluQuick),
        }
    }

    fn apply(self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::GeluErf => x.gelu_erf(),
            Self::GeluTanh => x.gelu(),
            Self::GeluQuick => x * candle_nn::ops::sigmoid(&(x * 1.702)?)?,
            Self::Silu => x.silu(),
        }
    }
}

/// A single ViT transformer block: LayerNorm → Attention → LayerNorm → MLP.
pub struct VisionTransformerBlock {
    ln1: candle_nn::LayerNorm,
    attn_qkv: Linear,
    attn_proj: Linear,
    ln2: candle_nn::LayerNorm,
    mlp_fc1: Linear,
    mlp_fc2: Linear,
    num_heads: usize,
    head_dim: usize,
    activation: EncoderActivation,
}

impl VisionTransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ln1: candle_nn::LayerNorm,
        attn_qkv: Linear,
        attn_proj: Linear,
        ln2: candle_nn::LayerNorm,
        mlp_fc1: Linear,
        mlp_fc2: Linear,
        num_heads: usize,
        hidden_dim: usize,
    ) -> Self {
        Self {
            ln1,
            attn_qkv,
            attn_proj,
            ln2,
            mlp_fc1,
            mlp_fc2,
            num_heads,
            head_dim: hidden_dim / num_heads,
            activation: EncoderActivation::GeluErf,
        }
    }

    /// The same block with the feed-forward's activation set (see
    /// [`EncoderActivation`]).
    pub fn with_activation(mut self, activation: EncoderActivation) -> Self {
        self.activation = activation;
        self
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, SwarmError> {
        let map_err = |msg: &'static str| {
            move |e: candle_core::Error| SwarmError::Inference(format!("{msg}: {e}"))
        };

        // Self-attention with pre-norm
        let residual = x;
        let h = self.ln1.forward(x).map_err(map_err("vit_ln1"))?;

        let (b, seq_len, _hidden) = h.dims3().map_err(map_err("vit_dims"))?;

        // QKV projection: (B, S, H) → (B, S, 3*H)
        let qkv = self.attn_qkv.forward(&h).map_err(map_err("vit_qkv"))?;

        // Split into Q, K, V: each (B, S, H) → (B, num_heads, S, head_dim)
        let hidden_dim = self.num_heads * self.head_dim;
        let q = qkv
            .narrow(2, 0, hidden_dim)
            .map_err(map_err("vit_q"))?
            .reshape(&[b, seq_len, self.num_heads, self.head_dim])
            .map_err(map_err("vit_q_reshape"))?
            .transpose(1, 2)
            .map_err(map_err("vit_q_transpose"))?
            .contiguous()
            .map_err(map_err("vit_q_contiguous"))?;
        let k = qkv
            .narrow(2, hidden_dim, hidden_dim)
            .map_err(map_err("vit_k"))?
            .reshape(&[b, seq_len, self.num_heads, self.head_dim])
            .map_err(map_err("vit_k_reshape"))?
            .transpose(1, 2)
            .map_err(map_err("vit_k_transpose"))?
            .contiguous()
            .map_err(map_err("vit_k_contiguous"))?;
        let v = qkv
            .narrow(2, 2 * hidden_dim, hidden_dim)
            .map_err(map_err("vit_v"))?
            .reshape(&[b, seq_len, self.num_heads, self.head_dim])
            .map_err(map_err("vit_v_reshape"))?
            .transpose(1, 2)
            .map_err(map_err("vit_v_transpose"))?
            .contiguous()
            .map_err(map_err("vit_v_contiguous"))?;

        // Scaled dot-product attention (no causal mask — ViT uses full
        // attention). On the processor it is the prompt pass's key-tiled
        // kernel, which never writes the score matrix: Gemma 3's 4,096 patches
        // × 16 heads would be a 1 GB matrix per block, made three times over
        // (scores, scaled, softmax). The matmul form stays as the fallback and
        // for any other device.
        let tiled = crate::inference::prefill_attn::gqa_prefill_attention_cpu(
            &q,
            &k,
            &v,
            None,
            1.0 / (self.head_dim as f32).sqrt(),
            None,
        )
        .map_err(map_err("vit_attn_tiled"))?;
        let attn_output = match tiled {
            Some(out) => out,
            None => {
                let scale = (self.head_dim as f64).sqrt();
                let attn_weights = q
                    .matmul(&k.t().map_err(map_err("vit_k_t"))?)
                    .map_err(map_err("vit_attn_matmul"))?;
                let attn_weights = (attn_weights / scale).map_err(map_err("vit_attn_scale"))?;
                let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)
                    .map_err(map_err("vit_softmax"))?;
                attn_weights.matmul(&v).map_err(map_err("vit_attn_v"))?
            }
        };

        // Reshape back: (B, num_heads, S, head_dim) → (B, S, hidden_dim)
        let attn_output = attn_output
            .transpose(1, 2)
            .map_err(map_err("vit_attn_transpose"))?
            .contiguous()
            .map_err(map_err("vit_attn_contiguous"))?
            .reshape(&[b, seq_len, hidden_dim])
            .map_err(map_err("vit_attn_reshape"))?;

        // Output projection
        let attn_output = self
            .attn_proj
            .forward(&attn_output)
            .map_err(map_err("vit_attn_proj"))?;

        // Residual connection
        let h = (attn_output + residual).map_err(map_err("vit_residual1"))?;

        // MLP with pre-norm
        let residual = &h;
        let ff_in = self.ln2.forward(&h).map_err(map_err("vit_ln2"))?;
        let ff = self.mlp_fc1.forward(&ff_in).map_err(map_err("vit_fc1"))?;
        let ff = self.activation.apply(&ff).map_err(map_err("vit_act"))?;
        let ff = self.mlp_fc2.forward(&ff).map_err(map_err("vit_fc2"))?;

        (ff + residual).map_err(map_err("vit_residual2"))
    }
}

/// Complete Vision Transformer (ViT) encoder.
///
/// Converts images to a sequence of vision embeddings that can be projected
/// into the LLM's embedding space for multimodal inference.
pub struct VisionEncoder {
    patch_embed: PatchEmbedding,
    /// The blocks that run — every one the file holds, or as many as its
    /// `feature_layer` names (see [`encoder_depth`]).
    blocks: Vec<VisionTransformerBlock>,
    /// `v.post_ln`, where the file has one. LLaVA's has none — it reads a
    /// hidden state from INSIDE the encoder — and none means no norm: it was a
    /// weight-1, bias-0 LayerNorm here, which still normalises (#310).
    final_ln: Option<candle_nn::LayerNorm>,
    /// Optional pre-layernorm applied after patch embedding, before transformer blocks.
    pub pre_ln: Option<candle_nn::LayerNorm>,
    config: VisionConfig,
}

impl VisionEncoder {
    pub fn new(
        patch_embed: PatchEmbedding,
        blocks: Vec<VisionTransformerBlock>,
        final_ln: Option<candle_nn::LayerNorm>,
        config: VisionConfig,
    ) -> Self {
        Self {
            patch_embed,
            blocks,
            final_ln,
            pre_ln: None,
            config,
        }
    }

    /// Encode images into vision feature embeddings: one row per PATCH.
    ///
    /// Input: (B, 3, H, W) preprocessed image tensor
    /// Output: (B, num_patches, vision_hidden_size). A `[CLS]` row takes part
    /// in attention and is then dropped — LLaVA's `select_feature = 'patch'`,
    /// llama.cpp's `get_rows` from 1 — so a LLaVA picture is 576 rows, not 577
    /// (#310).
    pub fn forward(&self, images: &Tensor) -> Result<Tensor, SwarmError> {
        let mut x = self.patch_embed.forward(images)?;

        // Apply pre-layernorm if present (CLIP models have this)
        if let Some(ref pre_ln) = self.pre_ln {
            x = pre_ln
                .forward(&x)
                .map_err(|e| SwarmError::Inference(format!("Vision pre_ln: {e}")))?;
        }

        for (i, block) in self.blocks.iter().enumerate() {
            x = block
                .forward(&x)
                .map_err(|e| SwarmError::Inference(format!("Vision encoder block {i}: {e}")))?;
        }

        if let Some(final_ln) = &self.final_ln {
            x = final_ln
                .forward(&x)
                .map_err(|e| SwarmError::Inference(format!("Vision encoder final_ln: {e}")))?;
        }

        if self.patch_embed.cls_token.is_none() {
            return Ok(x);
        }
        let rows = x
            .dim(1)
            .map_err(|e| SwarmError::Inference(format!("Vision encoder rows: {e}")))?;
        x.narrow(1, 1, rows - 1)
            .map_err(|e| SwarmError::Inference(format!("Vision encoder CLS drop: {e}")))
    }

    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// Rows [`Self::forward`] gives per image: one per patch.
    pub fn num_vision_tokens(&self) -> usize {
        self.patches_per_side().pow(2)
    }

    /// The transformer blocks this encoder runs.
    pub fn depth(&self) -> usize {
        self.blocks.len()
    }

    /// Patches along one side of the square.
    pub fn patches_per_side(&self) -> usize {
        (self.config.image_size / self.config.patch_size) as usize
    }
}

// ── Multimodal Projection ──

/// LLaVA's projector (`clip.projector_type` `mlp`): a 2-layer MLP
/// (vision_hidden → projection_dim → llm_hidden_dim). Gemma 3's is
/// [`Gemma3Projection`]; every other projector type is refused at load.
pub struct MultimodalProjection {
    proj1: Linear,
    proj2: Linear,
    llm_hidden_dim: usize,
}

impl MultimodalProjection {
    pub fn new(proj1: Linear, proj2: Linear, llm_hidden_dim: usize) -> Self {
        Self {
            proj1,
            proj2,
            llm_hidden_dim,
        }
    }

    /// Project vision features into LLM embedding space.
    ///
    /// Input: (B, num_vision_tokens, vision_hidden_size)
    /// Output: (B, num_vision_tokens, llm_hidden_dim)
    pub fn forward(&self, vision_features: &Tensor) -> Result<Tensor, SwarmError> {
        let h = self
            .proj1
            .forward(vision_features)
            .map_err(|e| SwarmError::Inference(format!("mm_proj1: {e}")))?;
        let h = h
            .gelu_erf()
            .map_err(|e| SwarmError::Inference(format!("mm_gelu: {e}")))?;
        self.proj2
            .forward(&h)
            .map_err(|e| SwarmError::Inference(format!("mm_proj2: {e}")))
    }

    /// LLM hidden dimension this projection targets.
    pub fn llm_hidden_dim(&self) -> usize {
        self.llm_hidden_dim
    }
}

/// Gemma 3's projection (`clip.projector_type = gemma3`, llama.cpp's
/// `PROJECTOR_TYPE_GEMMA3` in `models/siglip.cpp`): the encoder's 64×64 patch
/// grid averaged over 4×4 squares to 16×16 = 256 tokens, RMS-normed with
/// `mm.soft_emb_norm` (stored as the scale itself, 1 + w), then multiplied by
/// `mm.input_projection` into the language model's width. No bias, no scale.
pub struct Gemma3Projection {
    /// `mm.soft_emb_norm.weight`: (vision_hidden,)
    soft_emb_norm: Tensor,
    /// `mm.input_projection.weight` in candle order: (vision_hidden, llm_hidden),
    /// so the projection is `x @ W` (GGUF `ne = [llm_hidden, vision_hidden]`).
    input_projection: Tensor,
    /// The side of the averaged square (`clip.vision.projector.scale_factor`,
    /// 4 for every Gemma 3).
    pool: usize,
    eps: f64,
}

impl Gemma3Projection {
    pub fn new(soft_emb_norm: Tensor, input_projection: Tensor, pool: usize, eps: f64) -> Self {
        Self {
            soft_emb_norm,
            input_projection,
            pool,
            eps,
        }
    }

    /// (B, side², vision_hidden) → (B, (side / pool)², llm_hidden). Patches are
    /// row-major, so patch `y * side + x` is row y, column x of the grid, and
    /// each output token is the mean of one pool × pool square.
    pub fn forward(&self, features: &Tensor, side: usize) -> Result<Tensor, SwarmError> {
        let map_err = |msg: &'static str| {
            move |e: candle_core::Error| SwarmError::Inference(format!("{msg}: {e}"))
        };
        let (b, n, hidden) = features.dims3().map_err(map_err("gemma3_proj_dims"))?;
        if n != side * side || !side.is_multiple_of(self.pool) {
            return Err(SwarmError::Inference(format!(
                "Gemma 3 projection: {n} patches do not make a {side}-wide grid that pools by {}",
                self.pool
            )));
        }
        let out_side = side / self.pool;
        let pooled = features
            .reshape(&[b, out_side, self.pool, out_side, self.pool, hidden])
            .map_err(map_err("gemma3_pool_reshape"))?
            .mean(4)
            .map_err(map_err("gemma3_pool_x"))?
            .mean(2)
            .map_err(map_err("gemma3_pool_y"))?
            .reshape(&[b, out_side * out_side, hidden])
            .map_err(map_err("gemma3_pool_flatten"))?;
        let normed = candle_nn::ops::rms_norm(&pooled, &self.soft_emb_norm, self.eps as f32)
            .map_err(map_err("gemma3_soft_emb_norm"))?;
        normed
            .broadcast_matmul(&self.input_projection)
            .map_err(map_err("gemma3_input_projection"))
    }

    pub fn llm_hidden_dim(&self) -> usize {
        self.input_projection.dims().last().copied().unwrap_or(0)
    }

    /// Tokens one image becomes.
    pub fn tokens_per_image(&self, patches_per_side: usize) -> usize {
        (patches_per_side / self.pool).pow(2)
    }
}

/// How an encoder's features reach the language model — chosen ONCE, from the
/// file's `clip.projector_type`, by [`load_from_mmproj_gguf`].
pub enum Projector {
    /// LLaVA's 2-layer MLP (`mlp`, the default of a file that names none).
    Llava(MultimodalProjection),
    /// Gemma 3's pool + RMS norm + linear (`gemma3`).
    Gemma3(Gemma3Projection),
}

impl Projector {
    /// LLM hidden dimension this projection targets.
    pub fn llm_hidden_dim(&self) -> usize {
        match self {
            Self::Llava(p) => p.llm_hidden_dim(),
            Self::Gemma3(p) => p.llm_hidden_dim(),
        }
    }
}

// ── VLM Pipeline Integration ──

/// Holds a loaded vision encoder + projection, ready for inference.
/// Stored in SharedState for reuse across requests.
pub struct VisionModule {
    pub encoder: VisionEncoder,
    pub projection: Projector,
    /// How an image is made the encoder's square.
    pub preprocess: Preprocess,
    pub device: Device,
}

impl VisionModule {
    /// Number of tokens one image becomes in the prompt.
    /// CLIP ViT-L/14 @ 336px (LLaVA): (336/14)^2 = 576; Gemma 3: 256.
    pub fn num_image_tokens(&self) -> usize {
        match &self.projection {
            Projector::Llava(_) => self.encoder.num_vision_tokens(),
            Projector::Gemma3(p) => p.tokens_per_image(self.encoder.patches_per_side()),
        }
    }

    /// Process images from a multimodal request and produce LLM-space embeddings.
    ///
    /// Returns a tensor of shape (total_image_tokens, llm_hidden_dim) ready to be
    /// concatenated with the text token embeddings in the split model's forward pass.
    pub fn encode_images(&self, images: &[ImageData]) -> Result<Tensor, SwarmError> {
        if images.is_empty() {
            return Err(SwarmError::Inference("No images to encode".into()));
        }

        let start = std::time::Instant::now();
        let pixel_values = preprocess_images(images, &self.preprocess, &self.device)?;

        // (B, 3, H, W) → (B, num_patches, vision_hidden)
        let vision_features = self.encoder.forward(&pixel_values)?;

        // (B, num_patches, vision_hidden) → (B, num_tokens, llm_hidden)
        let projected = match &self.projection {
            Projector::Llava(p) => p.forward(&vision_features)?,
            Projector::Gemma3(p) => p.forward(&vision_features, self.encoder.patches_per_side())?,
        };

        // Flatten batch: (B, num_tokens, llm_hidden) → (B * num_tokens, llm_hidden)
        let (b, n, h) = projected
            .dims3()
            .map_err(|e| SwarmError::Inference(format!("Vision output dims: {e}")))?;

        let result = projected
            .reshape(&[b * n, h])
            .map_err(|e| SwarmError::Inference(format!("Vision output reshape: {e}")))?;

        tracing::debug!(
            image_count = images.len(),
            patch_count = b * n,
            elapsed_ms = start.elapsed().as_millis() as u64,
            "DIAG: encode_images complete"
        );

        Ok(result)
    }
}

// ── Where an image sits in the prompt ──
//
// An image travels from the API edge to the first segment as a MARKER in the
// prompt's text, where the caller put it (llama.cpp's server and `mtmd` do the
// same, with the same string), and becomes one [`IMAGE_TOKEN_ID`] per image
// when the prompt is tokenized. The first segment replaces each with that
// image's rows. Before #115 (2) nothing put a marker in a prompt at all, so
// every image was PREPENDED — ahead of the BOS and the system turn — and the
// coordinator counted LLaVA's `<image>` text as one position.

/// The text that stands for one image in a rendered prompt — llama.cpp's
/// `mtmd_default_marker()`. Inserted only for a message that carries images,
/// so a request without one tokenizes this string as the text it is.
pub const MEDIA_MARKER: &str = "<__media__>";

/// The id a [`MEDIA_MARKER`] becomes in a tokenized prompt: no vocabulary has
/// it, so the first segment finds it unambiguously.
pub const IMAGE_TOKEN_ID: i64 = -1;

/// One part of a message's content, in the order the caller gave them.
pub enum ContentPiece<'a> {
    Text(&'a str),
    Image,
}

/// A message's text with a [`MEDIA_MARKER`] where each image was — llama.cpp's
/// rule (`common/chat.cpp`, `concat_content_parts`): text parts are joined by a
/// newline, and no newline goes before or after an image. Every API surface
/// builds a message's content through this, so an image sits in the same
/// place whichever one it came in by.
pub fn join_content<'a>(pieces: impl IntoIterator<Item = ContentPiece<'a>>) -> String {
    let mut text = String::new();
    let mut last_was_image = false;
    for piece in pieces {
        match piece {
            ContentPiece::Text(t) => {
                if !last_was_image && !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(t);
                last_was_image = false;
            }
            ContentPiece::Image => {
                text.push_str(MEDIA_MARKER);
                last_was_image = true;
            }
        }
    }
    text
}

/// A rendered prompt with each image's marker wrapped the way the model's
/// family was trained to see an image — llama.cpp's `img_beg` / `img_end`
/// (`mtmd.cpp`): Gemma 3 reads `<start_of_image>` … `<end_of_image>`, both
/// CONTROL tokens its tokenizer matches whole; LLaVA reads the bare image.
///
/// Called where every path's prompt is built
/// (`PipelineExecutor::build_prompt_with_header`), so the coordinator's count
/// of positions and the first segment's tokens describe the same prompt.
pub fn wrap_images_for_family(prompt: String, architecture: Option<&str>) -> String {
    match architecture {
        Some("gemma3") if prompt.contains(MEDIA_MARKER) => prompt.replace(
            MEDIA_MARKER,
            &format!("<start_of_image>{MEDIA_MARKER}<end_of_image>"),
        ),
        _ => prompt,
    }
}

/// A prompt carrying images, tokenized: the text between markers as the
/// tokenizer reads it, one [`IMAGE_TOKEN_ID`] per marker. Only the opening
/// piece may get a BOS (`SplitTokenizer::encode`); the rest continue it —
/// llama.cpp's `mtmd_tokenize`, which tokenizes each piece on its own with
/// `add_special` only for the first.
///
/// The ONE tokenization of an image prompt: the first segment's worker builds
/// its input from it and the coordinator counts the prompt's positions from it
/// ([`positions_with_images`]), so the two cannot disagree.
pub fn encode_prompt_with_images(
    tokenizer: &crate::inference::split::SplitTokenizer,
    prompt: &str,
) -> Vec<i64> {
    let mut ids = Vec::new();
    for (i, piece) in prompt.split(MEDIA_MARKER).enumerate() {
        if i == 0 {
            ids.extend(tokenizer.encode(piece));
        } else {
            ids.push(IMAGE_TOKEN_ID);
            if !piece.is_empty() {
                ids.extend(tokenizer.encode_continuation(piece));
            }
        }
    }
    ids
}

/// How many positions a tokenized image prompt occupies once each marker is
/// replaced by its image's rows: `image_rows` is every image's together, as
/// the vision payload's header states it ([`wire_rows`]).
pub fn positions_with_images(ids: &[i64], image_rows: usize) -> usize {
    let markers = ids.iter().filter(|&&id| id == IMAGE_TOKEN_ID).count();
    ids.len() - markers + image_rows
}

/// Whether a family reads each picture's rows both ways — every row of a
/// picture attending to every other row of it, the text around it still in
/// order. llama.cpp's `mtmd_decode_use_non_causal`: Gemma 3's projector
/// (`LLAMA_NON_CAUSAL_TYPE_ALL`: every layer, the sliding window still
/// applied). LLaVA reads them in order. Gemma 4's pictures are refused
/// before this is asked (#115). The ONE answer, asked where a prompt's
/// spans are computed ([`image_spans`]).
pub fn rows_attend_both_ways(architecture: Option<&str>) -> bool {
    matches!(architecture, Some("gemma3"))
}

/// The most picture spans one prompt may carry. The vision payload is capped
/// at 50 MB, about 40 of Gemma 3's pictures; a list longer than this is not
/// a prompt this node built.
pub const MAX_IMAGE_SPANS: usize = 256;

/// The positions each picture's rows take in a prompt tokenized with its
/// markers ([`encode_prompt_with_images`]), counted from position 0:
/// `image_rows` is every picture's rows together ([`wire_rows`]), split
/// evenly between the markers as [`splice_image_rows`] splits them. Empty
/// when there is no marker or the rows do not divide — the splice refuses
/// that prompt.
pub fn image_spans(ids: &[i64], image_rows: usize) -> Vec<std::ops::Range<usize>> {
    let markers = ids.iter().filter(|&&id| id == IMAGE_TOKEN_ID).count();
    if markers == 0 || !image_rows.is_multiple_of(markers) {
        return Vec::new();
    }
    let per_image = image_rows / markers;
    let mut spans = Vec::with_capacity(markers);
    let mut position = 0usize;
    for &id in ids {
        if id == IMAGE_TOKEN_ID {
            spans.push(position..position + per_image);
            position += per_image;
        } else {
            position += 1;
        }
    }
    spans
}

/// Spans a segment may apply: ascending, disjoint, none empty, no more than
/// [`MAX_IMAGE_SPANS`]. They arrive from the coordinator, so a segment checks
/// them before it builds a mask from them.
pub fn spans_are_well_formed(spans: &[std::ops::Range<usize>]) -> bool {
    spans.len() <= MAX_IMAGE_SPANS
        && spans.iter().all(|s| s.start < s.end)
        && spans.windows(2).all(|w| w[0].end <= w[1].start)
}

/// A tokenized prompt's embeddings with each image's rows where its marker is.
///
/// `embed_text` embeds one run of ordinary ids as `(1, n, hidden)`, scaled as
/// the model scales a token's embedding (Gemma's `sqrt(hidden)`); an image's
/// rows are NOT scaled — llama.cpp's Gemma 3 graph scales `ubatch.token` input
/// only. `vision` is `(rows, hidden)`, every image's rows in order, split
/// evenly between the markers.
///
/// The ONE splice, for the first segment (`SplitModel::forward_multimodal`)
/// and a prompt embedded on this node (`LocalEmbedder::embed_prompt`). A
/// prompt with no marker is refused: the old fallback prepended the image,
/// and a count that does not divide means the payload is not this prompt's.
pub fn splice_image_rows(
    ids: &[i64],
    vision: &Tensor,
    mut embed_text: impl FnMut(&[i64]) -> Result<Tensor, SwarmError>,
) -> Result<Tensor, SwarmError> {
    let map_err = |msg: &'static str| {
        move |e: candle_core::Error| SwarmError::Inference(format!("{msg}: {e}"))
    };
    let markers = ids.iter().filter(|&&id| id == IMAGE_TOKEN_ID).count();
    let (rows, hidden) = vision.dims2().map_err(map_err("vision_dims"))?;
    if markers == 0 || !rows.is_multiple_of(markers) {
        return Err(SwarmError::Validation(format!(
            "the prompt marks {markers} place(s) for an image, and the image data \
             has {rows} rows, which do not divide between them"
        )));
    }
    let per_image = rows / markers;
    let mut parts: Vec<Tensor> = Vec::with_capacity(2 * markers + 1);
    // Where the text's embeddings live, which the image rows are moved to.
    let mut text_home: Option<(Device, candle_core::DType)> = None;
    for (i, run) in ids.split(|&id| id == IMAGE_TOKEN_ID).enumerate() {
        if i > 0 {
            parts.push(
                vision
                    .narrow(0, (i - 1) * per_image, per_image)
                    .and_then(|t| t.reshape(&[1, per_image, hidden]))
                    .map_err(map_err("vision_rows"))?,
            );
        }
        if !run.is_empty() {
            let text = embed_text(run)?;
            text_home.get_or_insert_with(|| (text.device().clone(), text.dtype()));
            parts.push(text);
        }
    }
    let parts = match text_home {
        Some((device, dtype)) => parts
            .into_iter()
            .map(|t| t.to_device(&device).and_then(|t| t.to_dtype(dtype)))
            .collect::<candle_core::Result<Vec<_>>>()
            .map_err(map_err("vision_text_device"))?,
        None => parts,
    };
    let merged = Tensor::cat(&parts, 1).map_err(map_err("vision_text_cat"))?;
    tracing::info!(
        markers,
        rows_per_image = per_image,
        hidden,
        positions = merged.dim(1).unwrap_or(0),
        "DIAG: splice_image_rows"
    );
    Ok(merged)
}

// ── The vision payload on the wire ──
//
// 8-byte header (rows u32 LE + hidden u32 LE), then zstd of the rows as f16.
// Written by `PipelineExecutor::compress_vision_embeddings` (and a remote
// encoder, `daemon::dispatch::vision`), read here.

/// The rows a vision payload's header states, or `None` for a payload too
/// short to have one.
pub fn wire_rows(payload: &[u8]) -> Option<usize> {
    let header: [u8; 4] = payload.get(..4)?.try_into().ok()?;
    Some(u32::from_le_bytes(header) as usize)
}

/// Decode a vision payload into a `(rows, hidden)` f32 tensor on the processor.
///
/// A payload that cannot be read is the request's failure, never an image
/// quietly dropped: the old reader logged and went on without it, and the
/// model then answered about a picture it never saw.
pub fn decode_wire_embeddings(payload: &[u8]) -> Result<Tensor, SwarmError> {
    /// Larger than any encoder's output for a few images (Gemma 3: 256 × 2560
    /// × 2 bytes = 1.3 MB an image), small enough that a peer cannot make a
    /// worker allocate without bound.
    const MAX_VISION_EMBEDDING_BYTES: usize = 50 * 1024 * 1024;
    let bad = |why: String| SwarmError::Validation(format!("unreadable image data: {why}"));
    if payload.len() < 8 {
        return Err(bad(format!(
            "{} bytes, shorter than its header",
            payload.len()
        )));
    }
    let rows = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let hidden = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;
    let mut raw = Vec::new();
    {
        use std::io::Read;
        let decoder = zstd::stream::read::Decoder::new(std::io::Cursor::new(&payload[8..]))
            .map_err(|e| bad(format!("zstd: {e}")))?;
        // One byte past the cap, so a payload that inflates past it is caught
        // without inflating all of it.
        decoder
            .take(MAX_VISION_EMBEDDING_BYTES as u64 + 1)
            .read_to_end(&mut raw)
            .map_err(|e| bad(format!("zstd: {e}")))?;
    }
    if raw.len() > MAX_VISION_EMBEDDING_BYTES || !raw.len().is_multiple_of(2) {
        return Err(bad(format!("{} bytes of rows", raw.len())));
    }
    // checked_mul: an adversarial header near u32::MAX must not wrap past the
    // length check on a 32-bit platform.
    if rows.checked_mul(hidden) != Some(raw.len() / 2) {
        return Err(bad(format!(
            "header says {rows} x {hidden}, payload holds {} values",
            raw.len() / 2
        )));
    }
    let values: Vec<f32> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| half::f16::from_le_bytes(*b).to_f32())
        .collect();
    // SEC: a non-finite row would poison the request's KV cache for every
    // later step — the same rule as `spec_logits` and AllReduce sums.
    if values.iter().any(|v| !v.is_finite()) {
        return Err(SwarmError::Validation(
            "Vision embeddings contain non-finite values".into(),
        ));
    }
    Tensor::from_vec(values, &[rows, hidden], &Device::Cpu)
        .map_err(|e| SwarmError::Inference(format!("vision tensor: {e}")))
}

/// Check if a list of chat messages contains any images.
pub fn has_images(messages: &[crate::types::ChatMessage]) -> bool {
    messages.iter().any(|m| !m.images.is_empty())
}

/// Collect all images from chat messages in order.
pub fn collect_images(messages: &[crate::types::ChatMessage]) -> Vec<&ImageData> {
    messages.iter().flat_map(|m| m.images.iter()).collect()
}

// ── mmproj GGUF Loading ──

/// Decide whether an mmproj GGUF uses the legacy *inverted* CLIP FFN naming.
///
/// GGUF CLIP exports disagree on this. LLaVA-1.5-era conversions store the
/// expansion (applied first, before GELU) as `ffn_down` and the contraction as
/// `ffn_up` — the inverse of the text-model convention used everywhere else in
/// this codebase. Newer exports name them correctly. Guessing wrong is not
/// silent: the first matmul fails on a dimension mismatch.
///
/// So decide from the shape, mirroring llama.cpp's `is_ffn_swapped` check in
/// `tools/mtmd/clip.cpp` — the expansion is whichever weight *consumes*
/// `hidden_size` inputs. `dims` is candle order (`[out, in]`); candle reverses
/// GGUF's `ne` on read, so this is llama.cpp's `ff_down_w->ne[0] == n_embd`
/// spelled the other way round.
///
/// A square FFN (`n_ff == hidden_size`) is genuinely undecidable from shape
/// alone. That returns `true`, preserving the legacy reading — the only
/// convention any mmproj this code has been exercised against actually uses.
/// No CLIP-family vision tower does this in practice; they are all 4×.
fn clip_ffn_is_swapped(ffn_down_dims: &[usize], hidden_size: usize) -> bool {
    ffn_down_dims.len() == 2 && ffn_down_dims[1] == hidden_size
}

/// Load a VisionModule from a LLaVA-style mmproj GGUF file.
///
/// The mmproj GGUF contains the CLIP ViT encoder weights and multimodal projection
/// weights. Tensor names follow the llama.cpp convention:
///
/// - `v.patch_embd.weight` — patch embedding conv projection
/// - `v.class_embd` — CLS token
/// - `v.position_embd.weight` — positional embeddings
/// - `v.blk.N.attn_q.weight/bias`, `v.blk.N.attn_k.weight/bias`, `v.blk.N.attn_v.weight/bias`
/// - `v.blk.N.attn_out.weight/bias` — attention output projection
/// - `v.blk.N.ln1.weight/bias`, `v.blk.N.ln2.weight/bias` — layer norms
/// - `v.blk.N.ffn_down.weight/bias`, `v.blk.N.ffn_up.weight/bias` — MLP
/// - `v.post_ln.weight/bias` — post layer norm
/// - `mm.0.weight/bias`, `mm.2.weight/bias` — multimodal projection (2-layer MLP with GELU)
///
/// The two MLP tensors are resolved by shape rather than by name — see
/// [`clip_ffn_is_swapped`].
pub fn load_from_mmproj_gguf(
    path: &std::path::Path,
    device: &Device,
) -> Result<VisionModule, SwarmError> {
    use candle_core::DType;

    // Buffered: the header walk is thousands of tiny reads and each one is a
    // syscall on a bare `File` (see `split::read_gguf_header`). The reader is
    // kept because the tensor loads below read through it too.
    let file = std::fs::File::open(path)
        .map_err(|e| SwarmError::Inference(format!("Failed to open mmproj GGUF: {e}")))?;
    let mut file = std::io::BufReader::new(file);

    let ct = candle_core::quantized::gguf_file::Content::read(&mut file)
        .map_err(|e| SwarmError::Inference(format!("Failed to parse mmproj GGUF: {e}")))?;

    // Extract vision config from GGUF metadata
    let get_u32 = |key: &str| -> Result<u32, SwarmError> {
        ct.metadata
            .get(key)
            .and_then(|v| v.to_u32().ok())
            .ok_or_else(|| SwarmError::Inference(format!("Missing metadata: {key}")))
    };

    let image_size = get_u32("clip.vision.image_size")?;
    let patch_size = get_u32("clip.vision.patch_size")?;
    let hidden_size = get_u32("clip.vision.embedding_length")? as usize;
    let num_heads = get_u32("clip.vision.attention.head_count")? as usize;
    let num_layers = get_u32("clip.vision.block_count")? as usize;
    if patch_size == 0 || !image_size.is_multiple_of(patch_size) || num_heads == 0 {
        return Err(SwarmError::Inference(format!(
            "mmproj geometry is not usable: image {image_size}, patch {patch_size}, heads {num_heads}"
        )));
    }

    // The projector decides everything after the encoder (llama.cpp's
    // `clip.projector_type`; a file that names none is LLaVA's `mlp`). One is
    // read here and nowhere else — an unknown kind is refused by name rather
    // than loaded as LLaVA, which is what every file was before (#115 (2)).
    let projector_type = ct
        .metadata
        .get("clip.projector_type")
        .and_then(|v| v.to_string().ok())
        .map(String::as_str)
        .unwrap_or("mlp");
    let kind = match projector_type {
        "mlp" => ProjectorKind::Llava,
        "gemma3" => ProjectorKind::Gemma3,
        other => {
            return Err(SwarmError::Validation(format!(
                "This model's image reader (projector type '{other}') is not supported yet. \
                 Supported: mlp (LLaVA), gemma3."
            )))
        }
    };

    let dim0_of = |name: &str| ct.tensor_infos.get(name).map(|t| t.shape.dims()[0]);
    let dims_of = |name: &str| ct.tensor_infos.get(name).map(|t| t.shape.dims().to_vec());
    let (projection_dim, llm_hidden_dim) = match kind {
        // mm.0's output width; the language model's width is mm.2's output.
        ProjectorKind::Llava => (
            dim0_of("mm.0.weight").unwrap_or(hidden_size),
            dim0_of("mm.2.weight").ok_or_else(|| {
                SwarmError::Inference(
                    "Missing mm.2.weight — cannot determine LLM hidden dim".into(),
                )
            })?,
        ),
        // candle order (vision_hidden, llm_hidden): the projection is x @ W.
        ProjectorKind::Gemma3 => match dims_of("mm.input_projection.weight").as_deref() {
            Some([vision, llm]) if *vision == hidden_size => (*llm, *llm),
            other => {
                return Err(SwarmError::Inference(format!(
                    "Gemma 3 mmproj: mm.input_projection.weight is {other:?}, \
                     expected [{hidden_size}, <language model width>]"
                )))
            }
        },
    };

    // Per-channel mean and spread: the file's own (SigLIP's are 0.5), CLIP's
    // published values where a file states none.
    let channel_triple = |key: &str| -> Option<[f32; 3]> {
        let v = ct.metadata.get(key)?.to_vec().ok()?;
        match v.as_slice() {
            [a, b, c] => Some([a.to_f32().ok()?, b.to_f32().ok()?, c.to_f32().ok()?]),
            _ => None,
        }
    };
    let mean = channel_triple("clip.vision.image_mean").unwrap_or(CLIP_MEAN);
    let preprocess = Preprocess {
        size: image_size,
        mean,
        std: channel_triple("clip.vision.image_std").unwrap_or(CLIP_STD),
        // llama.cpp's `clip.cpp` per projector: LLaVA pads with the mean and
        // resizes bicubic (the default), Gemma 3 pads black and resizes
        // bilinear. LLaVA was stretched and resized bilinear here (#310).
        pad: match kind {
            ProjectorKind::Llava => mean_as_bytes(mean),
            ProjectorKind::Gemma3 => [0, 0, 0],
        },
        resample: match kind {
            ProjectorKind::Llava => Resample::Bicubic,
            ProjectorKind::Gemma3 => Resample::Bilinear,
        },
    };
    let flag = |key: &str| {
        ct.metadata
            .get(key)
            .and_then(|v| v.to_bool().ok())
            .unwrap_or(false)
    };
    let activation = EncoderActivation::from_keys(flag("clip.use_gelu"), flag("clip.use_silu"))?;
    let feature_layers: Option<Vec<i64>> =
        ct.metadata
            .get("clip.vision.feature_layer")
            .map(|v| match v.to_vec() {
                Ok(items) => items.iter().filter_map(gguf_int).collect(),
                Err(_) => gguf_int(v).into_iter().collect(),
            });
    let depth_rule = match std::env::var("SWARMLLM_LLAVA_DEPTH").as_deref() {
        Ok("llama-cpp") => DepthRule::LlamaCpp,
        _ => DepthRule::Llava,
    };
    let depth = encoder_depth(kind, num_layers, feature_layers.as_deref(), depth_rule)?;

    let config = VisionConfig {
        image_size,
        patch_size,
        vision_hidden_size: hidden_size as u32,
        vision_num_layers: depth as u32,
        vision_num_heads: num_heads as u32,
        projection_dim: projection_dim as u32,
    };

    tracing::info!(
        projector_type,
        image_size,
        patch_size,
        hidden_size,
        num_heads,
        num_layers,
        depth,
        ?activation,
        projection_dim,
        llm_hidden_dim,
        "Loading mmproj vision encoder"
    );

    // Use a macro to avoid closure borrow issues with &mut file
    macro_rules! load {
        ($name:expr) => {{
            let qt = ct
                .tensor(&mut file, $name, device)
                .map_err(|e| SwarmError::Inference(format!("{}: {e}", $name)))?;
            qt.dequantize(device)
                .map_err(|e| SwarmError::Inference(format!("{}: {e}", $name)))?
        }};
    }
    macro_rules! load_opt {
        ($name:expr) => {{
            if ct.tensor_infos.contains_key($name) {
                Some(load!($name))
            } else {
                None
            }
        }};
    }

    // Layer norm epsilon from metadata (default 1e-5)
    let ln_eps = ct
        .metadata
        .get("clip.vision.attention.layer_norm_epsilon")
        .and_then(|v| v.to_f32().ok())
        .map(|v| v as f64)
        .unwrap_or(1e-5);

    // ── Patch embedding ──
    // GGUF stores conv2d weight as [kH, kW, C_in, C_out] but PatchEmbedding
    // expects (hidden_dim, patch_dim) where patch_dim = 3 * ps * ps.
    // We need to reshape [14, 14, 3, 1024] → (1024, 14*14*3) = (hidden, patch_dim)
    let patch_proj_raw = load!("v.patch_embd.weight");
    let patch_proj_weight = {
        let dims = patch_proj_raw.dims().to_vec();
        let total_elements: usize = dims.iter().product();
        let ps = patch_size as usize;
        let patch_dim = 3 * ps * ps;
        tracing::info!(
            ?dims,
            total_elements,
            hidden_size,
            patch_dim,
            "patch_embd.weight raw shape"
        );

        // The weight is a conv2d kernel. Regardless of how many dims candle gives us,
        // we need to reshape to (hidden_size, patch_dim) for our manual patch projection.
        // GGUF stores [kH, kW, C_in, C_out] which may be loaded as 2D [kH, kW*C_in*C_out]
        // or 4D. Either way, total elements = hidden_size * patch_dim.
        if total_elements != hidden_size * patch_dim {
            return Err(SwarmError::Inference(format!(
                "patch_embd size mismatch: {total_elements} vs {}",
                hidden_size * patch_dim
            )));
        }
        patch_proj_raw
            .reshape(&[hidden_size, patch_dim])
            .map_err(|e| SwarmError::Inference(format!("patch_embd reshape: {e}")))?
    };
    let patch_proj_bias = load_opt!("v.patch_embd.bias")
        .unwrap_or_else(|| Tensor::zeros(&[hidden_size], DType::F32, device).unwrap());
    // CLIP prepends a class token; SigLIP (Gemma 3's encoder) has none.
    let cls_token = match load_opt!("v.class_embd") {
        Some(t) => Some(
            t.reshape(&[1, 1, hidden_size])
                .map_err(|e| SwarmError::Inference(format!("cls_reshape: {e}")))?,
        ),
        None => None,
    };

    // Position embedding: GGUF shape [hidden, num_pos] → need (1, num_pos, hidden)
    let position_embedding_raw = load!("v.position_embd.weight");
    let pos_dims = position_embedding_raw.dims().to_vec();
    tracing::debug!(?pos_dims, "position_embd.weight raw shape");
    let position_embedding = if pos_dims.len() == 2 && pos_dims[0] == hidden_size {
        // Shape is (hidden, num_pos) — transpose to (num_pos, hidden)
        let transposed = position_embedding_raw
            .t()
            .and_then(|t| t.contiguous())
            .map_err(|e| SwarmError::Inference(format!("pos_transpose: {e}")))?;
        let num_pos = transposed
            .dim(0)
            .map_err(|e| SwarmError::Inference(format!("pos_dim0: {e}")))?;
        transposed
            .reshape(&[1, num_pos, hidden_size])
            .map_err(|e| SwarmError::Inference(format!("pos_reshape: {e}")))?
    } else {
        let num_pos = position_embedding_raw
            .dim(0)
            .map_err(|e| SwarmError::Inference(format!("pos_dim0: {e}")))?;
        position_embedding_raw
            .reshape(&[1, num_pos, hidden_size])
            .map_err(|e| SwarmError::Inference(format!("pos_reshape: {e}")))?
    };

    let patch_embed = PatchEmbedding::new(
        patch_proj_weight,
        patch_proj_bias,
        cls_token,
        position_embedding,
        patch_size as usize,
    );

    // ── Pre layer norm (optional, applied before transformer blocks) ──
    let pre_ln = if ct.tensor_infos.contains_key("v.pre_ln.weight") {
        let w = load!("v.pre_ln.weight");
        let b = load!("v.pre_ln.bias");
        Some(candle_nn::LayerNorm::new(w, b, ln_eps))
    } else {
        None
    };

    // ── Transformer blocks ──
    // Resolve the FFN naming convention once — every block shares it.
    let ffn_swapped = ct
        .tensor_infos
        .get("v.blk.0.ffn_down.weight")
        .map(|t| clip_ffn_is_swapped(t.shape.dims(), hidden_size))
        .unwrap_or(true);
    let (fc1_name, fc2_name) = if ffn_swapped {
        tracing::debug!("mmproj uses legacy inverted CLIP FFN naming (ffn_down is the expansion)");
        ("ffn_down", "ffn_up")
    } else {
        ("ffn_up", "ffn_down")
    };

    // Only the blocks that run are loaded.
    let mut blocks = Vec::with_capacity(depth);
    for i in 0..depth {
        // Layer norms
        let ln1_w = load!(&format!("v.blk.{i}.ln1.weight"));
        let ln1_b = load!(&format!("v.blk.{i}.ln1.bias"));
        let ln1 = candle_nn::LayerNorm::new(ln1_w, ln1_b, ln_eps);

        let ln2_w = load!(&format!("v.blk.{i}.ln2.weight"));
        let ln2_b = load!(&format!("v.blk.{i}.ln2.bias"));
        let ln2 = candle_nn::LayerNorm::new(ln2_w, ln2_b, ln_eps);

        // Attention — mmproj has separate Q/K/V, we concat into fused QKV
        let q_w = load!(&format!("v.blk.{i}.attn_q.weight"));
        let k_w = load!(&format!("v.blk.{i}.attn_k.weight"));
        let v_w = load!(&format!("v.blk.{i}.attn_v.weight"));
        let qkv_weight = Tensor::cat(&[&q_w, &k_w, &v_w], 0)
            .map_err(|e| SwarmError::Inference(format!("qkv_weight_cat: {e}")))?;

        let qkv_bias = if ct
            .tensor_infos
            .contains_key(&format!("v.blk.{i}.attn_q.bias"))
        {
            let q_b = load!(&format!("v.blk.{i}.attn_q.bias"));
            let k_b = load!(&format!("v.blk.{i}.attn_k.bias"));
            let v_b = load!(&format!("v.blk.{i}.attn_v.bias"));
            Some(
                Tensor::cat(&[&q_b, &k_b, &v_b], 0)
                    .map_err(|e| SwarmError::Inference(format!("qkv_bias_cat: {e}")))?,
            )
        } else {
            None
        };

        let attn_qkv = Linear::new(qkv_weight, qkv_bias);

        // Attention output projection
        let attn_out_w = load!(&format!("v.blk.{i}.attn_out.weight"));
        let attn_out_b = load_opt!(&format!("v.blk.{i}.attn_out.bias"));
        let attn_proj = Linear::new(attn_out_w, attn_out_b);

        // MLP — `fc1_name` is the expansion, applied before GELU. Which GGUF
        // tensor that is depends on the export convention; see
        // `clip_ffn_is_swapped`.
        let fc1_w = load!(&format!("v.blk.{i}.{fc1_name}.weight"));
        let fc1_b = load_opt!(&format!("v.blk.{i}.{fc1_name}.bias"));
        let mlp_fc1 = Linear::new(fc1_w, fc1_b);

        let fc2_w = load!(&format!("v.blk.{i}.{fc2_name}.weight"));
        let fc2_b = load_opt!(&format!("v.blk.{i}.{fc2_name}.bias"));
        let mlp_fc2 = Linear::new(fc2_w, fc2_b);

        blocks.push(
            VisionTransformerBlock::new(
                ln1,
                attn_qkv,
                attn_proj,
                ln2,
                mlp_fc1,
                mlp_fc2,
                num_heads,
                hidden_size,
            )
            .with_activation(activation),
        );
    }

    // ── Post layer norm, where the file has one (LLaVA's does not) ──
    let final_ln = if ct.tensor_infos.contains_key("v.post_ln.weight") {
        let post_ln_w = load!("v.post_ln.weight");
        let post_ln_b = load!("v.post_ln.bias");
        Some(candle_nn::LayerNorm::new(post_ln_w, post_ln_b, ln_eps))
    } else {
        None
    };

    tracing::info!(num_blocks = blocks.len(), "Vision encoder blocks loaded");

    let mut encoder = VisionEncoder::new(patch_embed, blocks, final_ln, config);
    encoder.pre_ln = pre_ln;

    let projection = match kind {
        // ── LLaVA: 2-layer MLP, mm.0 → GELU → mm.2 ──
        ProjectorKind::Llava => {
            let mm0_w = load!("mm.0.weight");
            let mm0_b = load_opt!("mm.0.bias");
            let mm2_w = load!("mm.2.weight");
            let mm2_b = load_opt!("mm.2.bias");
            let proj1 = Linear::new(mm0_w, mm0_b);
            let proj2 = Linear::new(mm2_w, mm2_b);
            Projector::Llava(MultimodalProjection::new(proj1, proj2, llm_hidden_dim))
        }
        // ── Gemma 3: 4×4 average pool, RMS norm, linear ──
        ProjectorKind::Gemma3 => {
            let pool = ct
                .metadata
                .get("clip.vision.projector.scale_factor")
                .and_then(|v| v.to_u32().ok())
                .unwrap_or(4) as usize;
            let side = (image_size / patch_size) as usize;
            if pool == 0 || !side.is_multiple_of(pool) {
                return Err(SwarmError::Inference(format!(
                    "Gemma 3 mmproj: a {side}-patch side does not pool by {pool}"
                )));
            }
            Projector::Gemma3(Gemma3Projection::new(
                load!("mm.soft_emb_norm.weight"),
                load!("mm.input_projection.weight"),
                pool,
                ln_eps,
            ))
        }
    };

    tracing::info!(llm_hidden_dim, "VisionModule loaded from mmproj GGUF");

    Ok(VisionModule {
        encoder,
        projection,
        preprocess,
        device: device.clone(),
    })
}

/// The projector kinds [`load_from_mmproj_gguf`] builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectorKind {
    Llava,
    Gemma3,
}

/// A GGUF integer of any width, as llama.cpp's `get_arr_int` reads one.
fn gguf_int(v: &candle_core::quantized::gguf_file::Value) -> Option<i64> {
    v.to_i32()
        .map(i64::from)
        .or_else(|_| v.to_u32().map(i64::from))
        .or_else(|_| v.to_i64())
        .or_else(|_| v.to_u64().map(|u| u as i64))
        .ok()
}

/// Whose reading of a LLaVA encoder's depth [`encoder_depth`] follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DepthRule {
    /// LLaVA's own: the hidden state after 23 of CLIP's 24 blocks.
    Llava,
    /// llama.cpp's, one block fewer on a file without `feature_layer` —
    /// `SWARMLLM_LLAVA_DEPTH=llama-cpp`, to compare the rest of the reader with
    /// llama.cpp's on equal terms (`examples/image_rows_probe.rs`).
    LlamaCpp,
}

/// How many of a file's encoder blocks run (#310).
///
/// LLaVA 1.5 reads CLIP's hidden state after 23 of its 24 blocks
/// (`mm_vision_select_layer = -2`), and llama.cpp's converter keeps exactly
/// those 23: it pops the last block and writes `block_count` 23
/// (`tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py`). So every
/// block the file holds runs. A newer conversion names the hidden state it
/// reads as `clip.vision.feature_layer`, counted in blocks (0 is the
/// embeddings), and keeps that many; it is honoured when it is one layer, and
/// several — granite's stacked features — are refused. llama.cpp's LLaVA graph
/// (`models/llava.cpp`) runs `n_layer - 1` blocks of a file WITHOUT the key:
/// on the 23-block files LLaVA 1.5 ships as, one block short of LLaVA's own
/// reading. `DepthRule::LlamaCpp` runs that, for comparison. Gemma 3's SigLIP
/// runs every block (llama.cpp's `build_vit`).
fn encoder_depth(
    kind: ProjectorKind,
    block_count: usize,
    feature_layers: Option<&[i64]>,
    rule: DepthRule,
) -> Result<usize, SwarmError> {
    if kind != ProjectorKind::Llava {
        return Ok(block_count);
    }
    match feature_layers {
        None | Some([]) => Ok(match rule {
            DepthRule::Llava => block_count,
            DepthRule::LlamaCpp => block_count.saturating_sub(1),
        }),
        Some(&[layer]) if layer >= 0 && layer as usize <= block_count => Ok(layer as usize),
        Some(&[layer]) => Err(SwarmError::Validation(format!(
            "This model's image reader names feature layer {layer}, but holds {block_count} \
             encoder layers."
        ))),
        Some(layers) => Err(SwarmError::Validation(format!(
            "This model's image reader stacks the features of {} encoder layers \
             (clip.vision.feature_layer {layers:?}); that is not supported yet.",
            layers.len()
        ))),
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    // ── CLIP FFN naming convention ──
    //
    // Shapes are candle order (`[out, in]`); candle reverses GGUF's `ne` on
    // read. The legacy values below are the real ones from
    // `llava-v1.5-7b-mmproj-f16.gguf` (CLIP ViT-L/14-336: hidden 1024,
    // n_ff 4096), whose GGUF stores `ffn_down` with `ne = [1024, 4096]`.

    #[test]
    fn legacy_llava_mmproj_is_detected_as_swapped() {
        // ffn_down consumes 1024 and emits 4096 — it is really the expansion.
        assert!(clip_ffn_is_swapped(&[4096, 1024], 1024));
    }

    #[test]
    fn correctly_named_mmproj_is_not_swapped() {
        // ffn_down consumes 4096 and emits 1024 — a genuine contraction.
        assert!(!clip_ffn_is_swapped(&[1024, 4096], 1024));
    }

    #[test]
    fn square_ffn_keeps_the_legacy_reading() {
        // Undecidable from shape alone; documented to preserve legacy behaviour.
        assert!(clip_ffn_is_swapped(&[1024, 1024], 1024));
    }

    #[test]
    fn non_2d_ffn_shape_is_not_swapped() {
        assert!(!clip_ffn_is_swapped(&[1024], 1024));
        assert!(!clip_ffn_is_swapped(&[], 1024));
    }

    /// LLaVA's square: CLIP's mean and spread, padded with the mean, bicubic.
    fn llava_square(size: u32) -> Preprocess {
        Preprocess {
            size,
            mean: CLIP_MEAN,
            std: CLIP_STD,
            pad: mean_as_bytes(CLIP_MEAN),
            resample: Resample::Bicubic,
        }
    }

    #[test]
    fn preprocess_image_creates_correct_shape() {
        let img = ImageData {
            rgb_bytes: vec![128u8; 3 * 64 * 64],
            width: 64,
            height: 64,
        };
        let device = Device::Cpu;
        let tensor = preprocess_image(&img, &llava_square(224), &device).unwrap();
        assert_eq!(tensor.dims(), &[3, 224, 224]);
    }

    #[test]
    fn preprocess_image_normalizes_values() {
        // All-white image (255, 255, 255)
        let img = ImageData {
            rgb_bytes: vec![255u8; 3 * 16 * 16],
            width: 16,
            height: 16,
        };
        let device = Device::Cpu;
        let tensor = preprocess_image(&img, &llava_square(16), &device).unwrap();

        // After normalization: (1.0 - mean) / std
        // Channel 0: (1.0 - 0.48145466) / 0.26862954 ≈ 1.93
        let data: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();
        let first_pixel = data[0];
        assert!(
            (first_pixel - 1.93).abs() < 0.1,
            "Expected ~1.93, got {first_pixel}"
        );
    }

    #[test]
    fn preprocess_batch_creates_correct_shape() {
        let images = vec![
            ImageData {
                rgb_bytes: vec![128u8; 3 * 32 * 32],
                width: 32,
                height: 32,
            },
            ImageData {
                rgb_bytes: vec![64u8; 3 * 48 * 48],
                width: 48,
                height: 48,
            },
        ];
        let device = Device::Cpu;
        let batch = preprocess_images(&images, &llava_square(224), &device).unwrap();
        assert_eq!(batch.dims(), &[2, 3, 224, 224]);
    }

    #[test]
    fn preprocess_empty_batch_errors() {
        let device = Device::Cpu;
        let result = preprocess_images(&[], &llava_square(224), &device);
        assert!(result.is_err());
    }

    #[test]
    fn has_images_detects_correctly() {
        use crate::types::{ChatMessage, Role};

        let text_only = vec![ChatMessage {
            role: Role::User,
            content: "Hello".into(),
            images: vec![],
            tool: Default::default(),
        }];
        assert!(!has_images(&text_only));

        let with_image = vec![ChatMessage {
            role: Role::User,
            content: "What's in this image?".into(),
            images: vec![ImageData {
                rgb_bytes: vec![0u8; 3 * 4 * 4],
                width: 4,
                height: 4,
            }],
            tool: Default::default(),
        }];
        assert!(has_images(&with_image));
    }

    #[test]
    fn collect_images_gathers_all() {
        use crate::types::{ChatMessage, Role};

        let messages = vec![
            ChatMessage {
                role: Role::User,
                content: "First".into(),
                images: vec![ImageData {
                    rgb_bytes: vec![1u8; 48],
                    width: 4,
                    height: 4,
                }],
                tool: Default::default(),
            },
            ChatMessage {
                role: Role::User,
                content: "Second".into(),
                images: vec![
                    ImageData {
                        rgb_bytes: vec![2u8; 48],
                        width: 4,
                        height: 4,
                    },
                    ImageData {
                        rgb_bytes: vec![3u8; 48],
                        width: 4,
                        height: 4,
                    },
                ],
                tool: Default::default(),
            },
        ];
        let collected = collect_images(&messages);
        assert_eq!(collected.len(), 3);
    }

    #[test]
    fn vision_config_defaults() {
        let config = VisionConfig {
            image_size: 336,
            patch_size: 14,
            vision_hidden_size: 1024,
            vision_num_layers: 24,
            vision_num_heads: 16,
            projection_dim: 4096,
        };
        assert_eq!(config.image_size, 336);
        assert_eq!((config.image_size / config.patch_size).pow(2), 576); // 576 patches
    }

    #[test]
    fn patch_embedding_construction() {
        // Verify PatchEmbedding accepts correct weight shapes
        let device = Device::Cpu;
        let patch_size = 4;
        let hidden_dim = 16;
        let num_patches = 4; // (8/4)^2 = 4
        let patch_dim = 3 * patch_size * patch_size; // 48

        let proj_weight = Tensor::ones(&[hidden_dim, patch_dim], DType::F32, &device).unwrap();
        let proj_bias = Tensor::zeros(&[hidden_dim], DType::F32, &device).unwrap();
        let cls_token = Tensor::zeros(&[1, 1, hidden_dim], DType::F32, &device).unwrap();
        let pos_embed =
            Tensor::zeros(&[1, num_patches + 1, hidden_dim], DType::F32, &device).unwrap();

        let _patch_embed = PatchEmbedding::new(
            proj_weight,
            proj_bias,
            Some(cls_token),
            pos_embed,
            patch_size,
        );
        // Construction succeeds — forward requires specific candle matmul broadcasting
        // which is validated with real model weights in E2E testing
    }

    #[test]
    fn vision_transformer_block_forward() {
        use candle_nn::VarMap;
        let device = Device::Cpu;
        let hidden_dim = 32;
        let num_heads = 4;
        let vm = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &device);

        let ln1 = candle_nn::layer_norm(hidden_dim, 1e-5, vb.pp("ln1")).unwrap();
        let attn_qkv = candle_nn::linear(hidden_dim, 3 * hidden_dim, vb.pp("qkv")).unwrap();
        let attn_proj = candle_nn::linear(hidden_dim, hidden_dim, vb.pp("proj")).unwrap();
        let ln2 = candle_nn::layer_norm(hidden_dim, 1e-5, vb.pp("ln2")).unwrap();
        let fc1 = candle_nn::linear(hidden_dim, 4 * hidden_dim, vb.pp("fc1")).unwrap();
        let fc2 = candle_nn::linear(4 * hidden_dim, hidden_dim, vb.pp("fc2")).unwrap();

        let block = VisionTransformerBlock::new(
            ln1, attn_qkv, attn_proj, ln2, fc1, fc2, num_heads, hidden_dim,
        );

        // (1, 5, 32) → (1, 5, 32) — shape preserved
        let x = Tensor::randn(0f32, 0.1, &[1, 5, hidden_dim], &device).unwrap();
        let out = block.forward(&x).unwrap();
        assert_eq!(out.dims(), x.dims());
    }

    #[test]
    fn multimodal_projection_forward() {
        use candle_nn::VarMap;
        let device = Device::Cpu;
        let vision_hidden = 64;
        let proj_dim = 32;
        let llm_hidden = 16;
        let vm = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &device);

        let proj1 = candle_nn::linear(vision_hidden, proj_dim, vb.pp("p1")).unwrap();
        let proj2 = candle_nn::linear(proj_dim, llm_hidden, vb.pp("p2")).unwrap();
        let mm = MultimodalProjection::new(proj1, proj2, llm_hidden);

        // (1, 5, 64) → (1, 5, 16) — projects to LLM space
        let features = Tensor::randn(0f32, 0.1, &[1, 5, vision_hidden], &device).unwrap();
        let out = mm.forward(&features).unwrap();
        assert_eq!(out.dims(), &[1, 5, llm_hidden]);
        assert_eq!(mm.llm_hidden_dim(), llm_hidden);
    }

    #[test]
    fn vision_encoder_num_tokens() {
        use candle_nn::VarMap;
        let device = Device::Cpu;
        let patch_size = 4;
        let image_size = 8;
        let hidden_dim = 16;
        let num_heads = 2;
        let num_patches = (image_size / patch_size) * (image_size / patch_size); // 4
        let patch_dim = 3 * patch_size * patch_size;

        let proj_weight = Tensor::zeros(&[hidden_dim, patch_dim], DType::F32, &device).unwrap();
        let proj_bias = Tensor::zeros(&[hidden_dim], DType::F32, &device).unwrap();
        let cls_token = Tensor::zeros(&[1, 1, hidden_dim], DType::F32, &device).unwrap();
        let pos_embed =
            Tensor::zeros(&[1, num_patches + 1, hidden_dim], DType::F32, &device).unwrap();
        let patch_embed = PatchEmbedding::new(
            proj_weight,
            proj_bias,
            Some(cls_token),
            pos_embed,
            patch_size,
        );

        let vm = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &device);
        let ln1 = candle_nn::layer_norm(hidden_dim, 1e-5, vb.pp("ln1")).unwrap();
        let qkv = candle_nn::linear(hidden_dim, 3 * hidden_dim, vb.pp("qkv")).unwrap();
        let proj = candle_nn::linear(hidden_dim, hidden_dim, vb.pp("proj")).unwrap();
        let ln2 = candle_nn::layer_norm(hidden_dim, 1e-5, vb.pp("ln2")).unwrap();
        let fc1 = candle_nn::linear(hidden_dim, 4 * hidden_dim, vb.pp("fc1")).unwrap();
        let fc2 = candle_nn::linear(4 * hidden_dim, hidden_dim, vb.pp("fc2")).unwrap();
        let block =
            VisionTransformerBlock::new(ln1, qkv, proj, ln2, fc1, fc2, num_heads, hidden_dim);

        let final_ln = Some(candle_nn::layer_norm(hidden_dim, 1e-5, vb.pp("fln")).unwrap());

        let config = VisionConfig {
            image_size: image_size as u32,
            patch_size: patch_size as u32,
            vision_hidden_size: hidden_dim as u32,
            vision_num_layers: 1,
            vision_num_heads: num_heads as u32,
            projection_dim: hidden_dim as u32,
        };

        let encoder = VisionEncoder::new(patch_embed, vec![block], final_ln, config);
        // One row per patch: the CLS row is attention's alone (#310).
        assert_eq!(encoder.num_vision_tokens(), num_patches);
        assert_eq!(encoder.config().image_size, image_size as u32);
        let images = Tensor::zeros(&[1, 3, image_size, image_size], DType::F32, &device).unwrap();
        assert_eq!(
            encoder.forward(&images).unwrap().dims(),
            &[1, num_patches, hidden_dim]
        );
    }

    #[test]
    fn preprocess_gradient_pattern_image() {
        // Synthetic gradient image — verify preprocessing handles non-uniform content
        let size = 32;
        let mut rgb = Vec::with_capacity(3 * size * size);
        for y in 0..size {
            for x in 0..size {
                rgb.push(((x * 255) / size) as u8);
                rgb.push(((y * 255) / size) as u8);
                rgb.push((((x + y) * 127) / size) as u8);
            }
        }
        let img = ImageData {
            rgb_bytes: rgb,
            width: size as u32,
            height: size as u32,
        };
        let device = Device::Cpu;
        let tensor = preprocess_image(&img, &llava_square(16), &device).unwrap();
        assert_eq!(tensor.dims(), &[3, 16, 16]);

        // Verify channel statistics differ (gradient creates variation)
        let flat: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();
        let ch0: Vec<f32> = flat[..256].to_vec();
        let ch1: Vec<f32> = flat[256..512].to_vec();
        let mean0: f32 = ch0.iter().sum::<f32>() / ch0.len() as f32;
        let mean1: f32 = ch1.iter().sum::<f32>() / ch1.len() as f32;
        // Channels should have different means due to different gradient patterns
        assert!((mean0 - mean1).abs() > 0.01, "Channels should differ");
    }

    // ── Where an image sits in the prompt (#115 (2)) ──

    /// llama.cpp's joining rule: text parts on their own lines, an image's
    /// marker against its neighbours with no newline either side.
    #[test]
    fn content_is_joined_with_a_marker_where_each_image_was() {
        use ContentPiece::{Image, Text};
        assert_eq!(join_content([Text("a"), Text("b")]), "a\nb");
        assert_eq!(
            join_content([Image, Text("Describe it.")]),
            "<__media__>Describe it."
        );
        assert_eq!(
            join_content([Text("Compare"), Image, Image, Text("please")]),
            "Compare<__media__><__media__>please"
        );
        assert_eq!(join_content([Text("Look:"), Image]), "Look:<__media__>");
    }

    #[test]
    fn gemma3_frames_each_image_and_other_families_do_not() {
        let prompt = || "x<__media__>y".to_string();
        assert_eq!(
            wrap_images_for_family(prompt(), Some("gemma3")),
            "x<start_of_image><__media__><end_of_image>y"
        );
        assert_eq!(wrap_images_for_family(prompt(), Some("llama")), prompt());
        assert_eq!(wrap_images_for_family(prompt(), None), prompt());
    }

    /// A tiny SentencePiece vocabulary with Gemma 3's image frame as CONTROL
    /// tokens: `<bos>` 1, `<start_of_image>` 3, `<end_of_image>` 4, `a` 5, `b` 6.
    fn image_vocab() -> crate::inference::split::SplitTokenizer {
        let toks: Vec<String> = [
            "<unk>",
            "<bos>",
            "<eos>",
            "<start_of_image>",
            "<end_of_image>",
            "a",
            "b",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let scores = vec![0.0; toks.len()];
        let mut types = vec![1i32; toks.len()];
        for t in types.iter_mut().take(5) {
            *t = 3;
        }
        crate::inference::split::SplitTokenizer::from_sentencepiece(
            &toks,
            &scores,
            false,
            true,
            Some(1),
            &types,
        )
    }

    /// One BOS for the whole prompt, one -1 per marker, the frame matched
    /// whole on both sides of it.
    #[test]
    fn an_image_prompt_is_tokenized_with_one_id_per_marker_and_one_bos() {
        let tok = image_vocab();
        let framed = wrap_images_for_family("<bos>a<__media__>b".into(), Some("gemma3"));
        assert_eq!(
            encode_prompt_with_images(&tok, &framed),
            vec![1, 5, 3, IMAGE_TOKEN_ID, 4, 6]
        );
        // The BOS the tokenizer adds goes on the first piece only.
        assert_eq!(
            encode_prompt_with_images(&tok, "a<__media__>b<__media__>"),
            vec![1, 5, IMAGE_TOKEN_ID, 6, IMAGE_TOKEN_ID]
        );
    }

    #[test]
    fn an_image_prompts_positions_count_each_marker_as_its_rows() {
        let ids = [1, 5, IMAGE_TOKEN_ID, 6, IMAGE_TOKEN_ID];
        assert_eq!(positions_with_images(&ids, 512), 3 + 512);
        assert_eq!(positions_with_images(&[1, 5, 6], 0), 3);
    }

    /// Each image's rows land at its own marker, in order, and the text is
    /// embedded run by run around them.
    #[test]
    fn image_rows_are_spliced_at_their_markers() {
        let device = Device::Cpu;
        // Two images of two rows each: rows 10, 11 then 20, 21.
        let vision = Tensor::new(&[[10f32], [11.], [20.], [21.]], &device).unwrap();
        let ids = [1i64, 2, IMAGE_TOKEN_ID, 3, IMAGE_TOKEN_ID, 4];
        let mut runs = Vec::new();
        let merged = splice_image_rows(&ids, &vision, |run| {
            runs.push(run.to_vec());
            // "Embed" an id as itself, scaled, so the text is told apart.
            let vals: Vec<f32> = run.iter().map(|&i| i as f32 * 100.0).collect();
            Ok(Tensor::from_vec(vals, &[1, run.len(), 1], &device).unwrap())
        })
        .unwrap();
        assert_eq!(runs, vec![vec![1, 2], vec![3], vec![4]]);
        let flat: Vec<f32> = merged.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(flat, vec![100., 200., 10., 11., 300., 20., 21., 400.]);
        assert_eq!(merged.dims(), &[1, positions_with_images(&ids, 4), 1]);
    }

    /// The old fallback PREPENDED an image to a prompt that marked no place
    /// for it. Refused now — and so is a row count the markers do not divide.
    #[test]
    fn an_image_with_no_marked_place_is_refused_not_prepended() {
        let device = Device::Cpu;
        let vision = Tensor::zeros(&[3, 1], DType::F32, &device).unwrap();
        let embed =
            |run: &[i64]| Ok(Tensor::zeros(&[1, run.len(), 1], DType::F32, &Device::Cpu).unwrap());
        assert!(splice_image_rows(&[1, 2, 3], &vision, embed).is_err());
        assert!(splice_image_rows(&[1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID], &vision, embed).is_err());
    }

    /// llama.cpp's PAD_CEIL letterbox, computed in f32 as it is there: the
    /// 1200×600 test image becomes 896×448 at (0, 224).
    #[test]
    fn letterbox_geometry_matches_llama_cpp() {
        assert_eq!(letterbox_geometry(1200, 600, 896), (896, 448, 0, 224));
        assert_eq!(letterbox_geometry(600, 1200, 896), (448, 896, 224, 0));
        assert_eq!(letterbox_geometry(896, 896, 896), (896, 896, 0, 0));
        // A 3-pixel-wide sliver keeps at least one column.
        let (w, h, _, _) = letterbox_geometry(3, 5000, 896);
        assert!(w >= 1 && h == 896);
    }

    #[test]
    fn a_letterboxed_image_is_centred_on_black() {
        let img = ImageData {
            rgb_bytes: vec![255u8; 3 * 8 * 4],
            width: 8,
            height: 4,
        };
        for (pad, resample) in [
            ([0u8, 0, 0], Resample::Bilinear),
            (mean_as_bytes(CLIP_MEAN), Resample::Bicubic),
        ] {
            let pre = Preprocess {
                size: 8,
                mean: [0.0; 3],
                std: [1.0; 3],
                pad,
                resample,
            };
            let t = preprocess_image(&img, &pre, &Device::Cpu).unwrap();
            // 8×4 into 8×8: rows 2..6 white, the bands above and below `pad`.
            for c in 0..3 {
                let rows: Vec<Vec<f32>> = t.i(c).unwrap().to_vec2().unwrap();
                for (y, row) in rows.iter().enumerate() {
                    let expect = if (2..6).contains(&y) {
                        1.0
                    } else {
                        pad[c] as f32 / 255.0
                    };
                    assert!(
                        row.iter().all(|&v| (v - expect).abs() < 1e-6),
                        "pad {pad:?} channel {c} row {y}: {row:?}"
                    );
                }
            }
        }
    }

    /// LLaVA pads with its mean as llama.cpp does: `{122, 116, 104}` for CLIP's.
    #[test]
    fn llava_pads_with_the_mean_llama_cpp_names() {
        assert_eq!(mean_as_bytes(CLIP_MEAN), [122, 116, 104]);
    }

    // ── An mmproj's keys, read as llama.cpp reads them (#310) ──

    /// A tiny LLaVA-shaped mmproj: an 8-pixel square of 4-pixel patches (four
    /// patches and a CLS), width 4, three blocks, the legacy FFN naming, no
    /// `v.post_ln` unless asked — written to a file the loader opens.
    fn tiny_llava_mmproj(
        meta: &[(&str, candle_core::quantized::gguf_file::Value)],
        post_ln: bool,
    ) -> tempfile::NamedTempFile {
        use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
        let device = Device::Cpu;
        let (hidden, ff, blocks, llm) = (4usize, 8usize, 3usize, 6usize);
        let mut seed = 0u32;
        let mut t = |shape: &[usize]| {
            let n: usize = shape.iter().product();
            let vals: Vec<f32> = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    ((seed >> 16) % 200) as f32 / 1000.0 - 0.1
                })
                .collect();
            QTensor::quantize(
                &Tensor::from_vec(vals, shape, &device).unwrap(),
                GgmlDType::F32,
            )
            .unwrap()
        };
        let mut tensors: Vec<(String, QTensor)> = vec![
            ("v.patch_embd.weight".into(), t(&[hidden, 3, 4, 4])),
            ("v.class_embd".into(), t(&[hidden])),
            ("v.position_embd.weight".into(), t(&[5, hidden])),
            ("v.pre_ln.weight".into(), t(&[hidden])),
            ("v.pre_ln.bias".into(), t(&[hidden])),
            ("mm.0.weight".into(), t(&[llm, hidden])),
            ("mm.0.bias".into(), t(&[llm])),
            ("mm.2.weight".into(), t(&[llm, llm])),
            ("mm.2.bias".into(), t(&[llm])),
        ];
        if post_ln {
            tensors.push(("v.post_ln.weight".into(), t(&[hidden])));
            tensors.push(("v.post_ln.bias".into(), t(&[hidden])));
        }
        for i in 0..blocks {
            for (name, shape) in [
                ("ln1.weight", vec![hidden]),
                ("ln1.bias", vec![hidden]),
                ("ln2.weight", vec![hidden]),
                ("ln2.bias", vec![hidden]),
                ("attn_q.weight", vec![hidden, hidden]),
                ("attn_k.weight", vec![hidden, hidden]),
                ("attn_v.weight", vec![hidden, hidden]),
                ("attn_out.weight", vec![hidden, hidden]),
                // The legacy naming: ffn_down is the expansion.
                ("ffn_down.weight", vec![ff, hidden]),
                ("ffn_up.weight", vec![hidden, ff]),
            ] {
                tensors.push((format!("v.blk.{i}.{name}"), t(&shape)));
            }
        }
        use gguf_file::Value;
        let mut metadata: Vec<(&str, Value)> = vec![
            ("clip.vision.image_size", Value::U32(8)),
            ("clip.vision.patch_size", Value::U32(4)),
            ("clip.vision.embedding_length", Value::U32(hidden as u32)),
            ("clip.vision.attention.head_count", Value::U32(1)),
            ("clip.vision.block_count", Value::U32(blocks as u32)),
            ("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-5)),
        ];
        metadata.extend(meta.iter().cloned());
        let meta_refs: Vec<(&str, &Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
        let tensor_refs: Vec<(&str, &QTensor)> =
            tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let mut file = tempfile::NamedTempFile::new().unwrap();
        gguf_file::write(file.as_file_mut(), &meta_refs, &tensor_refs).unwrap();
        file
    }

    /// LLaVA's file names no activation, no feature layer and no post-norm:
    /// quick GELU, every block it holds, no norm after them, one row per patch,
    /// and a picture padded with the mean and resized bicubic. Before #310 it
    /// ran the erf GELU, a weight-1 norm, 577 rows and a bilinear stretch.
    #[test]
    fn a_llava_mmproj_is_read_as_llava_reads_it() {
        use candle_core::quantized::gguf_file::Value;
        let file = tiny_llava_mmproj(&[("clip.use_gelu", Value::Bool(false))], false);
        let module = load_from_mmproj_gguf(file.path(), &Device::Cpu).unwrap();
        assert_eq!(module.encoder.depth(), 3);
        assert!(module
            .encoder
            .blocks
            .iter()
            .all(|b| b.activation == EncoderActivation::GeluQuick));
        assert!(module.encoder.final_ln.is_none());
        assert_eq!(module.preprocess.pad, [122, 116, 104]);
        assert_eq!(module.preprocess.resample, Resample::Bicubic);
        assert_eq!(module.num_image_tokens(), 4);
        let image = ImageData {
            rgb_bytes: (0..3 * 12 * 6).map(|i| (i * 7 % 256) as u8).collect(),
            width: 12,
            height: 6,
        };
        let rows = module.encode_images(std::slice::from_ref(&image)).unwrap();
        assert_eq!(rows.dims(), &[4, 6]);
    }

    /// The keys llama.cpp reads, when a file sets them: `use_gelu` is the tanh
    /// GELU, a post-norm is applied, a single `feature_layer` is how many
    /// blocks run and several are refused.
    #[test]
    fn an_mmproj_s_activation_depth_and_post_norm_come_from_its_keys() {
        use candle_core::quantized::gguf_file::Value;
        let file = tiny_llava_mmproj(
            &[
                ("clip.use_gelu", Value::Bool(true)),
                (
                    "clip.vision.feature_layer",
                    Value::Array(vec![Value::I32(2)]),
                ),
            ],
            true,
        );
        let module = load_from_mmproj_gguf(file.path(), &Device::Cpu).unwrap();
        assert_eq!(module.encoder.depth(), 2);
        assert_eq!(module.encoder.config().vision_num_layers, 2);
        assert!(module
            .encoder
            .blocks
            .iter()
            .all(|b| b.activation == EncoderActivation::GeluTanh));
        assert!(module.encoder.final_ln.is_some());

        let stacked = tiny_llava_mmproj(
            &[(
                "clip.vision.feature_layer",
                Value::Array(vec![Value::I32(1), Value::I32(3)]),
            )],
            false,
        );
        assert!(matches!(
            load_from_mmproj_gguf(stacked.path(), &Device::Cpu),
            Err(SwarmError::Validation(_))
        ));
        let both = tiny_llava_mmproj(
            &[
                ("clip.use_gelu", Value::Bool(true)),
                ("clip.use_silu", Value::Bool(true)),
            ],
            false,
        );
        assert!(matches!(
            load_from_mmproj_gguf(both.path(), &Device::Cpu),
            Err(SwarmError::Validation(_))
        ));
    }

    #[test]
    fn a_llava_encoder_runs_every_block_its_file_holds() {
        use DepthRule::{LlamaCpp, Llava};
        use ProjectorKind as K;
        assert_eq!(encoder_depth(K::Llava, 23, None, Llava).unwrap(), 23);
        assert_eq!(encoder_depth(K::Llava, 23, None, LlamaCpp).unwrap(), 22);
        assert_eq!(encoder_depth(K::Llava, 23, Some(&[]), Llava).unwrap(), 23);
        for rule in [Llava, LlamaCpp] {
            assert_eq!(encoder_depth(K::Llava, 24, Some(&[23]), rule).unwrap(), 23);
            assert!(encoder_depth(K::Llava, 23, Some(&[24]), rule).is_err());
            assert!(encoder_depth(K::Llava, 23, Some(&[-1]), rule).is_err());
            assert!(encoder_depth(K::Llava, 24, Some(&[3, 7]), rule).is_err());
            assert_eq!(encoder_depth(K::Gemma3, 27, None, rule).unwrap(), 27);
        }
    }

    /// CLIP's quick GELU, `x · sigmoid(1.702 x)` — not the erf form.
    #[test]
    fn quick_gelu_is_clips() {
        let x = Tensor::new(&[-2.0f32, -0.5, 0.0, 1.0, 3.0], &Device::Cpu).unwrap();
        let got: Vec<f32> = EncoderActivation::GeluQuick
            .apply(&x)
            .unwrap()
            .to_vec1()
            .unwrap();
        for (v, g) in [-2.0f32, -0.5, 0.0, 1.0, 3.0].iter().zip(&got) {
            let want = v / (1.0 + (-1.702 * v).exp());
            assert!((g - want).abs() < 1e-6, "{v}: {g} vs {want}");
        }
        // The erf GELU of 1 is 0.8413; quick GELU's is 0.8458.
        assert!((got[3] - 0.8458).abs() < 1e-3, "{}", got[3]);
        assert_eq!(
            EncoderActivation::from_keys(false, false).unwrap(),
            EncoderActivation::GeluQuick
        );
        assert_eq!(
            EncoderActivation::from_keys(false, true).unwrap(),
            EncoderActivation::Silu
        );
    }

    /// Gemma 3's pool: token t is the mean of one pool×pool square of the
    /// row-major patch grid, then RMS-normed and projected.
    #[test]
    fn gemma3_pools_squares_of_the_patch_grid() {
        let device = Device::Cpu;
        // A 4×4 grid of two-channel patches, so the RMS norm is not trivial:
        // patch p carries (p, 1).
        let vals: Vec<f32> = (0..16).flat_map(|p| [p as f32, 1.0]).collect();
        let features = Tensor::from_vec(vals, &[1, 16, 2], &device).unwrap();
        let proj = Gemma3Projection::new(
            Tensor::ones(&[2], DType::F32, &device).unwrap(),
            Tensor::eye(2, DType::F32, &device).unwrap(),
            2,
            1e-6,
        );
        let out = proj.forward(&features, 4).unwrap();
        assert_eq!(out.dims(), &[1, 4, 2]);
        assert_eq!(proj.tokens_per_image(4), 4);
        // Token 0 = mean of patches 0, 1, 4, 5 = 2.5; token 3 = mean of
        // 10, 11, 14, 15 = 12.5 — each against a second channel of 1.
        let got: Vec<Vec<f32>> = out.i(0).unwrap().to_vec2().unwrap();
        let rms = |a: f32| ((a * a + 1.0) / 2.0 + 1e-6).sqrt();
        for (t, mean) in [(0usize, 2.5f32), (1, 4.5), (2, 10.5), (3, 12.5)] {
            assert!(
                (got[t][0] - mean / rms(mean)).abs() < 1e-4,
                "token {t}: {:?}",
                got[t]
            );
            assert!(
                (got[t][1] - 1.0 / rms(mean)).abs() < 1e-4,
                "token {t}: {:?}",
                got[t]
            );
        }
    }

    /// The wire payload round-trips, and one that does not describe its rows
    /// is refused rather than dropped.
    #[test]
    fn a_vision_payload_round_trips_and_a_bad_one_is_refused() {
        let encode = |rows: u32, hidden: u32, vals: &[f32]| {
            let raw: Vec<u8> = vals
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                .collect();
            let mut out = rows.to_le_bytes().to_vec();
            out.extend(hidden.to_le_bytes());
            out.extend(zstd::encode_all(std::io::Cursor::new(raw), 1).unwrap());
            out
        };
        let good = encode(2, 3, &[1., 2., 3., 4., 5., 6.]);
        assert_eq!(wire_rows(&good), Some(2));
        let t = decode_wire_embeddings(&good).unwrap();
        assert_eq!(
            t.to_vec2::<f32>().unwrap(),
            vec![vec![1., 2., 3.], vec![4., 5., 6.]]
        );

        assert!(decode_wire_embeddings(&encode(3, 3, &[1., 2., 3., 4., 5., 6.])).is_err());
        assert!(decode_wire_embeddings(&encode(1, 1, &[f32::NAN])).is_err());
        assert!(decode_wire_embeddings(&good[..6]).is_err());
        assert_eq!(wire_rows(&good[..3]), None);
    }
}
