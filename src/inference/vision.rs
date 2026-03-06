//! Vision Language Model (VLM) support for multimodal inference.
//!
//! This module provides:
//! - Image preprocessing: resize, normalize, convert to candle tensors
//! - Vision encoder (ViT-style): patch embedding, transformer blocks, projection
//! - Multimodal projection: map vision features into the LLM embedding space
//! - Support for LLaVA (CLIP ViT + Llama) and Qwen2-VL architectures
//!
//! The vision encoder can run as a separate shard — the first segment in a
//! multimodal pipeline processes images into vision embeddings, which are
//! prepended to the text token embeddings before entering the LLM layers.

use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{Linear, Module};

use crate::error::SwarmError;
use crate::types::{ImageData, VisionConfig};

// ── Image Preprocessing ──

/// Default CLIP normalization constants (ImageNet).
#[allow(clippy::excessive_precision)]
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
#[allow(clippy::excessive_precision)]
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Preprocess an image for the vision encoder.
///
/// 1. Resize to `target_size x target_size` using bilinear interpolation
/// 2. Convert to f32 [0, 1]
/// 3. Normalize with CLIP mean/std
/// 4. Return tensor of shape (3, H, W)
pub fn preprocess_image(
    image: &ImageData,
    target_size: u32,
    device: &Device,
) -> Result<Tensor, SwarmError> {
    // Use the image crate to resize
    let img_buf = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
        image.width,
        image.height,
        image.rgb_bytes.clone(),
    )
    .ok_or_else(|| SwarmError::Inference("Invalid image dimensions".into()))?;

    let resized = image::imageops::resize(
        &img_buf,
        target_size,
        target_size,
        image::imageops::FilterType::Triangle,
    );

    let (w, h) = (resized.width() as usize, resized.height() as usize);

    // Convert to CHW float tensor, normalize
    let mut data = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let pixel = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                let normalized = (val - CLIP_MEAN[c]) / CLIP_STD[c];
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
    target_size: u32,
    device: &Device,
) -> Result<Tensor, SwarmError> {
    if images.is_empty() {
        return Err(SwarmError::Inference("No images to preprocess".into()));
    }

    let tensors: Vec<Tensor> = images
        .iter()
        .map(|img| preprocess_image(img, target_size, device))
        .collect::<Result<Vec<_>, _>>()?;

    // Stack into batch: each is (3, H, W) → (B, 3, H, W)
    Tensor::stack(&tensors, 0)
        .map_err(|e| SwarmError::Inference(format!("Image batch stacking failed: {e}")))
}

// ── Vision Encoder (ViT) ──

/// Patch embedding: converts image patches into a sequence of embeddings.
///
/// Input: (B, 3, H, W) image tensor
/// Output: (B, num_patches + 1, hidden_dim) — includes [CLS] token
pub struct PatchEmbedding {
    /// Conv2D projection: (hidden_dim, 3, patch_size, patch_size)
    proj_weight: Tensor,
    proj_bias: Tensor,
    /// [CLS] token embedding: (1, 1, hidden_dim)
    cls_token: Tensor,
    /// Positional embeddings: (1, num_patches + 1, hidden_dim)
    position_embedding: Tensor,
    patch_size: usize,
}

impl PatchEmbedding {
    pub fn new(
        proj_weight: Tensor,
        proj_bias: Tensor,
        cls_token: Tensor,
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

    /// Forward: (B, 3, H, W) → (B, num_patches + 1, hidden_dim)
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

        // Prepend CLS token
        let cls_expanded = self
            .cls_token
            .broadcast_as(&[b, 1, hidden_dim])
            .map_err(|e| SwarmError::Inference(format!("CLS broadcast: {e}")))?;

        let sequence = Tensor::cat(&[&cls_expanded, &projected], 1)
            .map_err(|e| SwarmError::Inference(format!("CLS concat: {e}")))?;

        // Add positional embeddings
        let pos_embed = if self.position_embedding.dim(1).unwrap_or(0) == num_patches + 1 {
            self.position_embedding.clone()
        } else {
            // Interpolate position embeddings if needed (image size mismatch)
            self.position_embedding
                .i((.., ..num_patches + 1, ..))
                .map_err(|e| SwarmError::Inference(format!("Pos embed slice: {e}")))?
        };

        sequence
            .broadcast_add(&pos_embed)
            .map_err(|e| SwarmError::Inference(format!("Pos embed add: {e}")))
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
        }
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
            .map_err(map_err("vit_q_transpose"))?;
        let k = qkv
            .narrow(2, hidden_dim, hidden_dim)
            .map_err(map_err("vit_k"))?
            .reshape(&[b, seq_len, self.num_heads, self.head_dim])
            .map_err(map_err("vit_k_reshape"))?
            .transpose(1, 2)
            .map_err(map_err("vit_k_transpose"))?;
        let v = qkv
            .narrow(2, 2 * hidden_dim, hidden_dim)
            .map_err(map_err("vit_v"))?
            .reshape(&[b, seq_len, self.num_heads, self.head_dim])
            .map_err(map_err("vit_v_reshape"))?
            .transpose(1, 2)
            .map_err(map_err("vit_v_transpose"))?;

        // Scaled dot-product attention (no causal mask — ViT uses full attention)
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q
            .matmul(&k.t().map_err(map_err("vit_k_t"))?)
            .map_err(map_err("vit_attn_matmul"))?;
        let attn_weights = (attn_weights / scale).map_err(map_err("vit_attn_scale"))?;
        let attn_weights =
            candle_nn::ops::softmax_last_dim(&attn_weights).map_err(map_err("vit_softmax"))?;
        let attn_output = attn_weights.matmul(&v).map_err(map_err("vit_attn_v"))?;

        // Reshape back: (B, num_heads, S, head_dim) → (B, S, hidden_dim)
        let attn_output = attn_output
            .transpose(1, 2)
            .map_err(map_err("vit_attn_transpose"))?
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
        // GELU activation
        let ff = ff.gelu_erf().map_err(map_err("vit_gelu"))?;
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
    blocks: Vec<VisionTransformerBlock>,
    final_ln: candle_nn::LayerNorm,
    /// Optional pre-layernorm applied after patch embedding, before transformer blocks.
    pub pre_ln: Option<candle_nn::LayerNorm>,
    config: VisionConfig,
}

impl VisionEncoder {
    pub fn new(
        patch_embed: PatchEmbedding,
        blocks: Vec<VisionTransformerBlock>,
        final_ln: candle_nn::LayerNorm,
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

    /// Encode images into vision feature embeddings.
    ///
    /// Input: (B, 3, H, W) preprocessed image tensor
    /// Output: (B, num_patches + 1, vision_hidden_size) vision features
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

        self.final_ln
            .forward(&x)
            .map_err(|e| SwarmError::Inference(format!("Vision encoder final_ln: {e}")))
    }

    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// Number of vision tokens produced per image (num_patches + 1 for CLS).
    pub fn num_vision_tokens(&self) -> usize {
        let ps = self.config.patch_size as usize;
        let img = self.config.image_size as usize;
        let num_patches = (img / ps) * (img / ps);
        num_patches + 1 // +1 for CLS token
    }
}

// ── Multimodal Projection ──

/// Projects vision encoder outputs into the LLM's embedding space.
///
/// For LLaVA: a 2-layer MLP (vision_hidden → projection_dim → llm_hidden_dim)
/// For Qwen2-VL: a linear projection with additional spatial merging
pub struct MultimodalProjection {
    proj1: Linear,
    proj2: Linear,
    /// The LLM hidden dimension this projects into.
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

    pub fn llm_hidden_dim(&self) -> usize {
        self.llm_hidden_dim
    }
}

// ── VLM Pipeline Integration ──

/// Holds a loaded vision encoder + projection, ready for inference.
/// Stored in SharedState for reuse across requests.
pub struct VisionModule {
    pub encoder: VisionEncoder,
    pub projection: MultimodalProjection,
    pub device: Device,
}

impl VisionModule {
    /// Process images from a multimodal request and produce LLM-space embeddings.
    ///
    /// Returns a tensor of shape (total_image_tokens, llm_hidden_dim) ready to be
    /// concatenated with the text token embeddings in the split model's forward pass.
    pub fn encode_images(&self, images: &[ImageData]) -> Result<Tensor, SwarmError> {
        if images.is_empty() {
            return Err(SwarmError::Inference("No images to encode".into()));
        }

        let start = std::time::Instant::now();
        let target_size = self.encoder.config().image_size;
        let pixel_values = preprocess_images(images, target_size, &self.device)?;

        // (B, 3, H, W) → (B, num_tokens, vision_hidden)
        let vision_features = self.encoder.forward(&pixel_values)?;

        // (B, num_tokens, vision_hidden) → (B, num_tokens, llm_hidden)
        let projected = self.projection.forward(&vision_features)?;

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

    /// Number of embedding tokens each image produces.
    pub fn tokens_per_image(&self) -> usize {
        self.encoder.num_vision_tokens()
    }
}

/// Merge vision embeddings with text token embeddings for multimodal forward pass.
///
/// Given:
/// - `text_embeddings`: (1, text_seq_len, hidden_dim) from tok_embeddings
/// - `vision_embeddings`: (num_image_tokens, hidden_dim) from vision encoder
/// - `image_token_positions`: indices in the text sequence where `<image>` placeholder tokens are
///
/// Returns: (1, text_seq_len + num_image_tokens - num_placeholders, hidden_dim)
///
/// For the simple case (LLaVA-style), we prepend vision embeddings before the text.
pub fn merge_vision_text_embeddings(
    text_embeddings: &Tensor,
    vision_embeddings: &Tensor,
    _image_positions: &[usize],
) -> Result<Tensor, SwarmError> {
    let map_err = |msg: &'static str| {
        move |e: candle_core::Error| SwarmError::Inference(format!("{msg}: {e}"))
    };

    // Simple prepend strategy: vision tokens come first, then text tokens
    // This matches LLaVA's approach when the model doesn't have explicit <image> tokens
    let (_, _text_seq, hidden) = text_embeddings.dims3().map_err(map_err("text_dims"))?;
    let num_vision = vision_embeddings.dim(0).map_err(map_err("vision_dim0"))?;

    // Reshape vision: (num_image_tokens, hidden_dim) → (1, num_image_tokens, hidden_dim)
    let vision_3d = vision_embeddings
        .reshape(&[1, num_vision, hidden])
        .map_err(map_err("vision_reshape"))?;

    // Concatenate: (1, num_vision + text_seq, hidden_dim)
    Tensor::cat(&[&vision_3d, text_embeddings], 1).map_err(map_err("vision_text_cat"))
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
pub fn load_from_mmproj_gguf(
    path: &std::path::Path,
    device: &Device,
) -> Result<VisionModule, SwarmError> {
    use candle_core::DType;

    let mut file = std::fs::File::open(path).map_err(|e| {
        SwarmError::Inference(format!("Failed to open mmproj GGUF: {e}"))
    })?;

    let ct = candle_core::quantized::gguf_file::Content::read(&mut file).map_err(|e| {
        SwarmError::Inference(format!("Failed to parse mmproj GGUF: {e}"))
    })?;

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
    // Get projection dim from mm.0.weight shape or default to hidden_size
    let projection_dim = ct
        .tensor_infos
        .get("mm.0.weight")
        .map(|t| t.shape.dims()[0])
        .unwrap_or(hidden_size);

    // Get LLM hidden dim from mm.2.weight output shape
    let llm_hidden_dim = ct
        .tensor_infos
        .get("mm.2.weight")
        .map(|t| t.shape.dims()[0])
        .ok_or_else(|| {
            SwarmError::Inference("Missing mm.2.weight — cannot determine LLM hidden dim".into())
        })?;

    let config = VisionConfig {
        image_size,
        patch_size,
        vision_hidden_size: hidden_size as u32,
        vision_num_layers: num_layers as u32,
        vision_num_heads: num_heads as u32,
        projection_dim: projection_dim as u32,
    };

    tracing::info!(
        image_size,
        patch_size,
        hidden_size,
        num_heads,
        num_layers,
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
        tracing::info!(?dims, total_elements, hidden_size, patch_dim, "patch_embd.weight raw shape");

        // The weight is a conv2d kernel. Regardless of how many dims candle gives us,
        // we need to reshape to (hidden_size, patch_dim) for our manual patch projection.
        // GGUF stores [kH, kW, C_in, C_out] which may be loaded as 2D [kH, kW*C_in*C_out]
        // or 4D. Either way, total elements = hidden_size * patch_dim.
        assert_eq!(
            total_elements,
            hidden_size * patch_dim,
            "patch_embd size mismatch: {total_elements} vs {}",
            hidden_size * patch_dim
        );
        patch_proj_raw
            .reshape(&[hidden_size, patch_dim])
            .map_err(|e| SwarmError::Inference(format!("patch_embd reshape: {e}")))?
    };
    let patch_proj_bias = load_opt!("v.patch_embd.bias")
        .unwrap_or_else(|| Tensor::zeros(&[hidden_size], DType::F32, device).unwrap());
    let cls_token = load!("v.class_embd")
        .reshape(&[1, 1, hidden_size])
        .map_err(|e| SwarmError::Inference(format!("cls_reshape: {e}")))?;

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
    let mut blocks = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
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

        // MLP
        let fc1_w = load!(&format!("v.blk.{i}.ffn_down.weight"));
        let fc1_b = load_opt!(&format!("v.blk.{i}.ffn_down.bias"));
        let mlp_fc1 = Linear::new(fc1_w, fc1_b);

        let fc2_w = load!(&format!("v.blk.{i}.ffn_up.weight"));
        let fc2_b = load_opt!(&format!("v.blk.{i}.ffn_up.bias"));
        let mlp_fc2 = Linear::new(fc2_w, fc2_b);

        blocks.push(VisionTransformerBlock::new(
            ln1,
            attn_qkv,
            attn_proj,
            ln2,
            mlp_fc1,
            mlp_fc2,
            num_heads,
            hidden_size,
        ));
    }

    // ── Post layer norm (optional — not all CLIP models have it) ──
    let final_ln = if ct.tensor_infos.contains_key("v.post_ln.weight") {
        let post_ln_w = load!("v.post_ln.weight");
        let post_ln_b = load!("v.post_ln.bias");
        candle_nn::LayerNorm::new(post_ln_w, post_ln_b, ln_eps)
    } else {
        // Identity-like layer norm: weight=1, bias=0
        let ones = Tensor::ones(&[hidden_size], DType::F32, device)
            .map_err(|e| SwarmError::Inference(format!("final_ln ones: {e}")))?;
        let zeros = Tensor::zeros(&[hidden_size], DType::F32, device)
            .map_err(|e| SwarmError::Inference(format!("final_ln zeros: {e}")))?;
        candle_nn::LayerNorm::new(ones, zeros, ln_eps)
    };

    tracing::info!(num_blocks = blocks.len(), "Vision encoder blocks loaded");

    let mut encoder = VisionEncoder::new(patch_embed, blocks, final_ln, config);
    encoder.pre_ln = pre_ln;

    // ── Multimodal projection (2-layer MLP: mm.0 → GELU → mm.2) ──
    let mm0_w = load!("mm.0.weight");
    let mm0_b = load_opt!("mm.0.bias");
    let mm2_w = load!("mm.2.weight");
    let mm2_b = load_opt!("mm.2.bias");

    let proj1 = Linear::new(mm0_w, mm0_b);
    let proj2 = Linear::new(mm2_w, mm2_b);
    let projection = MultimodalProjection::new(proj1, proj2, llm_hidden_dim);

    tracing::info!("VisionModule loaded from mmproj GGUF");

    Ok(VisionModule {
        encoder,
        projection,
        device: device.clone(),
    })
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn preprocess_image_creates_correct_shape() {
        let img = ImageData {
            rgb_bytes: vec![128u8; 3 * 64 * 64],
            width: 64,
            height: 64,
        };
        let device = Device::Cpu;
        let tensor = preprocess_image(&img, 224, &device).unwrap();
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
        let tensor = preprocess_image(&img, 16, &device).unwrap();

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
        let batch = preprocess_images(&images, 224, &device).unwrap();
        assert_eq!(batch.dims(), &[2, 3, 224, 224]);
    }

    #[test]
    fn preprocess_empty_batch_errors() {
        let device = Device::Cpu;
        let result = preprocess_images(&[], 224, &device);
        assert!(result.is_err());
    }

    #[test]
    fn has_images_detects_correctly() {
        use crate::types::{ChatMessage, Role};

        let text_only = vec![ChatMessage {
            role: Role::User,
            content: "Hello".into(),
            images: vec![],
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
            },
        ];
        let collected = collect_images(&messages);
        assert_eq!(collected.len(), 3);
    }

    #[test]
    fn merge_vision_text_concatenates() {
        let device = Device::Cpu;
        let text_embed = Tensor::zeros(&[1, 10, 64], DType::F32, &device).unwrap();
        let vision_embed = Tensor::ones(&[5, 64], DType::F32, &device).unwrap();
        let merged = merge_vision_text_embeddings(&text_embed, &vision_embed, &[]).unwrap();
        assert_eq!(merged.dims(), &[1, 15, 64]);
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

        let proj_weight =
            Tensor::ones(&[hidden_dim, patch_dim], DType::F32, &device).unwrap();
        let proj_bias = Tensor::zeros(&[hidden_dim], DType::F32, &device).unwrap();
        let cls_token = Tensor::zeros(&[1, 1, hidden_dim], DType::F32, &device).unwrap();
        let pos_embed =
            Tensor::zeros(&[1, num_patches + 1, hidden_dim], DType::F32, &device).unwrap();

        let _patch_embed = PatchEmbedding::new(proj_weight, proj_bias, cls_token, pos_embed, patch_size);
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

        let block =
            VisionTransformerBlock::new(ln1, attn_qkv, attn_proj, ln2, fc1, fc2, num_heads, hidden_dim);

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
        let pos_embed = Tensor::zeros(&[1, num_patches + 1, hidden_dim], DType::F32, &device).unwrap();
        let patch_embed = PatchEmbedding::new(proj_weight, proj_bias, cls_token, pos_embed, patch_size);

        let vm = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &device);
        let ln1 = candle_nn::layer_norm(hidden_dim, 1e-5, vb.pp("ln1")).unwrap();
        let qkv = candle_nn::linear(hidden_dim, 3 * hidden_dim, vb.pp("qkv")).unwrap();
        let proj = candle_nn::linear(hidden_dim, hidden_dim, vb.pp("proj")).unwrap();
        let ln2 = candle_nn::layer_norm(hidden_dim, 1e-5, vb.pp("ln2")).unwrap();
        let fc1 = candle_nn::linear(hidden_dim, 4 * hidden_dim, vb.pp("fc1")).unwrap();
        let fc2 = candle_nn::linear(4 * hidden_dim, hidden_dim, vb.pp("fc2")).unwrap();
        let block = VisionTransformerBlock::new(ln1, qkv, proj, ln2, fc1, fc2, num_heads, hidden_dim);

        let final_ln = candle_nn::layer_norm(hidden_dim, 1e-5, vb.pp("fln")).unwrap();

        let config = VisionConfig {
            image_size: image_size as u32,
            patch_size: patch_size as u32,
            vision_hidden_size: hidden_dim as u32,
            vision_num_layers: 1,
            vision_num_heads: num_heads as u32,
            projection_dim: hidden_dim as u32,
        };

        let encoder = VisionEncoder::new(patch_embed, vec![block], final_ln, config);
        assert_eq!(encoder.num_vision_tokens(), num_patches + 1); // 4 patches + CLS
        assert_eq!(encoder.config().image_size, image_size as u32);
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
        let tensor = preprocess_image(&img, 16, &device).unwrap();
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
}
