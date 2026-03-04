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
            .matmul(&weight_t)
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
            config,
        }
    }

    /// Encode images into vision feature embeddings.
    ///
    /// Input: (B, 3, H, W) preprocessed image tensor
    /// Output: (B, num_patches + 1, vision_hidden_size) vision features
    pub fn forward(&self, images: &Tensor) -> Result<Tensor, SwarmError> {
        let mut x = self.patch_embed.forward(images)?;

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
}
