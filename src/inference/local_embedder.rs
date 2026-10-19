//! Local embedding privacy (`inference.local_embedding_privacy`): this node
//! turns the prompt's tokens into the first layer's input itself, so a first
//! segment on another computer is handed hidden states, never token ids.
//! ⚠ Those states are rows of the model's own embedding table, which that
//! segment holds and maps back to the tokens EXACTLY: this keeps token ids off
//! the wire, it does not hide the prompt from the computer that reads it.
//!
//! **The embedder is the first segment's own loader, asked for no layers**: a
//! processor-only `SplitModel` over this node's copy of the model
//! (`daemon::shard_loader::load_model_range` — `gguf_header.bin` and
//! `shard_000.bin` for a model held in parts), which holds the embedding table
//! — rows read out of the quantized table as they are looked up — and the
//! tokenizer. Its rows are the first segment's own (`SplitModel::embed_ids` →
//! `embed_tokens`), bit for bit, and it reads the parts through the same
//! checks (`first_header_disagreement`, the tensor table).
//!
//! Until 2026-10-20 it opened `shard_000.bin` as a whole GGUF file, which a
//! part is not — its first bytes are tensor data — so every load failed with
//! "unknown magic", the setting did nothing, and a first segment elsewhere was
//! sent the token ids it exists to keep here (FUTURE_WORK #316).

use std::path::Path;
use std::sync::Arc;

use candle_core::Tensor;

use crate::error::SwarmError;
use crate::inference::split::{tensor_to_bytes, SplitTokenizer};
use crate::types::ModelId;

pub struct LocalEmbedder {
    rows: Rows,
}

/// Where the rows come from.
enum Rows {
    /// The first segment's loader, asked for no layers.
    FirstSegment(Box<crate::inference::split::SplitModel>),
    /// A bare table, for the pipeline's tests (which have no model files).
    #[cfg(test)]
    Table(candle_nn::Embedding),
}

impl LocalEmbedder {
    /// Load this node's embedder for `model_id`: the model's first segment with
    /// no layers, on the processor (the daemon holds no graphics context, and
    /// a lookup is a few rows). Fails where this node does not hold the
    /// model's first part — the embedding table is in it.
    pub fn load(data_dir: &Path, model_id: &ModelId) -> Result<Self, SwarmError> {
        use crate::model::manifest::ModelManifestExt;
        let model_dir = crate::model::shard::model_dir(data_dir, &model_id.0);
        let manifest = crate::types::ModelManifest::load_from_dir(&model_dir)?;
        let shard_store = crate::model::shard::ShardStore::new(data_dir);
        let model = crate::daemon::shard_loader::load_model_range(
            &crate::daemon::shard_loader::ShardLoadParams {
                model_dir: &model_dir,
                shard_store: &shard_store,
                model_id,
                layer_start: 0,
                layer_end: 0,
                is_first: true,
                is_last: false,
                manifest: &manifest,
                force_cpu: true,
            },
        )?;
        tracing::info!(
            model = %model_id,
            hidden_dim = model.hidden_dim,
            has_tokenizer = model.tokenizer().is_some(),
            "Loaded this model's embedding table for prompt privacy"
        );
        Ok(Self {
            rows: Rows::FirstSegment(Box::new(model)),
        })
    }

    /// An embedder over a bare `[vocab, hidden]` table, with no tokenizer (so
    /// a prompt is read byte by byte) — for tests with no model files.
    #[cfg(test)]
    pub(crate) fn from_table(table: Tensor) -> Self {
        let hidden = table.dim(1).expect("a [vocab, hidden] table");
        Self {
            rows: Rows::Table(candle_nn::Embedding::new(table, hidden)),
        }
    }

    fn tokenizer(&self) -> Option<&SplitTokenizer> {
        match &self.rows {
            Rows::FirstSegment(model) => model.tokenizer(),
            #[cfg(test)]
            Rows::Table(_) => None,
        }
    }

    /// Embed a full prompt string into hidden-state activations.
    ///
    /// Returns serialized tensor bytes (via `tensor_to_bytes`) with shape
    /// `[1, positions, hidden_dim]` and the positions. The output is ready to
    /// be sent as `LayerForward.activations` with `pre_embedded = true`.
    ///
    /// `vision` is the request's image rows, REQUIRED so no caller can leave
    /// them out: before #115 (2) this path tokenized the text alone, and a
    /// request with a picture that took it was answered without the picture.
    /// With rows, the prompt is tokenized with its markers and the rows put
    /// there (`vision::splice_image_rows`, the first segment's own splice).
    pub fn embed_prompt(
        &self,
        prompt: &str,
        vision: Option<&Tensor>,
    ) -> Result<(Vec<u8>, usize), SwarmError> {
        let token_ids: Vec<i64> = match (self.tokenizer(), vision) {
            (Some(tokenizer), Some(_)) => {
                crate::inference::vision::encode_prompt_with_images(tokenizer, prompt)
            }
            (Some(tokenizer), None) => tokenizer.encode(prompt),
            (None, Some(_)) => {
                return Err(SwarmError::Inference(
                    "this computer cannot read the model's vocabulary, so it cannot place \
                     the picture in the prompt"
                        .into(),
                ))
            }
            // The first segment's own fallback (`model_worker`): byte-level.
            (None, None) => prompt.bytes().map(|b| b as i64).collect(),
        };

        let emb = match vision {
            Some(rows) => crate::inference::vision::splice_image_rows(&token_ids, rows, |run| {
                self.embed_ids(run)
            })?,
            None => self.embed_ids(&token_ids)?,
        };
        let positions = emb
            .dim(1)
            .map_err(|e| SwarmError::Internal(format!("Embedding dims: {e}")))?;
        let bytes = tensor_to_bytes(&emb)?;
        Ok((bytes, positions))
    }

    /// `(1, n, hidden)` embeddings of `ids`: the first segment's lookup.
    fn embed_ids(&self, ids: &[i64]) -> Result<Tensor, SwarmError> {
        match &self.rows {
            Rows::FirstSegment(model) => model.embed_ids(ids),
            #[cfg(test)]
            Rows::Table(table) => {
                use candle_core::Module;
                let input = Tensor::new(ids, &candle_core::Device::Cpu)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(SwarmError::internal)?;
                table.forward(&input).map_err(SwarmError::internal)
            }
        }
    }

    /// Embed a single token ID into hidden-state activations.
    ///
    /// Returns serialized tensor bytes with shape `[1, 1, hidden_dim]`.
    pub fn embed_token(&self, token_id: u32) -> Result<Vec<u8>, SwarmError> {
        tensor_to_bytes(&self.embed_ids(&[i64::from(token_id)])?)
    }
}

/// This node's embedder for `model_id`, loaded the first time a request asks
/// (on a blocking thread — it reads the header and the embedding table) and
/// kept in `SharedState::local_embedders` until the model is unloaded or
/// deleted. `Err` where it cannot be built here: the caller refuses rather
/// than send token ids (`PipelineExecutor::execute_distributed`).
///
/// Loaded on demand, not at startup as it was: a model fetched after the node
/// started never had one.
pub async fn local_embedder_for(
    state: &crate::daemon::SharedState,
    model_id: &ModelId,
) -> Result<Arc<LocalEmbedder>, SwarmError> {
    if let Some(found) = state.local_embedders.get(model_id) {
        return Ok(found.value().clone());
    }
    let data_dir = state.config.node.data_dir.clone();
    let id = model_id.clone();
    let loaded = tokio::task::spawn_blocking(move || LocalEmbedder::load(&data_dir, &id))
        .await
        .map_err(|e| SwarmError::Internal(format!("local embedder load panicked: {e}")))??;
    // Two requests may have loaded it at once; the first one kept is the one
    // every later request uses.
    Ok(state
        .local_embedders
        .entry(model_id.clone())
        .or_insert_with(|| Arc::new(loaded))
        .value()
        .clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Loads the embedder from a real model directory — the parts as a node
    /// holds them (`gguf_header.bin` + `shard_000.bin`), never a whole GGUF —
    /// and checks its rows are the first segment's own: the same ids looked up
    /// by the first segment loaded with one layer.
    ///
    /// Red before FUTURE_WORK #316: `unknown magic 0x3f800000` — the part was
    /// opened as a GGUF file.
    ///
    /// ```sh
    /// SWARMLLM_TEST_MODEL_DIR=~/.local/share/swarmllm/models/llama-3.2-3b-instruct-q4-k-m \
    ///     cargo dev-test --lib local_embedder_load_from_real_model -- --ignored
    /// ```
    #[test]
    #[ignore]
    fn local_embedder_load_from_real_model() {
        let model_dir = std::path::PathBuf::from(
            std::env::var("SWARMLLM_TEST_MODEL_DIR")
                .expect("set SWARMLLM_TEST_MODEL_DIR to a model directory holding shard_000.bin"),
        );
        assert!(model_dir.join("shard_000.bin").exists());
        assert!(
            !model_dir.join("model.gguf").exists() && !model_dir.join("source_path").exists(),
            "the directory must hold the model in parts, or this proves nothing"
        );
        let model_id = ModelId(model_dir.file_name().unwrap().to_string_lossy().into());
        let data_dir = model_dir.parent().unwrap().parent().unwrap();

        let embedder = LocalEmbedder::load(data_dir, &model_id).expect("the embedder loads");
        let (bytes, positions) = embedder
            .embed_prompt("Hello world", None)
            .expect("embed_prompt");
        assert!(positions > 0 && !bytes.is_empty());

        use crate::model::manifest::ModelManifestExt;
        let manifest = crate::types::ModelManifest::load_from_dir(&model_dir).unwrap();
        let first = crate::daemon::shard_loader::load_model_range(
            &crate::daemon::shard_loader::ShardLoadParams {
                model_dir: &model_dir,
                shard_store: &crate::model::shard::ShardStore::new(data_dir),
                model_id: &model_id,
                layer_start: 0,
                layer_end: 1,
                is_first: true,
                is_last: false,
                manifest: &manifest,
                force_cpu: true,
            },
        )
        .unwrap();
        let ids = first.tokenizer().unwrap().encode("Hello world");
        let own = tensor_to_bytes(&first.embed_ids(&ids).unwrap()).unwrap();
        assert_eq!(bytes, own, "the first segment's own rows, bit for bit");
        assert_eq!(
            embedder.embed_token(1).unwrap(),
            tensor_to_bytes(&first.embed_ids(&[1]).unwrap()).unwrap()
        );
    }
}
