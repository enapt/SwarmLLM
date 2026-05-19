//! SWARM-SPEC microbenchmark harness.
//!
//! Validates that each layer's primitive is fast enough to live in the
//! per-token decode hot path, and shows synthetic-workload behavior
//! that demonstrates the cascade is firing correctly.
//!
//! Run with:
//!   cargo run --release --no-default-features --features dev,claude-subscription \
//!     --example swarm_spec_bench
//!
//! No new dependencies (uses std::time only). End-to-end speedup
//! measurement (running a real model on a multi-node cluster) is a
//! separate harness — see docs/FUTURE_WORK.md § R136 validation plan.

use std::time::Instant;

use swarmllm::inference::hedging::{HedgeConfig, HedgeKey, HedgeTracker};
use swarmllm::inference::ngram_lookup::{
    cascade_find_candidate, find_candidate, NgramHitSource, NgramLookupConfig,
};
use swarmllm::inference::prefetch::{PrefetchConfig, PrefetchOrchestrator};
use swarmllm::inference::quant::{dequantize_q8_0, quantize_q8_0};
use swarmllm::network::pipeline_stream::chunk_layer_forward;
use swarmllm::network::protocol::{decode_layer_forward, encode_layer_forward};
use swarmllm::types::{ChunkAssemblyState, LayerForward, ModelId, NodeId, TensorFormat};

fn main() {
    println!("=== SWARM-SPEC microbenchmark harness ===\n");
    bench_q8_0_roundtrip();
    println!();
    bench_ngram_lookup();
    println!();
    bench_hedge_decision();
    println!();
    bench_prefetch_history();
    println!();
    bench_cascade_synthetic_workload();
    println!();
    bench_chunked_send();
    println!();
    println!("=== Done. See docs/FUTURE_WORK.md § R136 for end-to-end methodology. ===");
}

// ─── Layer 0: Q8_0 wire compression ────────────────────────────────────────

fn bench_q8_0_roundtrip() {
    println!("--- Layer 0: Q8_0 activation compression ---");

    // Hidden state for a typical model: 4096-dim, simulated as N(0,1).
    let hidden_dim = 4096;
    let mut seed: u64 = 0xc0ffee_d15ea5e;
    let mut rng = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5
    };
    let vals: Vec<f32> = (0..hidden_dim).map(|_| rng() * 2.0).collect();
    let raw_bytes = hidden_dim * 4;

    let iters = 1000;
    let t0 = Instant::now();
    let mut total_q_bytes = 0usize;
    for _ in 0..iters {
        let bytes = quantize_q8_0(&vals);
        total_q_bytes = bytes.len();
        std::hint::black_box(&bytes);
    }
    let q_dur = t0.elapsed();

    let bytes = quantize_q8_0(&vals);
    let t1 = Instant::now();
    for _ in 0..iters {
        let recovered = dequantize_q8_0(&bytes, hidden_dim).unwrap();
        std::hint::black_box(&recovered);
    }
    let dq_dur = t1.elapsed();

    let q_us_per_iter = q_dur.as_micros() as f64 / iters as f64;
    let dq_us_per_iter = dq_dur.as_micros() as f64 / iters as f64;
    let ratio = raw_bytes as f64 / total_q_bytes as f64;

    println!(
        "  Quantize:       {:.1} µs/iter ({} f32 → {} bytes Q8_0, {:.2}× compression)",
        q_us_per_iter, raw_bytes, total_q_bytes, ratio
    );
    println!("  Dequantize:    {:.1} µs/iter", dq_us_per_iter);
    println!(
        "  Round-trip:    {:.1} µs/iter — verdict: {}",
        q_us_per_iter + dq_us_per_iter,
        if q_us_per_iter + dq_us_per_iter < 1000.0 {
            "FAST (< 1ms per layer hop)"
        } else {
            "SLOW (review)"
        }
    );
}

// ─── Layer 1: N-gram lookup ────────────────────────────────────────────────

fn bench_ngram_lookup() {
    println!("--- Layer 1: N-gram prompt lookup ---");

    // Simulate a 2K-token prompt with a recent 100-token generation.
    let prompt_len = 2048;
    let gen_len = 100;
    let mut tokens: Vec<u32> = (0..prompt_len).map(|i| ((i * 7) % 32_000) as u32).collect();
    tokens.extend((0..gen_len).map(|i| ((i + prompt_len) * 11 % 32_000) as u32));

    let cfg = NgramLookupConfig::default();

    let iters = 10_000;
    let t0 = Instant::now();
    let mut hit_count = 0;
    for _ in 0..iters {
        let cand = find_candidate(&tokens, &tokens[..prompt_len], cfg);
        if !cand.is_empty() {
            hit_count += 1;
        }
        std::hint::black_box(cand);
    }
    let dur = t0.elapsed();
    let us_per_iter = dur.as_micros() as f64 / iters as f64;

    println!(
        "  find_candidate (2K-prompt + 100-gen): {:.2} µs/iter ({} hits in {} runs)",
        us_per_iter, hit_count, iters
    );

    // Cascade (prompt search → generation tail search)
    let t1 = Instant::now();
    let mut prompt_hits = 0;
    let mut gen_hits = 0;
    for _ in 0..iters {
        let (cand, source) = cascade_find_candidate(&tokens, prompt_len, 500, cfg);
        match source {
            NgramHitSource::Prompt => prompt_hits += 1,
            NgramHitSource::Generation => gen_hits += 1,
            NgramHitSource::None => {}
        }
        std::hint::black_box(cand);
    }
    let cas_dur = t1.elapsed();
    let cas_us_per_iter = cas_dur.as_micros() as f64 / iters as f64;
    println!(
        "  cascade_find_candidate: {:.2} µs/iter (prompt: {}, generation: {})",
        cas_us_per_iter, prompt_hits, gen_hits
    );
    let verdict = if cas_us_per_iter < 100.0 {
        "FAST (< 100µs — well below decode-token budget)"
    } else {
        "REVIEW (over 100µs adds noticeable overhead per spec round)"
    };
    println!("  Verdict: {}", verdict);
}

// ─── Layer 2: Hedge decision ───────────────────────────────────────────────

fn bench_hedge_decision() {
    println!("--- Layer 2: Hedge decision ---");

    let tracker = HedgeTracker::new();
    let model_id = ModelId("bench-model".into());
    let cfg = HedgeConfig {
        enabled: true,
        after_factor: 1.5,
        max_rate: 0.05,
        min_samples: 5,
    };

    // Populate observations across 100 distinct (segment, holder) keys.
    let n_keys = 100;
    for k in 0..n_keys {
        let key = HedgeKey {
            model_id: model_id.clone(),
            segment_idx: (k % 8) as u8,
            holder: NodeId([(k % 256) as u8; 32]),
        };
        for _ in 0..20 {
            tracker.observe(key.clone(), 100.0);
        }
    }

    let iters = 100_000;
    let t0 = Instant::now();
    let mut hedge_count = 0;
    for i in 0..iters {
        let key = HedgeKey {
            model_id: model_id.clone(),
            segment_idx: ((i % n_keys) % 8) as u8,
            holder: NodeId([((i % n_keys) % 256) as u8; 32]),
        };
        // Half below threshold (no hedge), half above (would hedge).
        let elapsed = if i % 2 == 0 { 100.0 } else { 250.0 };
        if tracker.should_hedge(&key, elapsed, cfg) {
            hedge_count += 1;
        }
    }
    let dur = t0.elapsed();
    let ns_per_iter = dur.as_nanos() as f64 / iters as f64;
    println!(
        "  should_hedge: {:.1} ns/iter ({} hedge decisions in {} runs)",
        ns_per_iter, hedge_count, iters
    );

    let t1 = Instant::now();
    for i in 0..iters {
        let key = HedgeKey {
            model_id: model_id.clone(),
            segment_idx: ((i % n_keys) % 8) as u8,
            holder: NodeId([((i % n_keys) % 256) as u8; 32]),
        };
        tracker.observe(key, 100.0 + (i as f32 % 10.0));
    }
    let dur = t1.elapsed();
    let obs_ns = dur.as_nanos() as f64 / iters as f64;
    println!("  observe: {:.1} ns/iter", obs_ns);
    println!(
        "  Verdict: {}",
        if obs_ns < 5000.0 {
            "FAST (< 5µs — negligible overhead per forward)"
        } else {
            "REVIEW (lock contention may be limiting throughput)"
        }
    );
}

// ─── Layer 3: Prefetch history ─────────────────────────────────────────────

fn bench_prefetch_history() {
    println!("--- Layer 3: Prefetch orchestrator ---");

    let orch = PrefetchOrchestrator::new();
    let cfg = PrefetchConfig {
        enabled: true,
        min_idle_ms: 0,
        min_turns_for_prediction: 1,
        max_candidates: 3,
        min_useful_rate: 0.0,
        min_dispatches_for_throttle: 1_000_000,
    };

    // Populate 1000 sessions with 10 turns each.
    let n_sessions = 1000;
    for s in 0..n_sessions {
        let session_id = format!("sess-{s:04}");
        for t in 0..10 {
            orch.observe_user_turn(&session_id, ((s + t) % 32_000) as u32);
        }
        orch.record_response_completion(&session_id, 0);
    }

    let iters = 100_000;
    let t0 = Instant::now();
    let mut candidate_count = 0;
    for i in 0..iters {
        let session_id = format!("sess-{:04}", i % n_sessions);
        let cands = orch.should_prefetch(&session_id, 10_000, cfg);
        candidate_count += cands.len();
    }
    let dur = t0.elapsed();
    let us_per_iter = dur.as_micros() as f64 / iters as f64;
    println!(
        "  should_prefetch: {:.2} µs/iter ({} candidates returned across {} runs)",
        us_per_iter, candidate_count, iters
    );

    let t1 = Instant::now();
    for i in 0..iters {
        let session_id = format!("sess-{:04}", i % n_sessions);
        orch.observe_user_turn(&session_id, (i % 32_000) as u32);
    }
    let dur = t1.elapsed();
    let obs_us = dur.as_micros() as f64 / iters as f64;
    println!("  observe_user_turn: {:.2} µs/iter", obs_us);

    let verdict = if us_per_iter < 50.0 {
        "FAST (< 50µs — fires every ~10s, negligible)"
    } else {
        "REVIEW (decision is heavier than expected)"
    };
    println!("  Verdict: {}", verdict);
}

// ─── Cascade synthetic workload ────────────────────────────────────────────

fn bench_cascade_synthetic_workload() {
    println!("--- Synthetic cascade hit-rate (Layer 1 → SWIFT/DSD fallback) ---");

    // Three synthetic workloads, decode 100 tokens each, measure how
    // often n-gram lookup would hit (saving a draft-model forward).

    let configs = vec![
        (
            "Code completion (high repeat)",
            make_code_completion_workload(),
        ),
        ("RAG / summarisation (input copy)", make_rag_workload()),
        ("Free-form chat (low repeat)", make_chat_workload()),
    ];

    for (name, (prompt_tokens, generated_tokens)) in configs {
        let mut full: Vec<u32> = prompt_tokens.clone();
        let cfg = NgramLookupConfig::default();
        let mut ngram_hits = 0;
        let mut fallback = 0;
        for tok in &generated_tokens {
            let (cand, source) = cascade_find_candidate(&full, prompt_tokens.len(), 500, cfg);
            if !cand.is_empty() && cand.contains(tok) {
                ngram_hits += 1;
            } else {
                fallback += 1;
            }
            // (Inject hit token regardless — we're measuring synthetic
            // hit rate, not running real inference.)
            std::hint::black_box(source);
            full.push(*tok);
        }
        let hit_rate = 100.0 * ngram_hits as f64 / generated_tokens.len() as f64;
        println!(
            "  {} → {:.1}% n-gram hit rate ({}/{} tokens) — {} fall through to SWIFT/DSD",
            name,
            hit_rate,
            ngram_hits,
            generated_tokens.len(),
            fallback
        );
    }
    println!(
        "\n  Each n-gram hit saves ~5-20ms of draft-model forward time. At\n  a 60% hit rate (typical code workload) on a 100-token response,\n  the layer saves ~300-1200ms — typically more than the wire-time\n  saved by Q8_0 compression."
    );
}

fn make_code_completion_workload() -> (Vec<u32>, Vec<u32>) {
    // Code repeats identifiers heavily. Simulate by re-using tokens
    // from a 20-token "vocabulary".
    let prompt: Vec<u32> = (0..200).map(|i| (i % 20) as u32 + 1000).collect();
    // Generation: same 20-token vocabulary in similar order patterns.
    let generated: Vec<u32> = (0..100).map(|i| ((i + 3) % 20) as u32 + 1000).collect();
    (prompt, generated)
}

fn make_rag_workload() -> (Vec<u32>, Vec<u32>) {
    // RAG: long prompt with chunks, output copies chunk contents.
    let mut prompt = Vec::new();
    for chunk in 0..10 {
        for offset in 0..50 {
            prompt.push((chunk * 100 + offset) as u32);
        }
    }
    // Generation: copies from chunk 7 verbatim.
    let generated: Vec<u32> = (0..100).map(|i| (700 + i % 50) as u32).collect();
    (prompt, generated)
}

fn make_chat_workload() -> (Vec<u32>, Vec<u32>) {
    // Free-form chat — low overlap with prompt.
    let prompt: Vec<u32> = (0..100).map(|i| (i * 13 + 5) as u32).collect();
    // Generation uses a different distribution.
    let generated: Vec<u32> = (0..100).map(|i| (i * 17 + 1000) as u32).collect();
    (prompt, generated)
}

// ─── Tier 4K: chunked vs monolithic activation transport ────────────────────
//
// Measures the CPU overhead the chunked-send path adds vs the monolithic
// path at the activation sizes that matter for default config:
//   - 32 KiB:    below the streaming_min_activation_bytes=64 KiB floor;
//                chunk_layer_forward returns the input verbatim. Zero
//                chunk-meta cost.
//   - 64 KiB:    exactly the floor — one chunk, also passthrough.
//   - 256 KiB:   exactly streaming_chunk_size_bytes default — one chunk.
//   - 1 MiB:     four chunks, the typical K=2-4 sweet spot.
//   - 1.6 MiB:   prefill-class. ~7 chunks. Stresses the assembly path.
//
// Wire transit cost is identical between paths; only the serialise +
// split + reassemble CPU work differs. WAN measurement (where chunking
// recovers latency via encrypt/decrypt overlap) needs a real-network
// harness — see docs/FUTURE_WORK.md § Tier 4K.
fn bench_chunked_send() {
    println!("--- Tier 4K: chunked vs monolithic activation transport ---");

    const CHUNK_SIZE: usize = 256 * 1024;
    let sizes: &[(&str, usize)] = &[
        ("32 KiB ", 32 * 1024),
        ("64 KiB ", 64 * 1024),
        ("256 KiB", 256 * 1024),
        ("1 MiB  ", 1024 * 1024),
        ("1.6 MiB", 1600 * 1024),
    ];

    println!(
        "  {:<8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>6}",
        "size", "mono µs", "chunk µs", "split µs", "asm µs", "K"
    );

    for (label, size) in sizes {
        let activations: Vec<u8> = (0..*size).map(|i| (i & 0xFF) as u8).collect();
        let forward = make_bench_forward(activations);

        // Monolithic: encode → decode roundtrip.
        let iters = if *size < 256 * 1024 { 500 } else { 100 };
        let t0 = Instant::now();
        for _ in 0..iters {
            let encoded = encode_layer_forward(&forward).unwrap();
            let decoded = decode_layer_forward(&encoded).unwrap();
            std::hint::black_box(decoded);
        }
        let mono_us = t0.elapsed().as_micros() as f64 / iters as f64;

        // Chunked split cost only.
        let t_split = Instant::now();
        let mut k = 0usize;
        for _ in 0..iters {
            let chunks = chunk_layer_forward(&forward, CHUNK_SIZE);
            k = chunks.len();
            std::hint::black_box(chunks);
        }
        let split_us = t_split.elapsed().as_micros() as f64 / iters as f64;

        // Full chunked roundtrip. Mirrors production dispatch: when
        // chunk_layer_forward returns K==1 (size below or at chunk_size),
        // the helper sets chunk_meta=None and the receive path skips
        // assembly entirely (see pipeline_stream.rs:365 + tensors.rs:598).
        // So we only pay the assembly cost on the multi-chunk case.
        let t1 = Instant::now();
        let mut asm_us_acc = 0u128;
        for _ in 0..iters {
            let chunks = chunk_layer_forward(&forward, CHUNK_SIZE);
            let total_chunks = chunks.len() as u32;
            let encoded: Vec<Vec<u8>> = chunks
                .iter()
                .map(|c| encode_layer_forward(c).unwrap())
                .collect();
            let decoded: Vec<LayerForward> = encoded
                .iter()
                .map(|bytes| decode_layer_forward(bytes).unwrap())
                .collect();

            if total_chunks <= 1 {
                std::hint::black_box(&decoded);
                continue;
            }

            // Assembly: mimic SharedState::try_assemble_chunked_forward
            // without going through DashMap (the slot-table contention is
            // not what we're measuring here — the CPU work is).
            let t_asm = Instant::now();
            let mut template = decoded[0].clone();
            template.activations = Vec::new();
            template.chunk_meta = None;
            let mut state = ChunkAssemblyState::new(total_chunks, template, vec![1, 2, 3]);
            for d in &decoded {
                let cm = d
                    .chunk_meta
                    .expect("multi-chunk decoded forwards must carry chunk_meta");
                state.received[cm.chunk_idx as usize] = Some(d.activations.clone());
                state.filled += 1;
            }
            let assembled = state.assemble();
            asm_us_acc += t_asm.elapsed().as_micros();
            std::hint::black_box(assembled);
        }
        let chunked_us = t1.elapsed().as_micros() as f64 / iters as f64;
        let asm_us = if k > 1 {
            asm_us_acc as f64 / iters as f64
        } else {
            0.0
        };

        println!(
            "  {}  {:>10.1}  {:>10.1}  {:>10.1}  {:>10.1}  {:>6}",
            label, mono_us, chunked_us, split_us, asm_us, k
        );
    }
    println!(
        "\n  Reading: 'mono µs' is encode+decode of one LayerForward. 'chunk µs'\n  is the full split→encode×K→decode×K→assemble path. 'K' is the chunk\n  count at the default 256 KiB chunk size. The delta (chunk-mono) is the\n  CPU overhead the chunked path adds over monolithic on this host;\n  the WAN win comes from encrypt/decrypt and send/receive overlap that\n  this microbench does NOT capture."
    );
}

fn make_bench_forward(activations: Vec<u8>) -> LayerForward {
    LayerForward {
        request_id: uuid::Uuid::from_u128(0xDEAD_BEEF_CAFE_F00D_0011_2233_4455_6677),
        sequence_num: 0,
        index_pos: 0,
        activations,
        format: TensorFormat::FP32,
        model_id: ModelId("bench-model".into()),
        layer_range: (0, 16),
        tp_meta: None,
        vision_embeddings: None,
        sender_peer_bytes: None,
        requester_node_id: None,
        pre_embedded: false,
        generated_ids: Vec::new(),
        adapter_id: None,
        draft_tokens: Vec::new(),
        spec_logits_requested: false,
        truncate_kv_to: None,
        chunk_meta: None,
    }
}
