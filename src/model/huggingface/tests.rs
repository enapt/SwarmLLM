use super::shards::{coalesce_byte_ranges, parse_http_date, tensor_slices_in_chunk};
use super::*;
use std::time::SystemTime;

#[test]
fn download_url_format() {
    let url = download_url("TheBloke/Llama-2-7B-GGUF", "llama-2-7b.Q4_K_M.gguf").unwrap();
    assert_eq!(
        url,
        "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf"
    );
}

#[test]
fn download_url_rejects_path_traversal_filename() {
    assert!(download_url("org/name", "../../etc/passwd").is_err());
    assert!(download_url("org/name", "%2F%2Fevil.gguf").is_err());
    assert!(download_url("org/name", "evil%5Cwindows.gguf").is_err());
    assert!(download_url("org/name", "name\0null.gguf").is_err());
    assert!(download_url("org/name", "/abs/path.gguf").is_err());
    assert!(download_url("org/name", "").is_err());
}

#[test]
fn urlencoding_basic() {
    assert_eq!(urlencoding::encode("hello world"), "hello%20world");
    assert_eq!(urlencoding::encode("foo+bar"), "foo%2Bbar");
    assert_eq!(urlencoding::encode("simple"), "simple");
}

#[test]
fn coalesce_empty() {
    assert!(coalesce_byte_ranges(&[], 0).is_empty());
}

#[test]
fn coalesce_no_merge() {
    // Two ranges far apart
    let ranges = vec![(0, 100), (1000, 100)];
    let merged = coalesce_byte_ranges(&ranges, 50);
    assert_eq!(merged, vec![(0, 100), (1000, 1100)]);
}

#[test]
fn coalesce_adjacent() {
    // Two adjacent ranges
    let ranges = vec![(0, 100), (100, 100)];
    let merged = coalesce_byte_ranges(&ranges, 0);
    assert_eq!(merged, vec![(0, 200)]);
}

#[test]
fn coalesce_with_gap() {
    // Two ranges with a small gap within max_gap
    let ranges = vec![(0, 100), (110, 100)];
    let merged = coalesce_byte_ranges(&ranges, 20);
    assert_eq!(merged, vec![(0, 210)]);

    // Same gap but max_gap is too small
    let merged = coalesce_byte_ranges(&ranges, 5);
    assert_eq!(merged, vec![(0, 100), (110, 210)]);
}

#[test]
fn coalesce_unsorted() {
    // Input not sorted — should still work
    let ranges = vec![(200, 50), (0, 100), (100, 100)];
    let merged = coalesce_byte_ranges(&ranges, 0);
    assert_eq!(merged, vec![(0, 250)]);
}

#[test]
fn parse_retry_after_delta_seconds() {
    assert_eq!(parse_retry_after("120"), Some(120));
    assert_eq!(parse_retry_after("0"), Some(0));
    assert_eq!(parse_retry_after("1"), Some(1));
    assert_eq!(parse_retry_after(" 60 "), Some(60));
}

#[test]
fn parse_retry_after_http_date() {
    // Use a date far in the future to ensure it's always > now
    let result = parse_retry_after("Fri, 01 Jan 2100 00:00:00 GMT");
    assert!(result.is_some());
    let secs = result.unwrap();
    // Should be many years in the future
    assert!(secs > 365 * 24 * 3600);
}

#[test]
fn parse_retry_after_http_date_past_returns_small() {
    // A date in the past should return at least 1 second (clamped)
    // or None since duration_since would fail
    let result = parse_retry_after("Mon, 01 Jan 2001 00:00:00 GMT");
    // Past date: duration_since(now) fails, returns None
    assert!(result.is_none());
}

#[test]
fn parse_retry_after_invalid() {
    assert_eq!(parse_retry_after("not-a-number"), None);
    assert_eq!(parse_retry_after(""), None);
    assert_eq!(parse_retry_after("abc def"), None);
}

#[test]
fn parse_retry_after_all_months() {
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    for month in months {
        let date = format!("Mon, 15 {} 2100 12:00:00 GMT", month);
        let result = parse_retry_after(&date);
        assert!(result.is_some(), "Failed to parse month: {}", month);
    }
}

#[test]
fn coalesce_real_world() {
    // Simulate tensor ranges with small gaps (like GGUF alignment padding)
    let ranges = vec![
        (1000, 5000),      // tensor 1: [1000, 6000)
        (6032, 5000),      // tensor 2: [6032, 11032) — 32-byte gap
        (11064, 5000),     // tensor 3: [11064, 16064) — 32-byte gap
        (5_000_000, 5000), // tensor 4: [5000000, 5005000) — ~5MB gap
    ];
    let merged = coalesce_byte_ranges(&ranges, 4 * 1024 * 1024);
    // First 3 should merge (within 4MB gap), tensor 4 is separate (>4MB gap)
    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0], (1000, 16064));
    assert_eq!(merged[1], (5_000_000, 5_005_000));
}

#[test]
fn parse_http_date_basic() {
    let dt = parse_http_date("Fri, 28 Feb 2026 04:00:00 GMT");
    assert!(dt.is_some());
    // Verify it's a reasonable timestamp (after 2025)
    let since_epoch = dt
        .unwrap()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    // 2026-02-28 should be ~1772 million seconds since epoch
    assert!(since_epoch > 1_770_000_000);
    assert!(since_epoch < 1_780_000_000);
}

#[test]
fn parse_http_date_invalid() {
    assert!(parse_http_date("not a date").is_none());
    assert!(parse_http_date("").is_none());
    assert!(parse_http_date("Fri, 28 Xxx 2026 04:00:00 GMT").is_none());
    // Wrong timezone
    assert!(parse_http_date("Fri, 28 Feb 2026 04:00:00 EST").is_none());
}

// ── Streaming tensor extraction ─────────────────────────────────────────────
//
// A coalesced range carries gap bytes between tensors, and only the tensor bytes
// belong in the shard. Getting this wrong writes a subtly corrupt shard that
// BLAKE3 only catches after the whole download. Every boundary is pinned.

/// Reference implementation: what the old buffer-the-whole-range code produced.
/// The streaming version must agree with it for any chunking.
fn extract_whole(tensors: &[(u64, u64)], range_start: u64, buf: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for &(off, size) in tensors {
        let a = (off - range_start) as usize;
        out.extend_from_slice(&buf[a..a + size as usize]);
    }
    out
}

/// Feed `buf` through the streaming extractor in fixed-size chunks.
fn extract_streamed(tensors: &[(u64, u64)], range_start: u64, buf: &[u8], chunk: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut abs = range_start;
    for c in buf.chunks(chunk) {
        for (s, l) in tensor_slices_in_chunk(tensors, abs, c.len()) {
            out.extend_from_slice(&c[s..s + l]);
        }
        abs += c.len() as u64;
    }
    out
}

#[test]
fn streaming_extraction_matches_buffering_at_every_chunk_size() {
    // Two tensors with a gap between them, inside one coalesced range.
    let range_start = 1000u64;
    let buf: Vec<u8> = (0..200u32).map(|i| (i % 251) as u8).collect();
    // tensor A: 1010..1050, gap, tensor B: 1080..1180
    let tensors = [(1010u64, 40u64), (1080u64, 100u64)];

    let expected = extract_whole(&tensors, range_start, &buf);
    assert_eq!(expected.len(), 140);

    // Chunk boundaries must not matter — including sizes that split a tensor,
    // land exactly on a boundary, or fall entirely inside the gap.
    for chunk in [1usize, 3, 7, 16, 39, 40, 41, 100, 199, 200, 512] {
        assert_eq!(
            extract_streamed(&tensors, range_start, &buf, chunk),
            expected,
            "mismatch at chunk size {chunk}"
        );
    }
}

#[test]
fn a_chunk_entirely_inside_a_gap_writes_nothing() {
    let tensors = [(100u64, 10u64), (200u64, 10u64)];
    // Chunk covering 130..160 — wholly between the two tensors.
    assert!(tensor_slices_in_chunk(&tensors, 130, 30).is_empty());
}

#[test]
fn a_tensor_spanning_several_chunks_is_reassembled_in_order() {
    let tensors = [(0u64, 100u64)];
    let mut got = Vec::new();
    for (i, start) in [0u64, 30, 60, 90].iter().enumerate() {
        let len = if i == 3 { 10 } else { 30 };
        got.extend(tensor_slices_in_chunk(&tensors, *start, len));
    }
    // Every chunk is fully inside the tensor, so each contributes all its bytes.
    assert_eq!(got, vec![(0, 30), (0, 30), (0, 30), (0, 10)]);
}

#[test]
fn tensors_touching_the_chunk_edges_are_included_whole() {
    let tensors = [(10u64, 5u64), (20u64, 5u64)];
    // Chunk exactly spans 10..25, so both tensors are fully covered.
    assert_eq!(
        tensor_slices_in_chunk(&tensors, 10, 15),
        vec![(0, 5), (10, 5)]
    );
    // A chunk ending exactly where a tensor starts must not include it.
    assert!(tensor_slices_in_chunk(&tensors, 0, 10).is_empty());
    // A chunk starting exactly where a tensor ends must not include it.
    assert!(tensor_slices_in_chunk(&tensors, 15, 5).is_empty());
}

#[test]
fn no_tensors_means_no_writes() {
    assert!(tensor_slices_in_chunk(&[], 0, 4096).is_empty());
}
