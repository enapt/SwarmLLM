use super::shards::{coalesce_byte_ranges, parse_http_date};
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
