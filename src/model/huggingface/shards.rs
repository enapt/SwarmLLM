use std::time::SystemTime;

use super::probe::{download_gguf_header, download_tied_output_weight};
use super::{
    download_url, hf_headers, DownloadProgress, GgufFileInfo, BYTE_RANGE_COALESCE_GAP,
    HF_DOWNLOAD_CLIENT,
};

pub fn parse_retry_after(value: &str) -> Option<u64> {
    // Try delta-seconds first (most common for APIs)
    if let Ok(secs) = value.trim().parse::<u64>() {
        return Some(secs);
    }

    // Try HTTP-date format: "Day, DD Mon YYYY HH:MM:SS GMT"
    parse_http_date(value.trim()).and_then(|target| {
        let now = SystemTime::now();
        target.duration_since(now).ok().map(|d| d.as_secs().max(1))
    })
}

/// Parse an HTTP-date string into a SystemTime.
///
/// Supports the preferred format: "Day, DD Mon YYYY HH:MM:SS GMT"
pub(super) fn parse_http_date(s: &str) -> Option<SystemTime> {
    // Format: "Fri, 28 Feb 2026 04:00:00 GMT"
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() != 6 || parts[5] != "GMT" {
        return None;
    }

    let day: i64 = parts[1].trim_end_matches(',').parse().ok()?;
    let month: i64 = match parts[2] {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    };
    let year: i64 = parts[3].parse().ok()?;
    let time_parts: Vec<&str> = parts[4].split(':').collect();
    if time_parts.len() != 3 {
        return None;
    }
    let hour: i64 = time_parts[0].parse().ok()?;
    let min: i64 = time_parts[1].parse().ok()?;
    let sec: i64 = time_parts[2].parse().ok()?;

    // Convert to Unix timestamp using Rata Die algorithm
    let y = if month <= 2 { year - 1 } else { year };
    let m = if month <= 2 { month + 9 } else { month - 3 };
    let days = 365 * y + y / 4 - y / 100 + y / 400 + (m * 306 + 5) / 10 + (day - 1) - 719468; // Unix epoch offset
    let timestamp = days * 86400 + hour * 3600 + min * 60 + sec;
    if timestamp < 0 {
        return None;
    }

    Some(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(timestamp as u64))
}

/// Coalesce nearby byte ranges (within `max_gap` bytes) into fewer requests.
///
/// Input: sorted list of (offset, size) pairs.
pub fn coalesce_byte_ranges(ranges: &[(u64, u64)], max_gap: u64) -> Vec<(u64, u64)> {
    if ranges.is_empty() {
        return vec![];
    }
    let mut sorted: Vec<(u64, u64)> = ranges.iter().map(|&(off, sz)| (off, off + sz)).collect();
    sorted.sort_by_key(|r| r.0);

    let mut merged = vec![sorted[0]];
    for &(start, end) in &sorted[1..] {
        // Safety: merged always has at least one element (seeded above)
        let last = merged.last_mut().expect("merged is non-empty");
        if start <= last.1 + max_gap {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}

/// Download a layer-aligned shard from HuggingFace.
///
/// Takes a `LayerShardLayout` describing which tensors belong to this shard.
/// Coalesces nearby byte ranges into fewer HTTP Range requests, then packs
/// the tensor data sequentially into `shard_{idx:03}.bin`.
///
/// Connection errors are retried up to 3 times with backoff.
pub async fn download_shard(
    repo_id: &str,
    filename: &str,
    dest_dir: &std::path::Path,
    layout: &crate::inference::split::LayerShardLayout,
    progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
    cancel_flag: Option<&std::sync::atomic::AtomicBool>,
) -> Result<std::path::PathBuf, String> {
    let client = &*HF_DOWNLOAD_CLIENT;

    let url = download_url(repo_id, filename)?;
    let shard_index = layout.index;

    std::fs::create_dir_all(dest_dir).map_err(|e| format!("Failed to create dir: {e}"))?;

    // Pre-flight disk space check
    crate::model::check_disk_space(dest_dir, layout.size_bytes).map_err(|e| e.to_string())?;

    let dest_path = dest_dir.join(crate::model::shard::shard_filename(shard_index));
    let tmp_path = dest_dir.join(format!(
        "{}.tmp",
        crate::model::shard::shard_filename(shard_index)
    ));

    // Build byte ranges from tensor locations (gguf_offset, size)
    let tensor_ranges: Vec<(u64, u64)> = layout
        .tensors
        .iter()
        .map(|(_, offset, size)| (*offset, *size))
        .collect();

    // Coalesce nearby ranges (4MB gap tolerance) to reduce HTTP requests
    let coalesced = coalesce_byte_ranges(&tensor_ranges, BYTE_RANGE_COALESCE_GAP);
    let total_download_bytes: u64 = coalesced.iter().map(|(s, e)| e - s).sum();
    let expected_tensor_bytes: u64 = layout.tensors.iter().map(|(_, _, sz)| sz).sum();

    // Pre-compute how many tensor bytes each coalesced range contributes for resume support
    let mut range_tensor_bytes: Vec<u64> = Vec::with_capacity(coalesced.len());
    for (range_start, range_end) in &coalesced {
        let bytes: u64 = layout
            .tensors
            .iter()
            .filter(|(_, off, sz)| *off >= *range_start && *off + *sz <= *range_end)
            .map(|(_, _, sz)| sz)
            .sum();
        range_tensor_bytes.push(bytes);
    }

    // Resume support: check if tmp file exists with partial data
    let existing_bytes = tokio::fs::metadata(&tmp_path)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    let mut ranges_to_skip: usize = 0;
    if existing_bytes > 0 {
        // Determine how many complete ranges are already in the tmp file
        let mut cumulative = 0u64;
        for &rb in &range_tensor_bytes {
            if cumulative + rb <= existing_bytes {
                cumulative += rb;
                ranges_to_skip += 1;
            } else {
                break;
            }
        }
        if ranges_to_skip > 0 && cumulative == existing_bytes {
            tracing::info!(
                shard = shard_index,
                existing_bytes,
                ranges_complete = ranges_to_skip,
                ranges_total = coalesced.len(),
                "Resuming shard download from partial .tmp file"
            );
        } else if cumulative != existing_bytes {
            // Partial range — can't resume cleanly, restart
            ranges_to_skip = 0;
            tracing::info!(
                shard = shard_index,
                existing_bytes,
                expected_boundary = cumulative,
                "Partial .tmp file doesn't align to range boundary, restarting download"
            );
        }
    }

    tracing::info!(
        shard = shard_index,
        tensors = layout.tensors.len(),
        ranges = coalesced.len(),
        download_bytes = total_download_bytes,
        tensor_bytes = expected_tensor_bytes,
        resuming_from_range = ranges_to_skip,
        "Starting shard download"
    );

    use tokio::io::AsyncWriteExt;
    let mut file = if ranges_to_skip > 0 {
        // Append mode — open existing tmp file for appending
        tokio::fs::OpenOptions::new()
            .append(true)
            .open(&tmp_path)
            .await
            .map_err(|e| format!("Failed to open tmp file for resume: {e}"))?
    } else {
        tokio::fs::File::create(&tmp_path)
            .await
            .map_err(|e| format!("Failed to create tmp file: {e}"))?
    };

    // Account for already-downloaded bytes in progress tracking
    let mut downloaded: u64 = if ranges_to_skip > 0 {
        coalesced[..ranges_to_skip].iter().map(|(s, e)| e - s).sum()
    } else {
        0
    };

    // Download each coalesced range and write to file (skip already-completed ranges)
    for (range_idx, (range_start, range_end)) in coalesced.iter().enumerate() {
        if range_idx < ranges_to_skip {
            continue;
        }
        let http_retry_delays = crate::config::NETWORK_RETRY_DELAYS;
        let mut resp = None;

        for attempt in 0..=3u32 {
            let result = hf_headers(client.get(&url))
                .header("Range", format!("bytes={}-{}", range_start, range_end - 1))
                .send()
                .await;

            match result {
                Ok(r) => {
                    let status = r.status().as_u16();
                    if status == 429 || status == 503 {
                        if attempt < 3 {
                            let retry_secs = r
                                .headers()
                                .get("retry-after")
                                .and_then(|v| v.to_str().ok())
                                .and_then(parse_retry_after)
                                .unwrap_or(http_retry_delays[attempt as usize])
                                .min(600);
                            tracing::warn!(
                                status,
                                retry_after_secs = retry_secs,
                                attempt = attempt + 1,
                                "HF rate limited, retrying"
                            );
                            tokio::time::sleep(std::time::Duration::from_secs(retry_secs)).await;
                            continue;
                        }
                        return Err(format!("Shard download returned {} after retries", status));
                    }
                    resp = Some(r);
                    break;
                }
                Err(e) => {
                    if attempt < 3 {
                        tracing::warn!(error = %e, attempt = attempt + 1, "Shard range request failed, retrying");
                        tokio::time::sleep(std::time::Duration::from_secs(
                            http_retry_delays[attempt as usize],
                        ))
                        .await;
                        continue;
                    }
                    return Err(format!("Shard download failed after retries: {e}"));
                }
            }
        }
        let resp = resp.ok_or("Shard download failed: no response")?;

        if resp.status().as_u16() != 206 && !resp.status().is_success() {
            return Err(format!("Shard download returned {}", resp.status()));
        }

        // Buffer the coalesced range so we can extract only tensor data (skip gap bytes).
        // Coalesced ranges merge nearby tensors with up to 4MB gaps between them.
        // Writing the raw HTTP response would include those gap bytes, causing shard_offset
        // mismatches in ShardReader.
        use futures::StreamExt;
        let mut range_buf: Vec<u8> = Vec::new();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            // Check cancel flag every chunk
            if let Some(flag) = cancel_flag {
                if flag.load(std::sync::atomic::Ordering::Acquire) {
                    // Clean up tmp file
                    drop(file);
                    let _ = tokio::fs::remove_file(&tmp_path).await;
                    return Err("Download cancelled".to_string());
                }
            }
            let data = chunk.map_err(|e| format!("Stream error: {e}"))?;
            range_buf.extend_from_slice(&data);
            downloaded += data.len() as u64;

            if let Some(ref tx) = progress_tx {
                let _ = tx.try_send(DownloadProgress {
                    downloaded_bytes: downloaded,
                    total_bytes: total_download_bytes,
                });
            }
        }

        // Verify we received the expected number of bytes for this range
        let expected_range_bytes = *range_end - *range_start;
        if range_buf.len() as u64 != expected_range_bytes {
            tracing::warn!(
                shard = shard_index,
                expected = expected_range_bytes,
                received = range_buf.len(),
                range_start = range_start,
                range_end = range_end,
                "Coalesced range received incomplete data, retrying"
            );
            // Retry this range once — adjust progress counter for the failed bytes
            downloaded = downloaded.saturating_sub(range_buf.len() as u64);
            range_buf.clear();
            let retry_resp = hf_headers(client.get(&url))
                .header("Range", format!("bytes={}-{}", range_start, range_end - 1))
                .send()
                .await
                .map_err(|e| format!("Range retry failed: {e}"))?;
            if !retry_resp.status().is_success() {
                return Err(format!(
                    "Range retry returned HTTP {} for shard {}",
                    retry_resp.status(),
                    shard_index
                ));
            }
            let mut stream = retry_resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let data = chunk.map_err(|e| format!("Stream error on retry: {e}"))?;
                range_buf.extend_from_slice(&data);
                downloaded += data.len() as u64;
            }
            if range_buf.len() as u64 != expected_range_bytes {
                return Err(format!(
                    "Shard {} range {}-{}: expected {} bytes but got {} after retry",
                    shard_index,
                    range_start,
                    range_end,
                    expected_range_bytes,
                    range_buf.len()
                ));
            }
        }

        // Extract only tensor data from the buffered range, skipping inter-tensor gaps
        let mut tensors_extracted = 0u32;
        let mut bytes_written = 0u64;
        for (name, tensor_offset, tensor_size) in &layout.tensors {
            if *tensor_offset >= *range_start && *tensor_offset + *tensor_size <= *range_end {
                let buf_offset = (*tensor_offset - *range_start) as usize;
                let buf_end = buf_offset + *tensor_size as usize;
                if buf_end <= range_buf.len() {
                    file.write_all(&range_buf[buf_offset..buf_end])
                        .await
                        .map_err(|e| format!("Write error: {e}"))?;
                    tensors_extracted += 1;
                    bytes_written += *tensor_size;
                } else {
                    return Err(format!(
                        "Shard {}: tensor {} buffer overflow (need {} bytes, have {})",
                        shard_index,
                        name,
                        buf_end,
                        range_buf.len()
                    ));
                }
            }
        }
        tracing::debug!(
            shard = shard_index,
            tensors_extracted,
            bytes_written,
            "Range extraction complete"
        );
    }

    file.flush()
        .await
        .map_err(|e| format!("Flush error: {e}"))?;
    drop(file);

    // Verify file size matches expected tensor data total
    let expected_size: u64 = layout.tensors.iter().map(|(_, _, sz)| sz).sum();
    let actual_size = tokio::fs::metadata(&tmp_path)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    if actual_size != expected_size {
        // Clean up the incomplete file
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return Err(format!(
            "Shard {} size mismatch: expected {} bytes (from {} tensors) but wrote {} bytes",
            shard_index,
            expected_size,
            layout.tensors.len(),
            actual_size
        ));
    }

    // Atomic rename .tmp → .bin
    tokio::fs::rename(&tmp_path, &dest_path)
        .await
        .map_err(|e| format!("Failed to rename tmp to final shard file: {e}"))?;

    tracing::info!(
        shard = shard_index,
        size = actual_size,
        path = %dest_path.display(),
        "Downloaded layer-aligned shard from HuggingFace"
    );

    Ok(dest_path)
}

/// Download header + specified shards from HuggingFace.
///
/// Main entry point for shard-level downloads:
/// 1. Probes the remote file to get tensor metadata and layer-aligned shard layouts
/// 2. Downloads the GGUF header
/// 3. Downloads each requested shard via coalesced Range requests
/// 4. Returns the model directory and file info
pub async fn download_shards(
    repo_id: &str,
    filename: &str,
    dest_dir: &std::path::Path,
    shard_indices: &[u32],
    info: &GgufFileInfo,
    progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
) -> Result<std::path::PathBuf, String> {
    // Download the GGUF header
    download_gguf_header(repo_id, filename, dest_dir, info.header_size).await?;

    // Download tied output weight for weight-tied models (no separate output.weight)
    if let Err(e) =
        download_tied_output_weight(repo_id, filename, dest_dir, &info.tensor_meta).await
    {
        tracing::warn!(error = %e, "Tied output weight download failed (non-fatal)");
    }

    let total_shard_bytes: u64 = shard_indices
        .iter()
        .filter_map(|&idx| info.layouts.get(idx as usize))
        .map(|layout| layout.size_bytes)
        .sum();

    let mut cumulative_downloaded: u64 = 0;

    for &shard_idx in shard_indices {
        let layout = info.layouts.get(shard_idx as usize).ok_or_else(|| {
            format!(
                "Shard index {} out of range (max {})",
                shard_idx,
                info.layouts.len().saturating_sub(1)
            )
        })?;

        // Per-shard progress mapping to cumulative
        let (shard_tx, mut shard_rx) = tokio::sync::mpsc::channel::<DownloadProgress>(64);
        let progress_tx = progress_tx.clone();
        let base = cumulative_downloaded;
        let total = total_shard_bytes;
        let progress_task = tokio::spawn(async move {
            while let Some(prog) = shard_rx.recv().await {
                if let Some(ref tx) = progress_tx {
                    let _ = tx.try_send(DownloadProgress {
                        downloaded_bytes: base + prog.downloaded_bytes,
                        total_bytes: total,
                    });
                }
            }
        });

        download_shard(repo_id, filename, dest_dir, layout, Some(shard_tx), None).await?;
        let _ = progress_task.await;

        cumulative_downloaded += layout.size_bytes;
    }

    Ok(dest_dir.to_path_buf())
}

// ---- HF API response types ----
