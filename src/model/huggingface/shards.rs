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
/// Which parts of an incoming chunk are real tensor data?
///
/// A coalesced range merges nearby tensors and carries the gap bytes between
/// them; only the tensor bytes belong in the shard file. Given the tensor spans
/// that fall in this range (sorted by offset, non-overlapping) and the absolute
/// file offset the chunk starts at, this returns the `(start_in_chunk, len)`
/// slices to write, in order.
///
/// Exists so the download can write as bytes arrive instead of buffering the
/// whole range. The previous approach held an entire coalesced range in memory —
/// measured live at **159 MB for a single 353 MB shard** — and wrote it in one
/// go, which is both a memory risk on small nodes and the reason a download sat
/// at zero bytes for tens of seconds before anything appeared on disk.
///
/// Pure and separately tested: an off-by-one here would silently corrupt every
/// shard it touches, and the BLAKE3 manifest check would only catch it after a
/// full download.
pub(super) fn tensor_slices_in_chunk(
    tensors: &[(u64, u64)],
    chunk_start: u64,
    chunk_len: usize,
) -> Vec<(usize, usize)> {
    let chunk_end = chunk_start + chunk_len as u64;
    let mut out = Vec::new();
    for &(t_off, t_size) in tensors {
        let t_end = t_off + t_size;
        // Sorted by offset, so once a tensor starts past this chunk we are done.
        if t_off >= chunk_end {
            break;
        }
        if t_end <= chunk_start {
            continue;
        }
        let overlap_start = t_off.max(chunk_start);
        let overlap_end = t_end.min(chunk_end);
        if overlap_end > overlap_start {
            out.push((
                (overlap_start - chunk_start) as usize,
                (overlap_end - overlap_start) as usize,
            ));
        }
    }
    out
}

pub(super) fn coalesce_byte_ranges(ranges: &[(u64, u64)], max_gap: u64) -> Vec<(u64, u64)> {
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
    // R138 (closes R105 deferral): sidecar that binds the on-disk
    // partial .tmp to the layout it was downloaded against. Without
    // this, an HF tensor layout change between a partial download
    // (V1 offsets) and a resume attempt (V2 offsets) could coincidentally
    // size-match and resume by APPENDING V2 bytes to V1 prefix — silently
    // corrupting the shard. We hash the (offset, size) pairs sorted by
    // offset and refuse to resume if the layout drifted. The sidecar is
    // tiny (32 bytes hex + a u64) and lives alongside the .tmp file; it
    // is removed on successful rename to dest_path AND on any error
    // cleanup of the .tmp.
    let layout_path = dest_dir.join(format!(
        "{}.tmp.layout",
        crate::model::shard::shard_filename(shard_index)
    ));

    // Build byte ranges from tensor locations (gguf_offset, size)
    let tensor_ranges: Vec<(u64, u64)> = layout
        .tensors
        .iter()
        .map(|(_, offset, size)| (*offset, *size))
        .collect();

    // Compute a stable layout hash over the (offset, size) ordered list.
    // Tensor names omitted on purpose: the bytes we write only depend on
    // (offset, size). Two different tensor namings with identical offsets
    // would produce identical bytes; one HF revision that renamed a
    // tensor without changing offsets is safe to resume.
    let layout_hash: [u8; 32] = {
        let mut sorted = tensor_ranges.clone();
        sorted.sort_by_key(|(off, _)| *off);
        let mut hasher = blake3::Hasher::new();
        for (off, sz) in &sorted {
            hasher.update(&off.to_le_bytes());
            hasher.update(&sz.to_le_bytes());
        }
        hasher.finalize().into()
    };

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
        // R138 (closes R105 deferral) — guard the resume against an HF
        // tensor layout change between the prior partial download and
        // this attempt. We refuse to resume unless the sidecar layout
        // hash exists AND matches the current layout. Mismatch or
        // missing sidecar → discard both files and restart.
        let sidecar = tokio::fs::read(&layout_path).await.ok();
        let layout_matches = sidecar
            .as_deref()
            .map(|raw| raw == layout_hash.as_slice())
            .unwrap_or(false);
        if !layout_matches {
            tracing::info!(
                shard = shard_index,
                existing_bytes,
                sidecar_present = sidecar.is_some(),
                "HF layout drift detected (or sidecar missing) — discarding .tmp and restarting"
            );
            let _ = tokio::fs::remove_file(&tmp_path).await;
            let _ = tokio::fs::remove_file(&layout_path).await;
        } else {
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
                let _ = tokio::fs::remove_file(&tmp_path).await;
                let _ = tokio::fs::remove_file(&layout_path).await;
                tracing::info!(
                    shard = shard_index,
                    existing_bytes,
                    expected_boundary = cumulative,
                    "Partial .tmp file doesn't align to range boundary, restarting download"
                );
            }
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
        // Append mode — open existing tmp file for appending. Sidecar
        // already exists and matched (we checked above), so no rewrite.
        tokio::fs::OpenOptions::new()
            .append(true)
            .open(&tmp_path)
            .await
            .map_err(|e| format!("Failed to open tmp file for resume: {e}"))?
    } else {
        let f = tokio::fs::File::create(&tmp_path)
            .await
            .map_err(|e| format!("Failed to create tmp file: {e}"))?;
        // Pin the layout to the new .tmp by writing the sidecar BEFORE
        // any bytes land in .tmp — a subsequent resume can only match
        // the layout if the data we're about to write was produced by
        // THIS run's `layout`.
        tokio::fs::write(&layout_path, layout_hash)
            .await
            .map_err(|e| format!("Failed to write layout sidecar: {e}"))?;
        f
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

        // Write tensor data straight to disk as it arrives, skipping the gap
        // bytes a coalesced range carries between tensors. (Writing the raw HTTP
        // response would include those gaps and break `ShardReader`'s
        // shard_offset arithmetic.)
        //
        // Streamed rather than buffered: holding a whole coalesced range in
        // memory was measured live at 159 MB for one 353 MB shard, which is both
        // an OOM risk on a small node downloading several shards and the reason
        // the file sat at zero bytes for tens of seconds before anything landed.
        // Now memory is one chunk and the file grows continuously.
        use futures::StreamExt;
        // Tensor spans inside this range, in file order — the order they must be
        // written in. Same membership test as the old extract loop.
        let range_tensors: Vec<(u64, u64)> = {
            let mut v: Vec<(u64, u64)> = layout
                .tensors
                .iter()
                .filter(|(_, off, size)| *off >= *range_start && *off + *size <= *range_end)
                .map(|(_, off, size)| (*off, *size))
                .collect();
            v.sort_unstable_by_key(|(off, _)| *off);
            v
        };
        // Where to rewind to if this range fails: buffering used to give
        // per-range atomicity for free, streaming has to restore it explicitly.
        //
        // **Flush before asking how long the file is.** `tokio::fs::File`
        // buffers writes, and `metadata()` reports what the OS has, not what
        // is still sitting in that buffer — so without this the length reads
        // SHORT by however much of the previous ranges had not been written
        // out yet. That number is then used as a truncation target on the
        // retry path below, so a single incomplete range would rewind the file
        // past its own start and destroy ranges that had already succeeded,
        // leaving a download that reports far fewer bytes than it wrote.
        //
        // The retry path already flushes for exactly this reason before its
        // `set_len`; capturing the length without the same flush was the half
        // that was missed. Reported from the field 2026-08-31 as
        // `Shard 14 size mismatch: expected 523493376 bytes ... wrote 0 bytes`
        // following a `Coalesced range received incomplete data, retrying`.
        file.flush()
            .await
            .map_err(|e| format!("Flush before range checkpoint failed: {e}"))?;
        let file_len_before_range = file
            .metadata()
            .await
            .map(|m| m.len())
            .map_err(|e| format!("Failed to stat tmp file: {e}"))?;
        let mut range_pos = *range_start;
        let mut range_bytes_written = 0u64;
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            // Check cancel flag every chunk
            if let Some(flag) = cancel_flag {
                if flag.load(std::sync::atomic::Ordering::Acquire) {
                    // Clean up tmp file AND its R138 layout sidecar.
                    // Keeping the sidecar alone would mismatch a future
                    // fresh-download's data; removing the .tmp without
                    // the sidecar would let the next attempt resume
                    // against stale bytes — both files move as a unit.
                    drop(file);
                    let _ = tokio::fs::remove_file(&tmp_path).await;
                    let _ = tokio::fs::remove_file(&layout_path).await;
                    return Err("Download cancelled".to_string());
                }
            }
            let data = chunk.map_err(|e| format!("Stream error: {e}"))?;
            for (s, l) in tensor_slices_in_chunk(&range_tensors, range_pos, data.len()) {
                file.write_all(&data[s..s + l])
                    .await
                    .map_err(|e| format!("Write error: {e}"))?;
                range_bytes_written += l as u64;
            }
            range_pos += data.len() as u64;
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
        let received = range_pos - *range_start;
        if received != expected_range_bytes {
            tracing::warn!(
                shard = shard_index,
                expected = expected_range_bytes,
                received,
                range_start = range_start,
                range_end = range_end,
                "Coalesced range received incomplete data, retrying"
            );
            // Rewind the partial writes. Buffering used to make a failed range a
            // no-op on disk; streaming has to undo it explicitly or the retry
            // would append a second copy of the bytes it already wrote.
            file.flush()
                .await
                .map_err(|e| format!("Flush before rewind failed: {e}"))?;
            file.set_len(file_len_before_range)
                .await
                .map_err(|e| format!("Failed to rewind partial range: {e}"))?;
            {
                use tokio::io::AsyncSeekExt as _;
                file.seek(std::io::SeekFrom::End(0))
                    .await
                    .map_err(|e| format!("Failed to seek after rewind: {e}"))?;
            }
            // Adjust progress for the discarded bytes
            downloaded = downloaded.saturating_sub(received);
            range_bytes_written = 0;
            range_pos = *range_start;

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
                for (s, l) in tensor_slices_in_chunk(&range_tensors, range_pos, data.len()) {
                    file.write_all(&data[s..s + l])
                        .await
                        .map_err(|e| format!("Write error on retry: {e}"))?;
                    range_bytes_written += l as u64;
                }
                range_pos += data.len() as u64;
                downloaded += data.len() as u64;
            }
            if range_pos - *range_start != expected_range_bytes {
                return Err(format!(
                    "Shard {} range {}-{}: expected {} bytes but got {} after retry",
                    shard_index,
                    range_start,
                    range_end,
                    expected_range_bytes,
                    range_pos - *range_start
                ));
            }
        }

        let tensors_extracted = range_tensors.len() as u32;
        let bytes_written = range_bytes_written;
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
        // Clean up the incomplete file AND its R138 layout sidecar.
        let _ = tokio::fs::remove_file(&tmp_path).await;
        let _ = tokio::fs::remove_file(&layout_path).await;
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
    // Remove the R138 layout sidecar — the .bin file is the new source
    // of truth; the sidecar only existed to guard partial-resume.
    let _ = tokio::fs::remove_file(&layout_path).await;

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

        // Idempotency: a shard already on disk at exactly the expected size is
        // the shard we would download, so skip it (external report 2026-07-25:
        // `swarmllm get-model` on an already-complete model re-fetched ~353MB
        // byte-for-byte). Size is the right check here — the layout's
        // `size_bytes` comes from the same remote GGUF we would fetch from, and
        // a truncated/interrupted download lands in `<shard>.tmp` rather than at
        // `dest_path`, so a full-size file at the final path was completed. The
        // content is verified by BLAKE3 on every load regardless, so a corrupt
        // same-size file is caught there rather than by re-downloading blind.
        let dest_path = dest_dir.join(crate::model::shard::shard_filename(shard_idx));
        if std::fs::metadata(&dest_path).is_ok_and(|m| m.len() == layout.size_bytes) {
            tracing::info!(
                shard = shard_idx,
                size_bytes = layout.size_bytes,
                path = %dest_path.display(),
                "Shard already present at the expected size — skipping download"
            );
            cumulative_downloaded += layout.size_bytes;
            // Keep the progress stream monotonic so the dashboard doesn't stall
            // at 0% for a run where every shard is skipped.
            if let Some(ref tx) = progress_tx {
                let _ = tx.try_send(DownloadProgress {
                    downloaded_bytes: cumulative_downloaded,
                    total_bytes: total_shard_bytes,
                });
            }
            continue;
        }

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

/// The rewind checkpoint rests on a property of `tokio::fs::File`.
#[cfg(test)]
mod rewind_checkpoint {
    /// **`metadata()` reports what the OS has, not what is still in the write
    /// buffer, and whether those agree is a RACE** — tokio's writes complete on
    /// a background pool. Observed both ways on the same machine minutes apart:
    /// after two writes totalling 64 KiB, `metadata()` returned **8192** on one
    /// run and the full **65536** on the next.
    ///
    /// That is why the download loop flushes before capturing the file length
    /// it will later use as a `set_len` truncation target. Reading it short
    /// means the retry rewinds past ranges that had already succeeded and
    /// destroys them, so the finished file is smaller than the tensors it was
    /// built from and fails its size check having actually received the bytes.
    /// The retry path already flushed before its own `set_len`; capturing the
    /// checkpoint without the same flush was the half that was missed.
    ///
    /// The race itself cannot be asserted without flakiness — this pins the
    /// property the flush GUARANTEES, which is what the fix relies on.
    #[tokio::test]
    async fn a_flushed_file_reports_everything_written_to_it() {
        use tokio::io::AsyncWriteExt;
        let dir = std::env::temp_dir().join(format!("swarm_rewind_probe_{}", std::process::id()));
        let _ = tokio::fs::create_dir_all(&dir).await;
        let path = dir.join("probe.bin");
        let mut f = tokio::fs::File::create(&path).await.unwrap();

        f.write_all(&[7u8; 100]).await.unwrap();
        f.write_all(&vec![7u8; 64 * 1024 - 100]).await.unwrap();
        f.flush().await.unwrap();

        let flushed = f.metadata().await.unwrap().len();
        let _ = tokio::fs::remove_dir_all(&dir).await;
        assert_eq!(
            flushed,
            64 * 1024,
            "after a flush the checkpoint must be the whole file, or the rewind \
             target is short and a retry truncates good ranges away"
        );
    }
}
