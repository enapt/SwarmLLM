//! HuggingFace download progress tracking.
//!
//! Background task that reads DownloadProgress events from the HF downloader
//! and updates the shared AcquisitionStatus — also surfaces slow/stalled
//! warnings to the dashboard.

/// Slow-download detection threshold: 100 KB/s.
const SLOW_DOWNLOAD_SPEED_THRESHOLD: u64 = 102400;
/// Duration in seconds before emitting a slow-download warning.
const SLOW_DOWNLOAD_WARN_SECS: f64 = 30.0;
/// Hard abort: seconds of zero-byte progress before we trip the cancel flag.
/// The HF download task checks the flag every chunk, so this fires fast.
const STALL_ABORT_SECS: u64 = 120;
/// How often the stall-check branch wakes up to test whether STALL_ABORT_SECS has elapsed.
const STALL_CHECK_INTERVAL_SECS: u64 = 10;

/// Spawn a background task that reads download progress events and updates acquisition_progress.
pub(super) fn spawn_progress_updater(
    shared: std::sync::Arc<crate::daemon::state::SharedState>,
    mid: crate::types::ModelId,
    mut prx: tokio::sync::mpsc::Receiver<crate::model::huggingface::DownloadProgress>,
) {
    let mut shutdown_rx = shared.shutdown_rx();
    tokio::spawn(async move {
        let mut last_bytes = 0u64;
        let mut last_time = std::time::Instant::now();
        let mut last_progress_at = std::time::Instant::now();
        let mut slow_since: Option<std::time::Instant> = None;
        let mut throttle_warned = false;
        let stall_tick = std::time::Duration::from_secs(STALL_CHECK_INTERVAL_SECS);
        loop {
            tokio::select! {
                prog = prx.recv() => {
                    let Some(prog) = prog else { break };
                    if let Some(mut entry) = shared.models.acquisition_progress.get_mut(&mid) {
                        entry.downloaded_bytes = prog.downloaded_bytes;
                        entry.total_bytes = prog.total_bytes;
                        let now = std::time::Instant::now();
                        if prog.downloaded_bytes > last_bytes {
                            last_progress_at = now;
                        }
                        let dt = now.duration_since(last_time).as_secs_f64();
                        if dt > 0.5 {
                            let speed =
                                (prog.downloaded_bytes.saturating_sub(last_bytes) as f64 / dt) as u64;
                            entry.speed_bytes_per_sec = speed;
                            last_bytes = prog.downloaded_bytes;
                            last_time = now;

                            // Slow-download detection: warn once after sustained slow speed
                            if speed > 0 && speed < SLOW_DOWNLOAD_SPEED_THRESHOLD {
                                let since = *slow_since.get_or_insert(now);
                                if !throttle_warned && now.duration_since(since).as_secs_f64() > SLOW_DOWNLOAD_WARN_SECS {
                                    throttle_warned = true;
                                    let speed_str = format!("{:.1} KB/s", speed as f64 / 1024.0);
                                    shared.emit_activity(
                                        crate::daemon::state::ActivityEvent::new(
                                            "model", "download_slow",
                                            format!("Download is slow ({speed_str}) — this can happen with popular models. It will keep going."),
                                        )
                                        .with_model(mid.0.clone())
                                        .with_detail_str(speed_str)
                                        .with_toast("warning", 10000),
                                    );
                                }
                            } else {
                                slow_since = None;
                            }
                        }
                    }
                }
                _ = tokio::time::sleep(stall_tick) => {
                    if last_progress_at.elapsed().as_secs() >= STALL_ABORT_SECS {
                        if let Some(flag) = shared.models.download_cancel_flags.get(&mid) {
                            flag.store(true, std::sync::atomic::Ordering::Release);
                        }
                        let display = shared.model_registry.display_name(&mid);
                        shared.emit_activity(
                            crate::daemon::state::ActivityEvent::new(
                                "download",
                                "hf_download_stalled",
                                format!(
                                    "HuggingFace download of {} stalled for {}s — aborting",
                                    display, STALL_ABORT_SECS
                                ),
                            )
                            .with_model(mid.0.clone())
                            .with_toast("error", 8000),
                        );
                        break;
                    }
                }
                _ = shutdown_rx.changed() => break,
            }
        }
    });
}
