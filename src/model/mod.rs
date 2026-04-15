pub mod acquisition;
pub mod auto_manage;
pub mod distribution;
pub mod huggingface;
pub mod lora;
pub mod manifest;
pub mod registry;
pub mod shard;

use std::path::Path;

use crate::error::SwarmError;

/// Verify the filesystem hosting `dest_dir` has at least `need_bytes` free.
/// Returns `SwarmError::InsufficientDisk` with rounded-MB values when space
/// is short. Returns `Ok(())` when the check cannot determine available space
/// (unknown mount, permission error) — conservative: don't block a download
/// on a bad reading.
pub fn check_disk_space(dest_dir: &Path, need_bytes: u64) -> Result<(), SwarmError> {
    let mut disks = sysinfo::Disks::new();
    disks.refresh(true);
    let available = disks
        .iter()
        .filter(|d| dest_dir.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space());
    match available {
        Some(avail) if need_bytes > avail => Err(SwarmError::InsufficientDisk {
            need_mb: need_bytes / (1024 * 1024),
            have_mb: avail / (1024 * 1024),
        }),
        _ => Ok(()),
    }
}
