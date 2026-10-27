pub mod acquisition;
pub mod auto_manage;
pub mod canonical;
pub mod distribution;
pub mod huggingface;
pub mod lora;
pub mod manifest;
pub mod reference;
pub mod registry;
pub mod shard;

use std::path::Path;

use crate::error::SwarmError;

/// Verify the filesystem hosting `dest_dir` has at least `need_bytes` free.
/// Returns `SwarmError::InsufficientDisk` with rounded-MB values when space
/// is short. Returns `Ok(())` when the space cannot be read — don't block a
/// download on a bad reading. The disk is `auto_manage::disk_space_for`'s, the
/// one reading of the disk a path is on (#334 (4)).
pub fn check_disk_space(dest_dir: &Path, need_bytes: u64) -> Result<(), SwarmError> {
    match auto_manage::disk_space_for(dest_dir) {
        Some(disk) if need_bytes > disk.free_bytes => Err(SwarmError::InsufficientDisk {
            need_mb: need_bytes / (1024 * 1024),
            have_mb: disk.free_bytes / (1024 * 1024),
        }),
        _ => Ok(()),
    }
}
