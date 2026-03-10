//! Re-export all types from the swarmllm-types crate.
//!
//! This module exists for backwards compatibility so that existing
//! `use crate::types::*` imports continue to work without changes.
//!
//! Also re-exports extension traits for types that have methods
//! defined in the main crate.

pub use swarmllm_types::*;

// Extension traits for types defined in swarmllm-types
pub use crate::identity::nickname::NicknameRecordExt;
pub use crate::model::manifest::ModelManifestExt;
pub use crate::pool::types::BlindedPoolInvitationExt;
