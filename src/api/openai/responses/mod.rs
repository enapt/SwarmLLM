//! OpenAI `/v1/responses` endpoint — request/response types and (in later
//! milestones) handlers, translation, streaming, and persistence.
//!
//! Milestone 1 lands types only. Routes are wired in M2.

pub mod types;

pub use types::*;
