//! Integration tests for Phase 10-11 features.
//!
//! Run with: cargo test --test integration_phase10_11 -- --test-threads=1

mod end_to_end;
mod test_config_reload;
mod test_credit_escrow;
mod test_inference_features;
mod test_kv_cache;
mod test_metrics_health;
mod test_relay_mixed_version;
mod test_trust;
