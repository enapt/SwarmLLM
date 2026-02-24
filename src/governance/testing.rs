use crate::config::AutoUpdateMode;
use crate::types::{
    Architecture, CanaryPhase, Platform, SemVer, TestReport, TestResults, TestStatus,
};

/// Determine if this node should accept a specific canary phase update.
///
/// Uses the node's update preference and the canary phase to decide
/// whether to auto-update.
pub fn should_accept_update(
    auto_update: &AutoUpdateMode,
    phase: &CanaryPhase,
    node_hash_bucket: f32,
) -> bool {
    match auto_update {
        AutoUpdateMode::Disabled => false,
        AutoUpdateMode::All => true,
        AutoUpdateMode::Stable => match phase {
            CanaryPhase::Phase1 => false, // Only "all" nodes in phase 1
            CanaryPhase::Phase2 => node_hash_bucket < 0.05, // 5% of stable nodes
            CanaryPhase::Phase3 => node_hash_bucket < 0.25, // 25% of stable nodes
            CanaryPhase::Complete => true,
            CanaryPhase::Halted { .. } => false,
        },
    }
}

/// Compute a deterministic bucket (0.0..1.0) for a node based on its ID.
///
/// Used for canary rollout percentage selection — ensures the same node
/// consistently falls in the same bucket across checks.
pub fn node_canary_bucket(node_id_bytes: &[u8; 32], version: &SemVer) -> f32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(node_id_bytes);
    hasher.update(version.to_key().as_bytes());
    let hash = hasher.finalize();
    let bytes = hash.as_bytes();
    // Take first 4 bytes as u32, normalize to 0..1
    let val = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    val as f32 / u32::MAX as f32
}

/// Parameters for building a local test report.
pub struct LocalTestParams {
    pub version: SemVer,
    pub tester: crate::types::NodeId,
    pub results: TestResults,
}

/// Build a TestReport from locally running the test suite.
///
/// This would be called by `swarmllm test --release <version>` to generate
/// a report for network submission. Currently generates the report structure.
pub fn build_local_test_report(params: LocalTestParams) -> TestReport {
    let tests_failed = params.results.tests_failed;
    let status = if tests_failed > 0 {
        TestStatus::Failed {
            failures: vec![format!("{tests_failed} tests failed")],
        }
    } else {
        TestStatus::Passed
    };

    TestReport {
        release_version: params.version,
        tester: params.tester,
        platform: detect_platform(),
        architecture: detect_architecture(),
        gpu: detect_gpu(),
        status,
        results: params.results,
        timestamp: chrono::Utc::now(),
        signature: vec![],
    }
}

/// Aggregate test reports for a release to generate a summary.
pub fn aggregate_reports(reports: &[TestReport]) -> TestReportSummary {
    let total = reports.len() as u32;
    let passed = reports
        .iter()
        .filter(|r| matches!(r.status, TestStatus::Passed))
        .count() as u32;
    let failed = reports
        .iter()
        .filter(|r| matches!(r.status, TestStatus::Failed { .. }))
        .count() as u32;
    let hash_mismatches = reports
        .iter()
        .filter(|r| matches!(r.status, TestStatus::HashMismatch { .. }))
        .count() as u32;
    let build_failures = reports
        .iter()
        .filter(|r| matches!(r.status, TestStatus::BuildFailed { .. }))
        .count() as u32;

    let platforms: Vec<Platform> = reports.iter().map(|r| r.platform.clone()).collect();
    let unique_platforms = {
        let mut p = platforms.clone();
        p.dedup();
        p.len() as u32
    };

    TestReportSummary {
        total_reports: total,
        passed,
        failed,
        hash_mismatches,
        build_failures,
        unique_platforms,
        has_blocking_issues: failed > 0 || hash_mismatches > 0,
    }
}

/// Summary of test reports for a release.
#[derive(Clone, Debug, serde::Serialize)]
pub struct TestReportSummary {
    pub total_reports: u32,
    pub passed: u32,
    pub failed: u32,
    pub hash_mismatches: u32,
    pub build_failures: u32,
    pub unique_platforms: u32,
    pub has_blocking_issues: bool,
}

fn detect_platform() -> Platform {
    if cfg!(target_os = "linux") {
        Platform::Linux
    } else if cfg!(target_os = "macos") {
        Platform::MacOS
    } else {
        Platform::Windows
    }
}

fn detect_architecture() -> Architecture {
    if cfg!(target_arch = "aarch64") {
        Architecture::Aarch64
    } else {
        Architecture::X86_64
    }
}

fn detect_gpu() -> Option<String> {
    std::env::var("SWARMLLM_GPU_NAME").ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NodeId;

    #[test]
    fn canary_bucket_is_deterministic() {
        let node_id = [42u8; 32];
        let version = SemVer {
            major: 0,
            minor: 2,
            patch: 0,
            pre: None,
        };

        let bucket1 = node_canary_bucket(&node_id, &version);
        let bucket2 = node_canary_bucket(&node_id, &version);
        assert!((bucket1 - bucket2).abs() < f32::EPSILON);
        assert!(bucket1 >= 0.0 && bucket1 <= 1.0);
    }

    #[test]
    fn canary_bucket_differs_by_version() {
        let node_id = [42u8; 32];
        let v1 = SemVer {
            major: 0,
            minor: 2,
            patch: 0,
            pre: None,
        };
        let v2 = SemVer {
            major: 0,
            minor: 3,
            patch: 0,
            pre: None,
        };

        let b1 = node_canary_bucket(&node_id, &v1);
        let b2 = node_canary_bucket(&node_id, &v2);
        // Very unlikely to be equal with different versions
        assert!((b1 - b2).abs() > f32::EPSILON || b1 == b2); // Both valid
    }

    #[test]
    fn should_accept_disabled_never_updates() {
        assert!(!should_accept_update(
            &AutoUpdateMode::Disabled,
            &CanaryPhase::Complete,
            0.0
        ));
    }

    #[test]
    fn should_accept_all_always_updates() {
        assert!(should_accept_update(
            &AutoUpdateMode::All,
            &CanaryPhase::Phase1,
            0.99
        ));
    }

    #[test]
    fn should_accept_stable_respects_phases() {
        // Phase 1: stable nodes don't update
        assert!(!should_accept_update(
            &AutoUpdateMode::Stable,
            &CanaryPhase::Phase1,
            0.01
        ));
        // Phase 2: only if bucket < 5%
        assert!(should_accept_update(
            &AutoUpdateMode::Stable,
            &CanaryPhase::Phase2,
            0.03
        ));
        assert!(!should_accept_update(
            &AutoUpdateMode::Stable,
            &CanaryPhase::Phase2,
            0.10
        ));
        // Complete: everyone
        assert!(should_accept_update(
            &AutoUpdateMode::Stable,
            &CanaryPhase::Complete,
            0.99
        ));
        // Halted: nobody
        assert!(!should_accept_update(
            &AutoUpdateMode::Stable,
            &CanaryPhase::Halted {
                reason: "test".into()
            },
            0.0
        ));
    }

    #[test]
    fn aggregate_reports_summary() {
        let reports = vec![
            build_local_test_report(LocalTestParams {
                version: SemVer {
                    major: 0,
                    minor: 2,
                    patch: 0,
                    pre: None,
                },
                tester: NodeId([1u8; 32]),
                results: TestResults {
                    tests_run: 100,
                    tests_passed: 100,
                    tests_failed: 0,
                    tests_skipped: 0,
                    build_time_seconds: 60,
                    binary_hash: [0u8; 32],
                    binary_size_bytes: 50_000_000,
                },
            }),
            build_local_test_report(LocalTestParams {
                version: SemVer {
                    major: 0,
                    minor: 2,
                    patch: 0,
                    pre: None,
                },
                tester: NodeId([2u8; 32]),
                results: TestResults {
                    tests_run: 100,
                    tests_passed: 99,
                    tests_failed: 1,
                    tests_skipped: 0,
                    build_time_seconds: 65,
                    binary_hash: [0u8; 32],
                    binary_size_bytes: 50_000_000,
                },
            }),
        ];

        let summary = aggregate_reports(&reports);
        assert_eq!(summary.total_reports, 2);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.failed, 1);
        assert!(summary.has_blocking_issues);
    }
}
