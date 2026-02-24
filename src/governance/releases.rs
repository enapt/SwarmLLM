use std::path::{Path, PathBuf};

use crate::error::SwarmError;
use crate::storage::db::Database;
use crate::types::{
    Blake3Hash, CanaryPhase, GovernanceParams, GovernanceRole, ReleaseApproval, ReleaseCandidate,
    SemVer, TestReport, TestStatus,
};

const TREE_RELEASES: &str = "releases";
const TREE_APPROVALS: &str = "release_approvals";
const TREE_REPORTS: &str = "test_reports";

/// Publish a release candidate to the database.
pub fn publish_release_candidate(
    db: &Database,
    candidate: &ReleaseCandidate,
    role: &GovernanceRole,
) -> Result<(), SwarmError> {
    if !role.can_create_proposals() {
        return Err(SwarmError::InsufficientPermissions {
            action: "publish release candidate".into(),
            required_role: "Contributor".into(),
        });
    }

    if candidate.binaries.is_empty() {
        return Err(SwarmError::Governance(
            "Release candidate must include at least one binary".into(),
        ));
    }

    let key = candidate.version.to_key();
    db.put_json(TREE_RELEASES, &key, candidate)?;

    tracing::info!(version = %candidate.version, "Published release candidate");
    Ok(())
}

/// Get a release candidate by version.
pub fn get_release(
    db: &Database,
    version: &SemVer,
) -> Result<Option<ReleaseCandidate>, SwarmError> {
    let key = version.to_key();
    db.get_json(TREE_RELEASES, &key)
}

/// List all release candidates.
pub fn list_releases(db: &Database) -> Result<Vec<ReleaseCandidate>, SwarmError> {
    let tree = db.tree(TREE_RELEASES)?;
    let mut releases = Vec::new();
    for entry in tree.iter() {
        let (_, value) = entry.map_err(SwarmError::Database)?;
        if let Ok(rc) = serde_json::from_slice::<ReleaseCandidate>(&value) {
            releases.push(rc);
        }
    }
    Ok(releases)
}

/// Submit an approval for a release (Maintainer+ only).
pub fn submit_approval(db: &Database, approval: &ReleaseApproval) -> Result<u32, SwarmError> {
    if !approval.role.can_approve_releases() {
        return Err(SwarmError::InsufficientPermissions {
            action: "approve release".into(),
            required_role: "Maintainer".into(),
        });
    }

    // Verify the release exists
    let release_key = approval.release_version.to_key();
    let _release: ReleaseCandidate = db
        .get_json(TREE_RELEASES, &release_key)?
        .ok_or_else(|| SwarmError::ReleaseNotFound(release_key.clone()))?;

    // Store approval (deduplicate by approver)
    let key = format!("{}/{}", release_key, hex::encode(&approval.approver.0[..8]));
    db.put_json(TREE_APPROVALS, &key, approval)?;

    // Count approvals
    let count = count_approvals(db, &approval.release_version)?;

    tracing::info!(
        version = %approval.release_version,
        approver = %approval.approver,
        total_approvals = count,
        "Release approval submitted"
    );

    Ok(count)
}

/// Count the number of valid approvals for a release.
pub fn count_approvals(db: &Database, version: &SemVer) -> Result<u32, SwarmError> {
    let tree = db.tree(TREE_APPROVALS)?;
    let prefix = version.to_key();
    Ok(tree.scan_prefix(prefix.as_bytes()).count() as u32)
}

/// Get all approvals for a release.
pub fn get_approvals(db: &Database, version: &SemVer) -> Result<Vec<ReleaseApproval>, SwarmError> {
    let tree = db.tree(TREE_APPROVALS)?;
    let prefix = version.to_key();
    let mut approvals = Vec::new();
    for entry in tree.scan_prefix(prefix.as_bytes()) {
        let (_, value) = entry.map_err(SwarmError::Database)?;
        if let Ok(approval) = serde_json::from_slice::<ReleaseApproval>(&value) {
            approvals.push(approval);
        }
    }
    Ok(approvals)
}

/// Check if a release has met the approval threshold.
pub fn is_release_approved(
    db: &Database,
    version: &SemVer,
    params: &GovernanceParams,
) -> Result<bool, SwarmError> {
    let count = count_approvals(db, version)?;
    Ok(count >= params.release_approval_threshold)
}

/// Submit a test report for a release.
pub fn submit_test_report(db: &Database, report: &TestReport) -> Result<(), SwarmError> {
    // Verify the release exists
    let release_key = report.release_version.to_key();
    let _release: ReleaseCandidate = db
        .get_json(TREE_RELEASES, &release_key)?
        .ok_or_else(|| SwarmError::ReleaseNotFound(release_key.clone()))?;

    let key = format!("{}/{}", release_key, hex::encode(&report.tester.0[..8]));
    db.put_json(TREE_REPORTS, &key, report)?;

    tracing::info!(
        version = %report.release_version,
        tester = %report.tester,
        status = ?report.status,
        "Test report submitted"
    );

    Ok(())
}

/// Get all test reports for a release.
pub fn get_test_reports(db: &Database, version: &SemVer) -> Result<Vec<TestReport>, SwarmError> {
    let tree = db.tree(TREE_REPORTS)?;
    let prefix = version.to_key();
    let mut reports = Vec::new();
    for entry in tree.scan_prefix(prefix.as_bytes()) {
        let (_, value) = entry.map_err(SwarmError::Database)?;
        if let Ok(report) = serde_json::from_slice::<TestReport>(&value) {
            reports.push(report);
        }
    }
    Ok(reports)
}

/// Check if any test reports indicate failures (blocks canary advancement).
pub fn has_blocking_reports(reports: &[TestReport]) -> bool {
    reports.iter().any(|r| {
        matches!(
            r.status,
            TestStatus::Failed { .. } | TestStatus::HashMismatch { .. }
        )
    })
}

/// Determine the current canary rollout phase for a release.
pub fn determine_canary_phase(
    release: &ReleaseCandidate,
    reports: &[TestReport],
    params: &GovernanceParams,
) -> CanaryPhase {
    // If any blocking reports exist, halt
    if has_blocking_reports(reports) {
        let failures: Vec<String> = reports
            .iter()
            .filter_map(|r| match &r.status {
                TestStatus::Failed { failures } => Some(failures.join(", ")),
                TestStatus::HashMismatch { expected, actual } => {
                    Some(format!("hash mismatch: expected {expected}, got {actual}"))
                }
                _ => None,
            })
            .collect();
        return CanaryPhase::Halted {
            reason: failures.join("; "),
        };
    }

    let now = chrono::Utc::now();
    let age_days = (now - release.created_at).num_days();

    let phase1_end = params.canary_phase1_days as i64;
    let phase2_end = phase1_end + 4; // 3-7 days
    let phase3_end = phase2_end + 3; // 7-10 days

    if age_days < phase1_end {
        CanaryPhase::Phase1
    } else if age_days < phase2_end {
        CanaryPhase::Phase2
    } else if age_days < phase3_end {
        CanaryPhase::Phase3
    } else {
        CanaryPhase::Complete
    }
}

/// Update manager: checks for updates, downloads, verifies, and applies them.
pub struct UpdateManager {
    pub current_version: SemVer,
    pub update_dir: PathBuf,
    pub auto_restart: bool,
    pub keep_versions: u32,
}

impl UpdateManager {
    pub fn new(
        current_version: SemVer,
        data_dir: &Path,
        auto_restart: bool,
        keep_versions: u32,
    ) -> Self {
        let update_dir = data_dir.join("updates");
        Self {
            current_version,
            update_dir,
            auto_restart,
            keep_versions,
        }
    }

    /// Check if a release candidate is newer than the current version.
    pub fn is_newer(&self, candidate: &ReleaseCandidate) -> bool {
        let cv = &self.current_version;
        let nv = &candidate.version;
        (nv.major, nv.minor, nv.patch) > (cv.major, cv.minor, cv.patch)
    }

    /// Verify that a downloaded binary matches the release candidate hash.
    pub fn verify_binary(
        binary_path: &Path,
        expected_hash: &Blake3Hash,
    ) -> Result<bool, SwarmError> {
        let data = std::fs::read(binary_path).map_err(SwarmError::Io)?;
        let actual_hash = blake3::hash(&data);
        Ok(actual_hash.as_bytes() == expected_hash)
    }

    /// Get the path where a specific version's binary would be stored.
    pub fn version_path(&self, version: &SemVer) -> PathBuf {
        self.update_dir.join(version.to_key())
    }

    /// Clean up old versions, keeping only the N most recent.
    pub fn cleanup_old_versions(&self) -> Result<(), SwarmError> {
        if !self.update_dir.exists() {
            return Ok(());
        }

        let mut versions: Vec<(String, std::time::SystemTime)> = Vec::new();
        let entries = std::fs::read_dir(&self.update_dir).map_err(SwarmError::Io)?;
        for entry in entries {
            let entry = entry.map_err(SwarmError::Io)?;
            if entry.path().is_dir() {
                if let Ok(meta) = entry.metadata() {
                    if let Ok(modified) = meta.modified() {
                        versions.push((entry.file_name().to_string_lossy().to_string(), modified));
                    }
                }
            }
        }

        // Sort by modified time, newest first
        versions.sort_by(|a, b| b.1.cmp(&a.1));

        // Remove versions beyond the keep limit
        for (name, _) in versions.iter().skip(self.keep_versions as usize) {
            let path = self.update_dir.join(name);
            tracing::info!(version = %name, "Removing old version");
            std::fs::remove_dir_all(&path).map_err(SwarmError::Io)?;
        }

        Ok(())
    }
}

/// Post-update health check. Returns Ok if the daemon started successfully.
pub async fn post_update_health_check(port: u16) -> Result<(), SwarmError> {
    // Check if API server responds within 30 seconds
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);

    loop {
        if tokio::time::Instant::now() > deadline {
            return Err(SwarmError::Internal(
                "Post-update health check timed out after 30s".into(),
            ));
        }

        // Try to connect to the health endpoint
        match tokio::net::TcpStream::connect(format!("127.0.0.1:{port}")).await {
            Ok(_) => {
                tracing::info!("Post-update health check passed");
                return Ok(());
            }
            Err(_) => {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }
    }
}

/// Check if we're in the genesis period (relaxed governance thresholds).
pub fn is_genesis_period(network_stats: &crate::types::NetworkStats) -> bool {
    // Hard expiry: June 1, 2027
    let genesis_expiry = chrono::DateTime::parse_from_rfc3339("2027-06-01T00:00:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    let sufficient_nodes = network_stats.nodes_with_30d_uptime >= 50;

    chrono::Utc::now() < genesis_expiry && !sufficient_nodes
}

/// Get relaxed governance params for the genesis period.
pub fn genesis_params() -> GovernanceParams {
    GovernanceParams {
        code_change_quorum_pct: 0.05,
        protocol_change_quorum_pct: 0.05,
        governance_change_quorum_pct: 0.05,
        release_approval_threshold: 1,
        ..GovernanceParams::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Architecture, NodeId, Platform, ReleaseBinary, TestResults};

    fn test_db() -> Database {
        let dir = tempfile::tempdir().unwrap();
        Database::open(dir.path()).unwrap()
    }

    fn test_version() -> SemVer {
        SemVer {
            major: 0,
            minor: 2,
            patch: 0,
            pre: Some("rc.1".into()),
        }
    }

    fn test_release() -> ReleaseCandidate {
        ReleaseCandidate {
            version: test_version(),
            builder: NodeId([1u8; 32]),
            created_at: chrono::Utc::now(),
            changelog: "Fixed bugs".into(),
            included_proposals: vec![],
            binaries: vec![ReleaseBinary {
                platform: Platform::Linux,
                arch: Architecture::X86_64,
                hash: [0u8; 32],
                size_bytes: 50_000_000,
                shard_manifest: [0u8; 32],
            }],
            source_hash: [0u8; 32],
            signature: vec![],
        }
    }

    #[test]
    fn publish_and_get_release() {
        let db = test_db();
        let rc = test_release();
        publish_release_candidate(&db, &rc, &GovernanceRole::Contributor).unwrap();

        let fetched = get_release(&db, &rc.version).unwrap().unwrap();
        assert_eq!(fetched.changelog, "Fixed bugs");
    }

    #[test]
    fn member_cannot_publish_release() {
        let db = test_db();
        let rc = test_release();
        let result = publish_release_candidate(&db, &rc, &GovernanceRole::Member);
        assert!(result.is_err());
    }

    #[test]
    fn approval_threshold_check() {
        let db = test_db();
        let rc = test_release();
        publish_release_candidate(&db, &rc, &GovernanceRole::Contributor).unwrap();

        let mut params = GovernanceParams::default();
        params.release_approval_threshold = 2;

        // One approval is not enough
        let approval1 = ReleaseApproval {
            release_version: test_version(),
            approver: NodeId([10u8; 32]),
            role: GovernanceRole::Maintainer,
            binary_hashes_verified: true,
            test_suite_passed: true,
            timestamp: chrono::Utc::now(),
            signature: vec![],
        };
        submit_approval(&db, &approval1).unwrap();
        assert!(!is_release_approved(&db, &test_version(), &params).unwrap());

        // Two approvals meet the threshold
        let approval2 = ReleaseApproval {
            release_version: test_version(),
            approver: NodeId([20u8; 32]),
            role: GovernanceRole::Maintainer,
            binary_hashes_verified: true,
            test_suite_passed: true,
            timestamp: chrono::Utc::now(),
            signature: vec![],
        };
        submit_approval(&db, &approval2).unwrap();
        assert!(is_release_approved(&db, &test_version(), &params).unwrap());
    }

    #[test]
    fn blocking_reports_halt_canary() {
        let rc = test_release();
        let params = GovernanceParams::default();

        let report = TestReport {
            release_version: test_version(),
            tester: NodeId([5u8; 32]),
            platform: Platform::Linux,
            architecture: Architecture::X86_64,
            gpu: None,
            status: TestStatus::Failed {
                failures: vec!["test_network".into()],
            },
            results: TestResults {
                tests_run: 10,
                tests_passed: 9,
                tests_failed: 1,
                tests_skipped: 0,
                build_time_seconds: 120,
                binary_hash: [0u8; 32],
                binary_size_bytes: 50_000_000,
            },
            timestamp: chrono::Utc::now(),
            signature: vec![],
        };

        let phase = determine_canary_phase(&rc, &[report], &params);
        assert!(matches!(phase, CanaryPhase::Halted { .. }));
    }

    #[test]
    fn genesis_period_detection() {
        let stats = crate::types::NetworkStats {
            total_active_vote_weight: 1000,
            nodes_with_30d_uptime: 10, // Less than 50
            total_active_nodes: 20,
        };
        assert!(is_genesis_period(&stats));

        let mature_stats = crate::types::NetworkStats {
            total_active_vote_weight: 100_000,
            nodes_with_30d_uptime: 100,
            total_active_nodes: 500,
        };
        assert!(!is_genesis_period(&mature_stats));
    }

    #[test]
    fn semver_display() {
        let v = SemVer {
            major: 0,
            minor: 2,
            patch: 0,
            pre: Some("rc.1".into()),
        };
        assert_eq!(format!("{v}"), "0.2.0-rc.1");

        let stable = SemVer {
            major: 1,
            minor: 0,
            patch: 0,
            pre: None,
        };
        assert_eq!(format!("{stable}"), "1.0.0");
    }

    #[test]
    fn update_manager_version_comparison() {
        let dir = tempfile::tempdir().unwrap();
        let manager = UpdateManager::new(
            SemVer {
                major: 0,
                minor: 1,
                patch: 0,
                pre: None,
            },
            dir.path(),
            true,
            3,
        );

        let newer = test_release();
        assert!(manager.is_newer(&newer));

        let mut older = test_release();
        older.version = SemVer {
            major: 0,
            minor: 0,
            patch: 9,
            pre: None,
        };
        assert!(!manager.is_newer(&older));
    }
}
