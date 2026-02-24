use crate::error::SwarmError;
use crate::storage::db::Database;
use crate::types::{
    ChangelogEntry, ChangelogItem, Proposal, ProposalCategory, ProposalStatus, SemVer,
};

const TREE_CHANGELOG: &str = "changelog";

/// Auto-generate a changelog from accepted proposals included in a release.
pub fn generate_changelog(proposals: &[Proposal], version: SemVer) -> ChangelogEntry {
    let mut entries: Vec<ChangelogItem> = proposals
        .iter()
        .filter(|p| {
            matches!(
                p.status,
                ProposalStatus::Accepted | ProposalStatus::Implemented | ProposalStatus::Released
            )
        })
        .map(|p| ChangelogItem {
            category: p.category.clone(),
            title: p.title.clone(),
            proposal_hash: p.hash,
            author: p.author.clone(),
        })
        .collect();

    // Sort by category for readability
    entries.sort_by(|a, b| category_sort_key(&a.category).cmp(&category_sort_key(&b.category)));

    ChangelogEntry {
        version,
        date: chrono::Utc::now(),
        entries,
        signature: vec![],
    }
}

/// Format a changelog entry as human-readable markdown.
pub fn format_changelog_markdown(entry: &ChangelogEntry) -> String {
    let mut md = String::new();
    md.push_str(&format!(
        "## {} ({})\n\n",
        entry.version,
        entry.date.format("%Y-%m-%d")
    ));

    let mut current_category: Option<&ProposalCategory> = None;

    for item in &entry.entries {
        if current_category != Some(&item.category) {
            current_category = Some(&item.category);
            md.push_str(&format!("\n### {}\n\n", category_label(&item.category)));
        }
        md.push_str(&format!("- {} (by `{}`)\n", item.title, item.author));
    }

    md
}

/// Store a changelog entry in the database.
pub fn store_changelog(db: &Database, entry: &ChangelogEntry) -> Result<(), SwarmError> {
    let key = entry.version.to_key();
    db.put_json(TREE_CHANGELOG, &key, entry)?;
    Ok(())
}

/// Get a changelog entry by version.
pub fn get_changelog(
    db: &Database,
    version: &SemVer,
) -> Result<Option<ChangelogEntry>, SwarmError> {
    let key = version.to_key();
    db.get_json(TREE_CHANGELOG, &key)
}

/// Get all changelogs, newest first.
pub fn list_changelogs(db: &Database) -> Result<Vec<ChangelogEntry>, SwarmError> {
    let tree = db.tree(TREE_CHANGELOG)?;
    let mut entries = Vec::new();
    for entry in tree.iter() {
        let (_, value) = entry.map_err(SwarmError::Database)?;
        if let Ok(changelog) = serde_json::from_slice::<ChangelogEntry>(&value) {
            entries.push(changelog);
        }
    }
    // Sort newest first
    entries.sort_by(|a, b| b.date.cmp(&a.date));
    Ok(entries)
}

fn category_sort_key(cat: &ProposalCategory) -> u8 {
    match cat {
        ProposalCategory::CodeChange => 0,
        ProposalCategory::ProtocolChange => 1,
        ProposalCategory::GovernanceChange => 2,
        ProposalCategory::ModelAddition => 3,
        ProposalCategory::ModelDeprecation => 4,
        ProposalCategory::ParameterTuning => 5,
    }
}

fn category_label(cat: &ProposalCategory) -> &'static str {
    match cat {
        ProposalCategory::CodeChange => "Code Changes",
        ProposalCategory::ProtocolChange => "Protocol Changes",
        ProposalCategory::GovernanceChange => "Governance Changes",
        ProposalCategory::ModelAddition => "New Models",
        ProposalCategory::ModelDeprecation => "Deprecated Models",
        ProposalCategory::ParameterTuning => "Parameter Tuning",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NodeId;

    fn make_proposal(title: &str, category: ProposalCategory) -> Proposal {
        Proposal {
            hash: [0u8; 32],
            author: NodeId([1u8; 32]),
            title: title.into(),
            body: "Description".into(),
            category,
            status: ProposalStatus::Accepted,
            linked_issues: vec![],
            created_at: chrono::Utc::now(),
            voting_deadline: chrono::Utc::now(),
            signature: vec![],
            patch: None,
        }
    }

    #[test]
    fn generate_changelog_from_proposals() {
        let proposals = vec![
            make_proposal("Fix memory leak", ProposalCategory::CodeChange),
            make_proposal("Add Llama 4 support", ProposalCategory::ModelAddition),
            make_proposal("Upgrade to libp2p 0.55", ProposalCategory::ProtocolChange),
        ];

        let version = SemVer {
            major: 0,
            minor: 2,
            patch: 0,
            pre: None,
        };

        let entry = generate_changelog(&proposals, version);
        assert_eq!(entry.entries.len(), 3);
        // Sorted by category: code change first, then protocol, then model
        assert_eq!(entry.entries[0].title, "Fix memory leak");
        assert_eq!(entry.entries[1].title, "Upgrade to libp2p 0.55");
        assert_eq!(entry.entries[2].title, "Add Llama 4 support");
    }

    #[test]
    fn format_markdown() {
        let proposals = vec![make_proposal(
            "Fix crash on startup",
            ProposalCategory::CodeChange,
        )];

        let version = SemVer {
            major: 1,
            minor: 0,
            patch: 0,
            pre: None,
        };

        let entry = generate_changelog(&proposals, version);
        let md = format_changelog_markdown(&entry);
        assert!(md.contains("## 1.0.0"));
        assert!(md.contains("### Code Changes"));
        assert!(md.contains("Fix crash on startup"));
    }

    #[test]
    fn store_and_retrieve_changelog() {
        let dir = tempfile::tempdir().unwrap();
        let db = crate::storage::db::Database::open(dir.path()).unwrap();

        let version = SemVer {
            major: 0,
            minor: 2,
            patch: 0,
            pre: None,
        };
        let entry = generate_changelog(&[], version.clone());
        store_changelog(&db, &entry).unwrap();

        let fetched = get_changelog(&db, &version).unwrap().unwrap();
        assert_eq!(fetched.entries.len(), 0);
    }
}
