use crate::error::SwarmError;
use crate::storage::db::Database;
use crate::types::{
    Blake3Hash, GovernanceParams, GovernanceRole, Issue, IssueCategory, IssueComment,
    IssueSeverity, IssueStatus, IssueStatusChange, IssueUpvote, NodeId,
};

const TREE_ISSUES: &str = "issues";
const TREE_COMMENTS: &str = "issue_comments";
const TREE_UPVOTES: &str = "issue_upvotes";

const MAX_TITLE_LEN: usize = 200;
const MAX_BODY_LEN: usize = 10_000;
const MAX_COMMENT_LEN: usize = 5_000;
const MAX_TAGS: usize = 5;
const MAX_TAG_LEN: usize = 30;

/// Parameters for creating a new issue.
pub struct CreateIssueParams {
    pub author: NodeId,
    pub title: String,
    pub body: String,
    pub category: IssueCategory,
    pub severity: Option<IssueSeverity>,
    pub tags: Vec<String>,
    pub credit_balance: i64,
}

/// Create a new issue. Any node with positive credit balance can create issues.
pub fn create_issue(
    db: &Database,
    p: CreateIssueParams,
    params: &GovernanceParams,
) -> Result<Issue, SwarmError> {
    let CreateIssueParams {
        author,
        title,
        body,
        category,
        severity,
        tags,
        credit_balance,
    } = p;
    // Credit balance check
    if credit_balance < params.min_credit_balance_for_issues {
        return Err(SwarmError::Governance(
            "Positive credit balance required to file issues".into(),
        ));
    }

    // Validate fields
    if title.is_empty() || title.len() > MAX_TITLE_LEN {
        return Err(SwarmError::Governance(format!(
            "Title must be 1-{MAX_TITLE_LEN} characters"
        )));
    }
    if body.is_empty() || body.len() > MAX_BODY_LEN {
        return Err(SwarmError::Governance(format!(
            "Body must be 1-{MAX_BODY_LEN} characters"
        )));
    }
    if tags.len() > MAX_TAGS {
        return Err(SwarmError::Governance(format!(
            "Maximum {MAX_TAGS} tags allowed"
        )));
    }
    for tag in &tags {
        if tag.len() > MAX_TAG_LEN {
            return Err(SwarmError::Governance(format!(
                "Tags must be at most {MAX_TAG_LEN} characters"
            )));
        }
    }

    // Severity is only valid for Bug category
    if severity.is_some() && category != IssueCategory::Bug {
        return Err(SwarmError::Governance(
            "Severity is only applicable to bug reports".into(),
        ));
    }

    let now = chrono::Utc::now();
    let hash = compute_issue_hash(&author, &title, &body, &now);

    let issue = Issue {
        hash,
        author,
        title,
        body,
        category,
        severity,
        status: IssueStatus::Open,
        created_at: now,
        upvotes: 0,
        tags,
        signature: vec![],
    };

    let key = hex::encode(issue.hash);
    db.put_json(TREE_ISSUES, &key, &issue)?;

    Ok(issue)
}

/// Get an issue by hash.
pub fn get_issue(db: &Database, hash: &Blake3Hash) -> Result<Option<Issue>, SwarmError> {
    let key = hex::encode(hash);
    db.get_json(TREE_ISSUES, &key)
}

/// List all issues.
pub fn list_issues(db: &Database) -> Result<Vec<Issue>, SwarmError> {
    let tree = db.tree(TREE_ISSUES)?;
    let mut issues = Vec::new();
    for entry in tree.iter() {
        let (_, value) = entry.map_err(SwarmError::Database)?;
        if let Ok(issue) = serde_json::from_slice::<Issue>(&value) {
            issues.push(issue);
        }
    }
    Ok(issues)
}

/// List issues filtered by status.
pub fn list_issues_by_status(
    db: &Database,
    status: &IssueStatus,
) -> Result<Vec<Issue>, SwarmError> {
    let all = list_issues(db)?;
    Ok(all.into_iter().filter(|i| &i.status == status).collect())
}

/// Add a comment to an issue.
pub fn add_comment(db: &Database, comment: &IssueComment) -> Result<(), SwarmError> {
    // Validate comment body
    if comment.body.is_empty() || comment.body.len() > MAX_COMMENT_LEN {
        return Err(SwarmError::Governance(format!(
            "Comment must be 1-{MAX_COMMENT_LEN} characters"
        )));
    }

    // Verify the issue exists
    let issue_key = hex::encode(comment.issue_hash);
    let _issue: Issue = db
        .get_json(TREE_ISSUES, &issue_key)?
        .ok_or_else(|| SwarmError::IssueNotFound(issue_key))?;

    // Store comment (key: issue_hash/timestamp for ordering)
    let key = format!(
        "{}/{}",
        hex::encode(comment.issue_hash),
        comment.created_at.timestamp_millis()
    );
    db.put_json(TREE_COMMENTS, &key, comment)?;

    Ok(())
}

/// Get all comments for an issue.
pub fn get_comments(
    db: &Database,
    issue_hash: &Blake3Hash,
) -> Result<Vec<IssueComment>, SwarmError> {
    let tree = db.tree(TREE_COMMENTS)?;
    let prefix = hex::encode(issue_hash);
    let mut comments = Vec::new();

    for entry in tree.scan_prefix(prefix.as_bytes()) {
        let (_, value) = entry.map_err(SwarmError::Database)?;
        if let Ok(comment) = serde_json::from_slice::<IssueComment>(&value) {
            comments.push(comment);
        }
    }

    Ok(comments)
}

/// Upvote an issue (contribution-weighted).
pub fn upvote_issue(db: &Database, upvote: &IssueUpvote) -> Result<Issue, SwarmError> {
    let issue_key = hex::encode(upvote.issue_hash);
    let mut issue: Issue = db
        .get_json(TREE_ISSUES, &issue_key)?
        .ok_or_else(|| SwarmError::IssueNotFound(issue_key.clone()))?;

    // Store upvote (deduplicate by voter)
    let upvote_key = format!(
        "{}/{}",
        hex::encode(upvote.issue_hash),
        hex::encode(&upvote.voter.0[..8])
    );
    db.put_json(TREE_UPVOTES, &upvote_key, upvote)?;

    // Recount upvotes
    let tree = db.tree(TREE_UPVOTES)?;
    let prefix = hex::encode(upvote.issue_hash);
    let count = tree.scan_prefix(prefix.as_bytes()).count() as u32;
    issue.upvotes = count;

    db.put_json(TREE_ISSUES, &issue_key, &issue)?;
    Ok(issue)
}

/// Change an issue's status (Contributor+ required for Acknowledged).
pub fn change_status(
    db: &Database,
    change: &IssueStatusChange,
    role: &GovernanceRole,
) -> Result<Issue, SwarmError> {
    let issue_key = hex::encode(change.issue_hash);
    let mut issue: Issue = db
        .get_json(TREE_ISSUES, &issue_key)?
        .ok_or_else(|| SwarmError::IssueNotFound(issue_key.clone()))?;

    // Acknowledged requires Contributor+ role
    if change.new_status == IssueStatus::Acknowledged && !role.can_create_proposals() {
        return Err(SwarmError::InsufficientPermissions {
            action: "acknowledge issue".into(),
            required_role: "Contributor".into(),
        });
    }

    issue.status = change.new_status.clone();
    db.put_json(TREE_ISSUES, &issue_key, &issue)?;
    Ok(issue)
}

/// Auto-close stale issues (no activity for N days).
pub fn auto_close_stale(db: &Database, params: &GovernanceParams) -> Result<u32, SwarmError> {
    let cutoff = chrono::Utc::now() - chrono::Duration::days(params.issue_auto_close_days as i64);
    let issues = list_issues(db)?;
    let mut closed = 0u32;

    for issue in issues {
        if matches!(issue.status, IssueStatus::Open | IssueStatus::Acknowledged)
            && issue.created_at < cutoff
        {
            let key = hex::encode(issue.hash);
            let mut updated = issue;
            updated.status = IssueStatus::Closed;
            db.put_json(TREE_ISSUES, &key, &updated)?;
            closed += 1;
        }
    }

    if closed > 0 {
        tracing::info!(count = closed, "Auto-closed stale issues");
    }

    Ok(closed)
}

fn compute_issue_hash(
    author: &NodeId,
    title: &str,
    body: &str,
    created_at: &chrono::DateTime<chrono::Utc>,
) -> Blake3Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&author.0);
    hasher.update(title.as_bytes());
    hasher.update(body.as_bytes());
    hasher.update(created_at.to_rfc3339().as_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GovernanceParams;

    fn test_db() -> Database {
        let dir = tempfile::tempdir().unwrap();
        Database::open(dir.path()).unwrap()
    }

    fn test_node() -> NodeId {
        NodeId([1u8; 32])
    }

    fn make_issue(
        title: &str,
        cat: IssueCategory,
        sev: Option<IssueSeverity>,
        balance: i64,
    ) -> (Database, CreateIssueParams) {
        let db = test_db();
        let p = CreateIssueParams {
            author: test_node(),
            title: title.into(),
            body: "Issue body text".into(),
            category: cat,
            severity: sev,
            tags: vec![],
            credit_balance: balance,
        };
        (db, p)
    }

    #[test]
    fn create_and_get_issue() {
        let (db, mut p) = make_issue(
            "Bug: Pipeline crashes",
            IssueCategory::Bug,
            Some(IssueSeverity::High),
            100,
        );
        p.body = "Steps to reproduce...".into();
        p.tags = vec!["pipeline".into()];
        let issue = create_issue(&db, p, &GovernanceParams::default()).unwrap();

        assert_eq!(issue.status, IssueStatus::Open);
        assert_eq!(issue.upvotes, 0);

        let fetched = get_issue(&db, &issue.hash).unwrap().unwrap();
        assert_eq!(fetched.title, "Bug: Pipeline crashes");
    }

    #[test]
    fn negative_balance_cannot_create_issue() {
        let (db, p) = make_issue("Spam issue", IssueCategory::FeatureRequest, None, -10);
        let result = create_issue(&db, p, &GovernanceParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn severity_only_for_bugs() {
        let (db, p) = make_issue(
            "Feature",
            IssueCategory::FeatureRequest,
            Some(IssueSeverity::High),
            100,
        );
        let result = create_issue(&db, p, &GovernanceParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn upvote_increments_count() {
        let (db, p) = make_issue("Feature request", IssueCategory::FeatureRequest, None, 100);
        let issue = create_issue(&db, p, &GovernanceParams::default()).unwrap();

        let upvote = IssueUpvote {
            issue_hash: issue.hash,
            voter: NodeId([2u8; 32]),
            weight: 50,
            signature: vec![],
        };
        let updated = upvote_issue(&db, &upvote).unwrap();
        assert_eq!(updated.upvotes, 1);
    }

    #[test]
    fn add_and_get_comments() {
        let (db, p) = make_issue(
            "Test issue",
            IssueCategory::Bug,
            Some(IssueSeverity::Low),
            100,
        );
        let issue = create_issue(&db, p, &GovernanceParams::default()).unwrap();

        let comment = IssueComment {
            issue_hash: issue.hash,
            author: NodeId([2u8; 32]),
            body: "I can reproduce this.".into(),
            created_at: chrono::Utc::now(),
            signature: vec![],
        };
        add_comment(&db, &comment).unwrap();

        let comments = get_comments(&db, &issue.hash).unwrap();
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].body, "I can reproduce this.");
    }
}
