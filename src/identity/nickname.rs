use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::error::SwarmError;
use crate::storage::db::Database;
// Re-export NicknameRecord from swarmllm-types crate
pub use crate::types::NicknameRecord;
use crate::types::NodeId;

/// redb tree key for nickname records.
pub const TREE_NICKNAMES: &str = "nicknames";
/// redb tree key for local identity preferences.
pub const TREE_IDENTITY_PREFS: &str = "identity_prefs";

/// Extension methods for NicknameRecord (defined in swarmllm-types crate).
pub trait NicknameRecordExt {
    fn signing_payload(
        node_id: &NodeId,
        nickname: &str,
        timestamp: &chrono::DateTime<chrono::Utc>,
    ) -> Vec<u8>;
    fn new_signed(
        identity: &crate::identity::Identity,
        nickname: String,
    ) -> Result<NicknameRecord, SwarmError>;
    fn verify(&self) -> Result<(), SwarmError>;
}

impl NicknameRecordExt for NicknameRecord {
    /// Build the signing payload: `"swarmllm-nickname|{hex_node_id}|{nickname}|{timestamp}"`
    fn signing_payload(
        node_id: &NodeId,
        nickname: &str,
        timestamp: &chrono::DateTime<chrono::Utc>,
    ) -> Vec<u8> {
        format!(
            "swarmllm-nickname|{}|{}|{}",
            hex::encode(node_id.0),
            nickname,
            timestamp.to_rfc3339()
        )
        .into_bytes()
    }

    /// Create a new signed nickname record using the node's identity.
    fn new_signed(
        identity: &crate::identity::Identity,
        nickname: String,
    ) -> Result<NicknameRecord, SwarmError> {
        validate_nickname(&nickname)?;
        let timestamp = chrono::Utc::now();
        let payload = Self::signing_payload(identity.node_id(), &nickname, &timestamp);
        let signature = identity.sign(&payload);
        tracing::debug!(
            node = %identity.node_id(),
            nickname = %nickname,
            "DIAG: nickname record created"
        );
        Ok(Self {
            node_id: identity.node_id().clone(),
            nickname,
            timestamp,
            signature,
        })
    }

    /// Verify the Ed25519 signature on this record.
    /// SEC-I3: Also rejects records older than 24 hours to prevent stale replay.
    fn verify(&self) -> Result<(), SwarmError> {
        // Timestamp freshness check: reject records older than 24 hours
        // (matches the gossip dispatcher's age filter in dispatch.rs)
        let age = chrono::Utc::now() - self.timestamp;
        if age > chrono::Duration::hours(24) {
            return Err(SwarmError::InvalidNickname(
                "Nickname record is stale (older than 24 hours)".into(),
            ));
        }
        // Also reject records with timestamps in the future (clock skew tolerance: 5 min)
        if self.timestamp > chrono::Utc::now() + chrono::Duration::minutes(5) {
            return Err(SwarmError::InvalidNickname(
                "Nickname record has future timestamp".into(),
            ));
        }

        let payload = Self::signing_payload(&self.node_id, &self.nickname, &self.timestamp);
        let vk = VerifyingKey::from_bytes(&self.node_id.0)
            .map_err(|e| SwarmError::InvalidNickname(format!("Invalid public key: {e}")))?;
        let sig_bytes: [u8; 64] = self
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| SwarmError::InvalidNickname("Signature must be 64 bytes".into()))?;
        let sig = Signature::from_bytes(&sig_bytes);
        vk.verify(&payload, &sig)
            .map_err(|_| SwarmError::InvalidSignature)
    }
}

/// How a node presents itself on the network.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum VisibilityMode {
    #[default]
    Anonymous,
    Nickname,
}

/// Local identity preferences stored in redb.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdentityPrefs {
    pub nickname: Option<String>,
    pub visibility: VisibilityMode,
}

impl Default for IdentityPrefs {
    fn default() -> Self {
        Self {
            nickname: None,
            visibility: VisibilityMode::Anonymous,
        }
    }
}

/// Persistence layer for nickname records and identity prefs.
pub struct NicknameStore {
    db: Database,
}

impl NicknameStore {
    pub fn new(db: Database) -> Self {
        Self { db }
    }

    /// Save a nickname record to the database.
    pub fn put_record(&self, record: &NicknameRecord) -> Result<(), SwarmError> {
        let key = hex::encode(record.node_id.0);
        self.db.put_json(TREE_NICKNAMES, &key, record)
    }

    /// Remove a nickname record from the database.
    pub fn remove_record(&self, node_id: &NodeId) -> Result<(), SwarmError> {
        let key = hex::encode(node_id.0);
        self.db.remove(TREE_NICKNAMES, &key)
    }

    /// Load all persisted nickname records.
    pub fn load_all(&self) -> Result<Vec<NicknameRecord>, SwarmError> {
        self.db.iter_json(TREE_NICKNAMES)
    }

    /// Save local identity preferences.
    pub fn put_prefs(&self, prefs: &IdentityPrefs) -> Result<(), SwarmError> {
        self.db.put_json(TREE_IDENTITY_PREFS, "local", prefs)
    }

    /// Load local identity preferences.
    pub fn get_prefs(&self) -> Result<IdentityPrefs, SwarmError> {
        Ok(self
            .db
            .get_json::<IdentityPrefs>(TREE_IDENTITY_PREFS, "local")?
            .unwrap_or_default())
    }
}

/// Validate a nickname: 1-32 chars, `[a-zA-Z0-9_-]` only.
pub fn validate_nickname(nickname: &str) -> Result<(), SwarmError> {
    if nickname.is_empty() || nickname.len() > 32 {
        return Err(SwarmError::InvalidNickname(
            "Nickname must be 1-32 characters".into(),
        ));
    }
    if !nickname
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(SwarmError::InvalidNickname(
            "Nickname may only contain [a-zA-Z0-9_-]".into(),
        ));
    }
    Ok(())
}

/// Resolve a node's display name. If the nickname collides with another node,
/// append `#ab12` (first 4 hex chars of node_id) as a disambiguator.
pub fn display_name(
    node_id: &NodeId,
    registry: &dashmap::DashMap<NodeId, NicknameRecord>,
) -> String {
    let record = match registry.get(node_id) {
        Some(r) => r.clone(),
        None => return format!("{node_id}"),
    };
    let nick = &record.nickname;

    // Check for collision: another node_id with the same nickname
    let collision = registry
        .iter()
        .any(|entry| entry.key() != node_id && entry.value().nickname == *nick);

    if collision {
        let suffix = &hex::encode(&node_id.0[..2]); // 4 hex chars
        format!("{nick}#{suffix}")
    } else {
        nick.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::Identity;

    #[test]
    fn validate_nickname_valid() {
        assert!(validate_nickname("alice").is_ok());
        assert!(validate_nickname("Bob_42").is_ok());
        assert!(validate_nickname("a-b-c").is_ok());
        assert!(validate_nickname("A").is_ok());
        // 32 chars max
        let long = "a".repeat(32);
        assert!(validate_nickname(&long).is_ok());
    }

    #[test]
    fn validate_nickname_invalid() {
        assert!(validate_nickname("").is_err());
        assert!(validate_nickname(&"x".repeat(33)).is_err());
        assert!(validate_nickname("hello world").is_err());
        assert!(validate_nickname("nick@name").is_err());
        assert!(validate_nickname("a.b").is_err());
    }

    #[test]
    fn sign_and_verify_record() {
        let id = Identity::generate();
        let record = NicknameRecord::new_signed(&id, "alice".into()).unwrap();
        assert!(record.verify().is_ok());
    }

    #[test]
    fn tampered_record_fails_verify() {
        let id = Identity::generate();
        let mut record = NicknameRecord::new_signed(&id, "alice".into()).unwrap();
        record.nickname = "bob".into(); // tamper
        assert!(record.verify().is_err());
    }

    #[test]
    fn wrong_signer_fails_verify() {
        let id1 = Identity::generate();
        let id2 = Identity::generate();
        let mut record = NicknameRecord::new_signed(&id1, "alice".into()).unwrap();
        record.node_id = id2.node_id().clone(); // wrong identity
        assert!(record.verify().is_err());
    }

    #[test]
    fn display_name_anonymous() {
        let id = Identity::generate();
        let registry = dashmap::DashMap::new();
        let name = display_name(id.node_id(), &registry);
        assert_eq!(name, format!("{}", id.node_id()));
    }

    #[test]
    fn display_name_no_collision() {
        let id = Identity::generate();
        let registry = dashmap::DashMap::new();
        let record = NicknameRecord::new_signed(&id, "alice".into()).unwrap();
        registry.insert(id.node_id().clone(), record);
        assert_eq!(display_name(id.node_id(), &registry), "alice");
    }

    #[test]
    fn display_name_collision_appends_suffix() {
        let id1 = Identity::generate();
        let id2 = Identity::generate();
        let registry = dashmap::DashMap::new();

        let r1 = NicknameRecord::new_signed(&id1, "alice".into()).unwrap();
        let mut r2 = NicknameRecord::new_signed(&id2, "alice".into()).unwrap();
        // Force same nickname for collision test
        r2.nickname = "alice".into();

        registry.insert(id1.node_id().clone(), r1);
        registry.insert(id2.node_id().clone(), r2);

        let name1 = display_name(id1.node_id(), &registry);
        let name2 = display_name(id2.node_id(), &registry);

        // Both should have disambiguator suffix
        assert!(name1.starts_with("alice#"), "got: {name1}");
        assert!(name2.starts_with("alice#"), "got: {name2}");
        // Suffixes should differ
        assert_ne!(name1, name2);
    }

    #[test]
    fn nickname_store_roundtrip() {
        let db = Database::open_temp().unwrap();
        let store = NicknameStore::new(db);

        let id = Identity::generate();
        let record = NicknameRecord::new_signed(&id, "bob".into()).unwrap();
        store.put_record(&record).unwrap();

        let all = store.load_all().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].nickname, "bob");
    }

    #[test]
    fn nickname_store_remove() {
        let db = Database::open_temp().unwrap();
        let store = NicknameStore::new(db);

        let id = Identity::generate();
        let record = NicknameRecord::new_signed(&id, "charlie".into()).unwrap();
        store.put_record(&record).unwrap();
        store.remove_record(id.node_id()).unwrap();

        let all = store.load_all().unwrap();
        assert!(all.is_empty());
    }

    #[test]
    fn identity_prefs_roundtrip() {
        let db = Database::open_temp().unwrap();
        let store = NicknameStore::new(db);

        let prefs = IdentityPrefs {
            nickname: Some("delta".into()),
            visibility: VisibilityMode::Nickname,
        };
        store.put_prefs(&prefs).unwrap();

        let loaded = store.get_prefs().unwrap();
        assert_eq!(loaded.nickname.as_deref(), Some("delta"));
        assert_eq!(loaded.visibility, VisibilityMode::Nickname);
    }

    #[test]
    fn identity_prefs_defaults() {
        let db = Database::open_temp().unwrap();
        let store = NicknameStore::new(db);
        let prefs = store.get_prefs().unwrap();
        assert!(prefs.nickname.is_none());
        assert_eq!(prefs.visibility, VisibilityMode::Anonymous);
    }
}
