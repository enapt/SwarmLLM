use serde::Deserialize;

#[derive(Deserialize)]
pub(super) struct HfRepoInfo {
    pub id: String,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
}

#[derive(Deserialize)]
pub(super) struct HfModelDetail {
    pub siblings: Option<Vec<HfFileInfo>>,
}

#[derive(Deserialize, Clone)]
pub(super) struct HfFileInfo {
    pub rfilename: String,
    pub size: Option<u64>,
}
