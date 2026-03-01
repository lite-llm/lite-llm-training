#[derive(Debug, Clone)]
pub struct ReplayEvent {
    pub step: u64,
    pub seed: u128,
    pub description: String,
}

#[derive(Debug, Clone, Default)]
pub struct ReplayContext {
    pub checkpoint_id: String,
    pub events: Vec<ReplayEvent>,
}

impl ReplayContext {
    pub fn is_replayable(&self) -> bool {
        !self.checkpoint_id.is_empty() && !self.events.is_empty()
    }
}

