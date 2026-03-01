use crate::TierId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpansionPhase {
    Preparation,
    Isolation,
    Integration,
    JointTraining,
    Complete,
}

#[derive(Debug, Clone)]
pub struct TierExpansionPlan {
    pub new_tier: TierId,
    pub start_step: u64,
    pub end_step: u64,
    pub phase: ExpansionPhase,
}

impl TierExpansionPlan {
    pub fn advance(&mut self) {
        self.phase = match self.phase {
            ExpansionPhase::Preparation => ExpansionPhase::Isolation,
            ExpansionPhase::Isolation => ExpansionPhase::Integration,
            ExpansionPhase::Integration => ExpansionPhase::JointTraining,
            ExpansionPhase::JointTraining => ExpansionPhase::Complete,
            ExpansionPhase::Complete => ExpansionPhase::Complete,
        }
    }
}
