use crate::types::TierId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpansionPhase {
    Preparation,
    Isolation,
    Integration,
    JointTraining,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrationSchedule {
    Linear,
    Cosine,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpansionDirective {
    pub phase: ExpansionPhase,
    pub route_to_new_tier_probability: f32,
    pub freeze_existing_tiers: bool,
    pub freeze_existing_router_heads: bool,
    pub train_new_tier_only: bool,
    pub unfreeze_all_tiers: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpansionWindow {
    pub start_step: u64,
    pub preparation_steps: u64,
    pub isolation_steps: u64,
    pub integration_steps: u64,
    pub joint_training_steps: u64,
}

impl ExpansionWindow {
    pub fn end_step(self) -> u64 {
        self.start_step
            .saturating_add(self.preparation_steps)
            .saturating_add(self.isolation_steps)
            .saturating_add(self.integration_steps)
            .saturating_add(self.joint_training_steps)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TierExpansionPlan {
    pub new_tier: TierId,
    pub window: ExpansionWindow,
    pub integration_schedule: IntegrationSchedule,
    pub deterministic_seed: u64,
}

impl TierExpansionPlan {
    pub fn phase_at(self, step: u64) -> ExpansionPhase {
        if step < self.window.start_step {
            return ExpansionPhase::Preparation;
        }

        let prep_end = self.window.start_step + self.window.preparation_steps;
        if step < prep_end {
            return ExpansionPhase::Preparation;
        }

        let isolation_end = prep_end + self.window.isolation_steps;
        if step < isolation_end {
            return ExpansionPhase::Isolation;
        }

        let integration_end = isolation_end + self.window.integration_steps;
        if step < integration_end {
            return ExpansionPhase::Integration;
        }

        let joint_end = integration_end + self.window.joint_training_steps;
        if step < joint_end {
            return ExpansionPhase::JointTraining;
        }

        ExpansionPhase::Complete
    }

    pub fn integration_progress(self, step: u64) -> f32 {
        let prep_end = self.window.start_step + self.window.preparation_steps;
        let isolation_end = prep_end + self.window.isolation_steps;
        let integration_end = isolation_end + self.window.integration_steps;

        if self.window.integration_steps == 0 {
            return if step >= isolation_end { 1.0 } else { 0.0 };
        }

        if step < isolation_end {
            return 0.0;
        }
        if step >= integration_end {
            return 1.0;
        }

        let progress = (step - isolation_end) as f32 / self.window.integration_steps as f32;
        progress.clamp(0.0, 1.0)
    }

    pub fn route_probability(self, step: u64) -> f32 {
        match self.phase_at(step) {
            ExpansionPhase::Preparation => 0.0,
            ExpansionPhase::Isolation => 0.05,
            ExpansionPhase::Integration => {
                let p = self.integration_progress(step);
                match self.integration_schedule {
                    IntegrationSchedule::Linear => 0.05 + 0.95 * p,
                    IntegrationSchedule::Cosine => {
                        let cosine = 0.5 * (1.0 - (std::f32::consts::PI * p).cos());
                        0.05 + 0.95 * cosine
                    }
                }
            }
            ExpansionPhase::JointTraining | ExpansionPhase::Complete => 1.0,
        }
    }

    pub fn directive_for(self, step: u64) -> ExpansionDirective {
        let phase = self.phase_at(step);
        ExpansionDirective {
            phase,
            route_to_new_tier_probability: self.route_probability(step),
            freeze_existing_tiers: matches!(phase, ExpansionPhase::Isolation),
            freeze_existing_router_heads: matches!(phase, ExpansionPhase::Isolation),
            train_new_tier_only: matches!(phase, ExpansionPhase::Isolation),
            unfreeze_all_tiers: matches!(
                phase,
                ExpansionPhase::JointTraining | ExpansionPhase::Complete
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CurriculumScheduler {
    plan: TierExpansionPlan,
    current_step: u64,
}

impl CurriculumScheduler {
    pub fn new(plan: TierExpansionPlan, start_step: u64) -> Self {
        Self {
            plan,
            current_step: start_step,
        }
    }

    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    pub fn plan(&self) -> TierExpansionPlan {
        self.plan
    }

    pub fn peek(&self) -> ExpansionDirective {
        self.plan.directive_for(self.current_step)
    }

    pub fn advance(&mut self) -> ExpansionDirective {
        let directive = self.plan.directive_for(self.current_step);
        self.current_step = self.current_step.saturating_add(1);
        directive
    }

    pub fn seek(&mut self, step: u64) {
        self.current_step = step;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CurriculumScheduler, ExpansionPhase, ExpansionWindow, IntegrationSchedule,
        TierExpansionPlan,
    };

    fn sample_plan(schedule: IntegrationSchedule) -> TierExpansionPlan {
        TierExpansionPlan {
            new_tier: 100,
            window: ExpansionWindow {
                start_step: 10,
                preparation_steps: 2,
                isolation_steps: 3,
                integration_steps: 4,
                joint_training_steps: 2,
            },
            integration_schedule: schedule,
            deterministic_seed: 42,
        }
    }

    #[test]
    fn scheduler_phase_progression_matches_spec() {
        let plan = sample_plan(IntegrationSchedule::Linear);
        assert_eq!(plan.phase_at(9), ExpansionPhase::Preparation);
        assert_eq!(plan.phase_at(12), ExpansionPhase::Isolation);
        assert_eq!(plan.phase_at(15), ExpansionPhase::Integration);
        assert_eq!(plan.phase_at(19), ExpansionPhase::JointTraining);
        assert_eq!(plan.phase_at(22), ExpansionPhase::Complete);
    }

    #[test]
    fn integration_probability_is_deterministic() {
        let plan = sample_plan(IntegrationSchedule::Cosine);
        let p1 = plan.route_probability(16);
        let p2 = plan.route_probability(16);
        assert_eq!(p1, p2);
        assert!(p1 > 0.05);
        assert!(p1 < 1.0);
    }

    #[test]
    fn scheduler_replay_is_stable() {
        let plan = sample_plan(IntegrationSchedule::Linear);
        let mut first = CurriculumScheduler::new(plan, 10);
        let mut second = CurriculumScheduler::new(plan, 10);

        let mut trace_a = Vec::new();
        let mut trace_b = Vec::new();

        for _ in 0..16 {
            trace_a.push(first.advance());
            trace_b.push(second.advance());
        }

        assert_eq!(trace_a, trace_b);
    }
}
