use std::collections::BTreeMap;

use crate::types::{ExpertKey, TierId};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StarvationConfig {
    pub min_assignments_per_expert: f32,
    pub probability_floor: f32,
    pub starvation_window_steps: u64,
    pub prune_threshold_steps: u64,
    pub exploration_boost: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierAssignmentStats {
    pub tier: TierId,
    pub assignments_per_step: u64,
    pub expert_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StarvationReport {
    pub update_density_by_tier: BTreeMap<TierId, f32>,
    pub starved_experts: Vec<ExpertKey>,
    pub no_update_probability_upper_bound: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StarvationIntervention {
    ForcedExploration { experts: Vec<ExpertKey>, boost: f32 },
    MergeExperts { experts: Vec<ExpertKey> },
    PruneExperts { experts: Vec<ExpertKey> },
}

pub fn no_update_probability_upper_bound(delta: f64, assignments_per_step: f64, steps: u64) -> f64 {
    let lambda = delta.max(0.0) * assignments_per_step.max(0.0) * steps as f64;
    (-lambda).exp()
}

pub fn analyze_starvation(
    tier_stats: &[TierAssignmentStats],
    expert_assignments_in_window: &BTreeMap<ExpertKey, u64>,
    cfg: StarvationConfig,
) -> StarvationReport {
    let mut update_density_by_tier = BTreeMap::new();
    let mut worst_case_probability = 0.0_f64;

    for stats in tier_stats {
        let density = if stats.expert_count == 0 {
            0.0
        } else {
            stats.assignments_per_step as f32 / stats.expert_count as f32
        };
        update_density_by_tier.insert(stats.tier, density);

        let bound = no_update_probability_upper_bound(
            f64::from(cfg.probability_floor),
            stats.assignments_per_step as f64,
            cfg.starvation_window_steps,
        );
        worst_case_probability = worst_case_probability.max(bound);
    }

    let mut starved = Vec::new();
    for (expert, assignments) in expert_assignments_in_window {
        if *assignments == 0 {
            starved.push(*expert);
        }
    }

    starved.sort();

    StarvationReport {
        update_density_by_tier,
        starved_experts: starved,
        no_update_probability_upper_bound: worst_case_probability,
    }
}

pub fn recommend_interventions(
    report: &StarvationReport,
    cfg: StarvationConfig,
) -> Vec<StarvationIntervention> {
    let mut actions = Vec::new();

    if !report.starved_experts.is_empty() {
        actions.push(StarvationIntervention::ForcedExploration {
            experts: report.starved_experts.clone(),
            boost: cfg.exploration_boost,
        });

        if cfg.starvation_window_steps >= cfg.prune_threshold_steps {
            actions.push(StarvationIntervention::PruneExperts {
                experts: report.starved_experts.clone(),
            });
        } else if report.starved_experts.len() >= 2 {
            actions.push(StarvationIntervention::MergeExperts {
                experts: report.starved_experts.clone(),
            });
        }
    }

    for density in report.update_density_by_tier.values() {
        if *density < cfg.min_assignments_per_expert {
            if !actions
                .iter()
                .any(|action| matches!(action, StarvationIntervention::ForcedExploration { .. }))
            {
                actions.push(StarvationIntervention::ForcedExploration {
                    experts: report.starved_experts.clone(),
                    boost: cfg.exploration_boost,
                });
            }
            break;
        }
    }

    actions
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        analyze_starvation, no_update_probability_upper_bound, recommend_interventions,
        StarvationConfig, TierAssignmentStats,
    };
    use crate::types::ExpertKey;

    fn config() -> StarvationConfig {
        StarvationConfig {
            min_assignments_per_expert: 1.0,
            probability_floor: 1e-3,
            starvation_window_steps: 100,
            prune_threshold_steps: 500,
            exploration_boost: 0.2,
        }
    }

    #[test]
    fn no_update_bound_decays_with_steps() {
        let short = no_update_probability_upper_bound(1e-3, 32.0, 10);
        let long = no_update_probability_upper_bound(1e-3, 32.0, 1_000);
        assert!(long < short);
    }

    #[test]
    fn starvation_analysis_detects_zero_assignment_experts() {
        let tier_stats = vec![TierAssignmentStats {
            tier: 1,
            assignments_per_step: 32,
            expert_count: 8,
        }];
        let mut assignments = BTreeMap::new();
        assignments.insert(ExpertKey::new(1, 0, 0), 100);
        assignments.insert(ExpertKey::new(1, 0, 1), 0);

        let report = analyze_starvation(&tier_stats, &assignments, config());
        assert_eq!(report.starved_experts, vec![ExpertKey::new(1, 0, 1)]);
    }

    #[test]
    fn interventions_include_forced_exploration_when_starved() {
        let tier_stats = vec![TierAssignmentStats {
            tier: 1,
            assignments_per_step: 1,
            expert_count: 64,
        }];

        let mut assignments = BTreeMap::new();
        assignments.insert(ExpertKey::new(1, 0, 0), 0);
        assignments.insert(ExpertKey::new(1, 0, 1), 0);

        let report = analyze_starvation(&tier_stats, &assignments, config());
        let actions = recommend_interventions(&report, config());

        assert!(!actions.is_empty());
    }
}
