#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadBalancingConfig {
    pub alpha_tier: f32,
    pub alpha_group: f32,
    pub alpha_expert: f32,
    pub probability_floor: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalRoutingUsage {
    pub tier_probs: Vec<f32>,
    pub group_probs: Vec<Vec<f32>>,
    pub expert_probs: Vec<Vec<Vec<f32>>>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LoadBalancingMetrics {
    pub tier_kl: f32,
    pub group_kl: f32,
    pub expert_kl: f32,
}

impl LoadBalancingMetrics {
    pub fn weighted_loss(self, cfg: LoadBalancingConfig) -> f32 {
        cfg.alpha_tier * self.tier_kl
            + cfg.alpha_group * self.group_kl
            + cfg.alpha_expert * self.expert_kl
    }
}

pub fn hierarchical_load_balancing_loss(
    usage: &HierarchicalRoutingUsage,
    cfg: LoadBalancingConfig,
) -> LoadBalancingMetrics {
    let tier = normalize_with_floor(&usage.tier_probs, cfg.probability_floor);
    let tier_kl = kl_to_uniform(&tier);

    let mut group_terms = Vec::new();
    for groups in &usage.group_probs {
        let p = normalize_with_floor(groups, cfg.probability_floor);
        group_terms.push(kl_to_uniform(&p));
    }

    let mut expert_terms = Vec::new();
    for tier_groups in &usage.expert_probs {
        for experts in tier_groups {
            let p = normalize_with_floor(experts, cfg.probability_floor);
            expert_terms.push(kl_to_uniform(&p));
        }
    }

    LoadBalancingMetrics {
        tier_kl,
        group_kl: average(&group_terms),
        expert_kl: average(&expert_terms),
    }
}

pub fn normalize_with_floor(values: &[f32], floor: f32) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }

    let floor = floor.max(0.0);
    let mut shifted: Vec<f32> = values.iter().map(|v| v.max(0.0) + floor).collect();
    let sum: f32 = shifted.iter().sum();

    if sum <= f32::EPSILON {
        let uniform = 1.0 / shifted.len() as f32;
        shifted.fill(uniform);
        return shifted;
    }

    for value in &mut shifted {
        *value /= sum;
    }

    shifted
}

pub fn kl_to_uniform(probabilities: &[f32]) -> f32 {
    if probabilities.is_empty() {
        return 0.0;
    }

    let n = probabilities.len() as f32;
    let uniform = 1.0 / n;
    let mut kl = 0.0_f32;

    for probability in probabilities {
        if *probability > 0.0 {
            kl += *probability * (*probability / uniform).ln();
        }
    }

    kl
}

fn average(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

#[cfg(test)]
mod tests {
    use super::{
        hierarchical_load_balancing_loss, kl_to_uniform, normalize_with_floor,
        HierarchicalRoutingUsage, LoadBalancingConfig,
    };

    #[test]
    fn kl_is_zero_for_uniform_distribution() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        assert!((kl_to_uniform(&p) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn floor_normalization_is_deterministic() {
        let values = vec![0.0, 0.0, 1.0];
        let a = normalize_with_floor(&values, 0.01);
        let b = normalize_with_floor(&values, 0.01);
        assert_eq!(a, b);
    }

    #[test]
    fn hierarchical_loss_computes_all_levels() {
        let usage = HierarchicalRoutingUsage {
            tier_probs: vec![0.8, 0.2],
            group_probs: vec![vec![0.7, 0.3], vec![0.1, 0.9]],
            expert_probs: vec![
                vec![vec![0.6, 0.2, 0.2], vec![0.1, 0.7, 0.2]],
                vec![vec![0.4, 0.4, 0.2], vec![0.8, 0.1, 0.1]],
            ],
        };

        let cfg = LoadBalancingConfig {
            alpha_tier: 1.0,
            alpha_group: 1.0,
            alpha_expert: 1.0,
            probability_floor: 1e-4,
        };

        let metrics = hierarchical_load_balancing_loss(&usage, cfg);
        assert!(metrics.tier_kl > 0.0);
        assert!(metrics.group_kl > 0.0);
        assert!(metrics.expert_kl > 0.0);
        assert!(metrics.weighted_loss(cfg) > 0.0);
    }
}
