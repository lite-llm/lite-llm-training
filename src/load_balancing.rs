#[derive(Debug, Clone, Copy)]
pub struct LoadBalancingConfig {
    pub alpha_tier: f32,
    pub alpha_group: f32,
    pub alpha_expert: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LoadBalancingMetrics {
    pub tier_kl: f32,
    pub group_kl: f32,
    pub expert_kl: f32,
}

impl LoadBalancingMetrics {
    pub fn weighted_loss(self, cfg: LoadBalancingConfig) -> f32 {
        (cfg.alpha_tier * self.tier_kl)
            + (cfg.alpha_group * self.group_kl)
            + (cfg.alpha_expert * self.expert_kl)
    }
}

