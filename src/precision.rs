#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionMode {
    Fp32,
    Bf16,
    Fp16,
    Int8,
    Int4,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossScalingPolicy {
    pub initial_scale: f32,
    pub min_scale: f32,
    pub max_scale: f32,
    pub growth_factor: f32,
    pub backoff_factor: f32,
    pub growth_interval: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixedPrecisionPolicy {
    pub activations: PrecisionMode,
    pub weights: PrecisionMode,
    pub gradients: PrecisionMode,
    pub optimizer_state: PrecisionMode,
    pub use_master_weights: bool,
    pub loss_scaling: Option<LossScalingPolicy>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossScaler {
    scale: f32,
    stable_steps: u32,
    policy: LossScalingPolicy,
}

impl LossScaler {
    pub fn new(policy: LossScalingPolicy) -> Self {
        Self {
            scale: policy.initial_scale,
            stable_steps: 0,
            policy,
        }
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    pub fn unscale_gradients(&self, gradients: &mut [f32]) {
        if self.scale <= f32::EPSILON {
            return;
        }
        for grad in gradients {
            *grad /= self.scale;
        }
    }

    pub fn update(&mut self, overflow: bool) {
        if overflow {
            self.scale = (self.scale * self.policy.backoff_factor).max(self.policy.min_scale);
            self.stable_steps = 0;
            return;
        }

        self.stable_steps = self.stable_steps.saturating_add(1);
        if self.stable_steps >= self.policy.growth_interval {
            self.scale = (self.scale * self.policy.growth_factor).min(self.policy.max_scale);
            self.stable_steps = 0;
        }
    }
}

pub fn cast_values(values: &[f32], mode: PrecisionMode) -> Vec<f32> {
    match mode {
        PrecisionMode::Fp32 => values.to_vec(),
        PrecisionMode::Bf16 => values.iter().map(|v| bf16_round(*v)).collect(),
        PrecisionMode::Fp16 => values.iter().map(|v| fp16_like_round(*v)).collect(),
        PrecisionMode::Int8 => quantize_symmetric(values, 127.0),
        PrecisionMode::Int4 => quantize_symmetric(values, 7.0),
    }
}

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = bits & 0xFFFF_0000;
    f32::from_bits(rounded)
}

fn fp16_like_round(value: f32) -> f32 {
    (value * 1024.0).round() / 1024.0
}

fn quantize_symmetric(values: &[f32], max_q: f32) -> Vec<f32> {
    let max_abs = values
        .iter()
        .fold(0.0_f32, |acc, v| if v.abs() > acc { v.abs() } else { acc })
        .max(1e-8);
    let scale = max_q / max_abs;

    values
        .iter()
        .map(|value| {
            let q = (value * scale).round().clamp(-max_q, max_q);
            q / scale
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{cast_values, LossScaler, LossScalingPolicy, PrecisionMode};

    #[test]
    fn precision_cast_is_deterministic() {
        let values = vec![0.12345, -1.2345, 8.8888];
        let a = cast_values(&values, PrecisionMode::Bf16);
        let b = cast_values(&values, PrecisionMode::Bf16);
        assert_eq!(a, b);
    }

    #[test]
    fn loss_scaler_backoff_and_growth_work() {
        let mut scaler = LossScaler::new(LossScalingPolicy {
            initial_scale: 128.0,
            min_scale: 1.0,
            max_scale: 1024.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2,
        });

        scaler.update(true);
        assert_eq!(scaler.scale(), 64.0);

        scaler.update(false);
        scaler.update(false);
        assert_eq!(scaler.scale(), 128.0);
    }
}
