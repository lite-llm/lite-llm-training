use std::collections::BTreeMap;

use crate::error::{TrainingError, TrainingResult};
use crate::optimizer::{Optimizer, Tensor};
use crate::types::fnv64_hex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccumulationConfig {
    pub micro_batch_size: u32,
    pub accumulation_steps: u32,
    pub data_parallel_ranks: u32,
    pub scale_learning_rate: bool,
}

impl AccumulationConfig {
    pub fn validate(self) -> TrainingResult<Self> {
        if self.micro_batch_size == 0 {
            return Err(TrainingError::InvalidConfig(
                "micro_batch_size must be greater than zero",
            ));
        }
        if self.accumulation_steps == 0 {
            return Err(TrainingError::InvalidConfig(
                "accumulation_steps must be greater than zero",
            ));
        }
        if self.data_parallel_ranks == 0 {
            return Err(TrainingError::InvalidConfig(
                "data_parallel_ranks must be greater than zero",
            ));
        }
        Ok(self)
    }

    pub fn effective_batch_size(self) -> u64 {
        u64::from(self.micro_batch_size)
            * u64::from(self.accumulation_steps)
            * u64::from(self.data_parallel_ranks)
    }

    pub fn scaled_learning_rate(self, base_lr: f32) -> f32 {
        if self.scale_learning_rate {
            base_lr * self.accumulation_steps as f32 * self.data_parallel_ranks as f32
        } else {
            base_lr
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MicroBatchSchedule {
    base_seed: u64,
    accumulation_steps: u32,
    update_step: u64,
    micro_batch_index: u32,
}

impl MicroBatchSchedule {
    pub fn new(
        base_seed: u64,
        start_update_step: u64,
        accumulation_steps: u32,
    ) -> TrainingResult<Self> {
        if accumulation_steps == 0 {
            return Err(TrainingError::InvalidConfig(
                "accumulation_steps must be greater than zero",
            ));
        }

        Ok(Self {
            base_seed,
            accumulation_steps,
            update_step: start_update_step,
            micro_batch_index: 0,
        })
    }

    pub fn update_step(&self) -> u64 {
        self.update_step
    }

    pub fn micro_batch_index(&self) -> u32 {
        self.micro_batch_index
    }

    pub fn current_seed(&self) -> u64 {
        seed_for_micro_batch(self.base_seed, self.update_step, self.micro_batch_index)
    }

    pub fn advance(&mut self) {
        self.micro_batch_index += 1;
        if self.micro_batch_index >= self.accumulation_steps {
            self.micro_batch_index = 0;
            self.update_step = self.update_step.saturating_add(1);
        }
    }
}

pub fn seed_for_micro_batch(base_seed: u64, update_step: u64, micro_batch_index: u32) -> u64 {
    let canonical = format!("{base_seed}|{update_step}|{micro_batch_index}");
    let hash = fnv64_hex(canonical.as_bytes());
    u64::from_str_radix(&hash, 16).unwrap_or(0)
}

#[derive(Debug, Clone)]
pub struct AccumulationState {
    config: AccumulationConfig,
    processed_micro_batches: u32,
    gradient_sums: BTreeMap<u64, Tensor>,
}

impl AccumulationState {
    pub fn new(config: AccumulationConfig) -> TrainingResult<Self> {
        Ok(Self {
            config: config.validate()?,
            processed_micro_batches: 0,
            gradient_sums: BTreeMap::new(),
        })
    }

    pub fn config(&self) -> AccumulationConfig {
        self.config
    }

    pub fn processed_micro_batches(&self) -> u32 {
        self.processed_micro_batches
    }

    pub fn is_update_ready(&self) -> bool {
        self.processed_micro_batches == self.config.accumulation_steps
    }

    pub fn reset_cycle(&mut self) {
        self.processed_micro_batches = 0;
        for tensor in self.gradient_sums.values_mut() {
            for value in &mut tensor.data {
                *value = 0.0;
            }
        }
    }

    pub fn accumulate_micro_batch(
        &mut self,
        micro_batch_index: u32,
        gradients: &BTreeMap<u64, Tensor>,
    ) -> TrainingResult<()> {
        if micro_batch_index != self.processed_micro_batches {
            return Err(TrainingError::InvalidInput(
                "micro-batch index must be contiguous and deterministic",
            ));
        }
        if gradients.is_empty() {
            return Err(TrainingError::InvalidInput(
                "micro-batch gradients must not be empty",
            ));
        }
        if self.processed_micro_batches >= self.config.accumulation_steps {
            return Err(TrainingError::InvalidState(
                "accumulation window is already full; consume gradients before adding more",
            ));
        }

        if self.gradient_sums.is_empty() {
            for (param_id, grad) in gradients {
                self.gradient_sums
                    .insert(*param_id, Tensor::zeros(&grad.shape, grad.dtype));
            }
        }

        if self.gradient_sums.len() != gradients.len() {
            return Err(TrainingError::InvalidInput(
                "gradient parameter set changed within accumulation window",
            ));
        }

        for (param_id, sum) in &mut self.gradient_sums {
            let grad = gradients.get(param_id).ok_or(TrainingError::InvalidInput(
                "missing gradient for parameter in accumulation window",
            ))?;

            if sum.shape != grad.shape || sum.dtype != grad.dtype || sum.len() != grad.len() {
                return Err(TrainingError::InvalidInput(
                    "gradient tensor metadata changed within accumulation window",
                ));
            }

            for idx in 0..sum.data.len() {
                sum.data[idx] += grad.data[idx];
            }
        }

        self.processed_micro_batches += 1;
        Ok(())
    }

    pub fn take_mean_gradients(&mut self) -> TrainingResult<BTreeMap<u64, Tensor>> {
        if !self.is_update_ready() {
            return Err(TrainingError::InvalidState(
                "cannot consume gradients before accumulation window is complete",
            ));
        }

        let mut result = BTreeMap::new();
        let scale = self.config.accumulation_steps as f32;

        for (param_id, sum) in &self.gradient_sums {
            let mut mean = sum.clone();
            for value in &mut mean.data {
                *value /= scale;
            }
            result.insert(*param_id, mean);
        }

        self.reset_cycle();
        Ok(result)
    }
}

pub fn apply_updates_deterministic<O: Optimizer>(
    optimizer: &mut O,
    params: &mut BTreeMap<u64, Tensor>,
    mean_gradients: &BTreeMap<u64, Tensor>,
    update_step: usize,
    learning_rate: f32,
) -> TrainingResult<()> {
    if params.len() != mean_gradients.len() {
        return Err(TrainingError::InvalidInput(
            "parameter and gradient maps must have identical keys",
        ));
    }

    for (param_id, param) in params {
        let grad = mean_gradients
            .get(param_id)
            .ok_or(TrainingError::InvalidInput(
                "missing gradient for parameter during optimizer update",
            ))?;

        optimizer.update(*param_id, param, grad, update_step, learning_rate)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        apply_updates_deterministic, seed_for_micro_batch, AccumulationConfig, AccumulationState,
        MicroBatchSchedule,
    };
    use crate::optimizer::{DType, Optimizer, SgdMomentum, Tensor};

    fn gradients() -> BTreeMap<u64, Tensor> {
        let mut map = BTreeMap::new();
        map.insert(
            1,
            Tensor::new(&[2], DType::Fp32, vec![1.0, 2.0]).expect("valid gradient"),
        );
        map.insert(
            2,
            Tensor::new(&[2], DType::Fp32, vec![3.0, 4.0]).expect("valid gradient"),
        );
        map
    }

    #[test]
    fn effective_batch_size_and_scaled_lr_follow_spec() {
        let cfg = AccumulationConfig {
            micro_batch_size: 8,
            accumulation_steps: 4,
            data_parallel_ranks: 2,
            scale_learning_rate: true,
        }
        .validate()
        .expect("config must be valid");

        assert_eq!(cfg.effective_batch_size(), 64);
        assert!((cfg.scaled_learning_rate(1e-3) - 8e-3).abs() < 1e-8);
    }

    #[test]
    fn micro_batch_seed_is_deterministic() {
        let a = seed_for_micro_batch(42, 10, 1);
        let b = seed_for_micro_batch(42, 10, 1);
        let c = seed_for_micro_batch(42, 10, 2);
        assert_eq!(a, b);
        assert_ne!(a, c);

        let mut schedule = MicroBatchSchedule::new(42, 10, 2).expect("schedule should initialize");
        let first = schedule.current_seed();
        schedule.advance();
        let second = schedule.current_seed();
        schedule.advance();
        let third = schedule.current_seed();

        assert_ne!(first, second);
        assert_ne!(second, third);
        assert_eq!(schedule.update_step(), 11);
    }

    #[test]
    fn accumulation_requires_contiguous_micro_batch_order() {
        let mut state = AccumulationState::new(AccumulationConfig {
            micro_batch_size: 2,
            accumulation_steps: 2,
            data_parallel_ranks: 1,
            scale_learning_rate: false,
        })
        .expect("state should initialize");

        let grads = gradients();
        let err = state.accumulate_micro_batch(1, &grads);
        assert!(err.is_err());

        state
            .accumulate_micro_batch(0, &grads)
            .expect("first micro-batch should be accepted");
        state
            .accumulate_micro_batch(1, &grads)
            .expect("second micro-batch should be accepted");

        assert!(state.is_update_ready());
    }

    #[test]
    fn mean_gradients_and_deterministic_update_are_stable() {
        let mut state = AccumulationState::new(AccumulationConfig {
            micro_batch_size: 2,
            accumulation_steps: 2,
            data_parallel_ranks: 1,
            scale_learning_rate: false,
        })
        .expect("state should initialize");

        let grads_a = gradients();
        let grads_b = gradients();

        state
            .accumulate_micro_batch(0, &grads_a)
            .expect("first micro-batch should accumulate");
        state
            .accumulate_micro_batch(1, &grads_b)
            .expect("second micro-batch should accumulate");

        let mean = state
            .take_mean_gradients()
            .expect("mean gradients should be available");
        assert_eq!(state.processed_micro_batches(), 0);

        let mut params = BTreeMap::new();
        params.insert(
            1,
            Tensor::new(&[2], DType::Fp32, vec![10.0, 10.0]).expect("valid param"),
        );
        params.insert(
            2,
            Tensor::new(&[2], DType::Fp32, vec![20.0, 20.0]).expect("valid param"),
        );

        let mut params_ref = params.clone();
        let mut opt_a = SgdMomentum::new(0.0);
        let mut opt_b = SgdMomentum::new(0.0);

        apply_updates_deterministic(&mut opt_a, &mut params, &mean, 1, 0.1)
            .expect("update should succeed");
        apply_updates_deterministic(&mut opt_b, &mut params_ref, &mean, 1, 0.1)
            .expect("update should succeed");

        assert_eq!(params, params_ref);
        assert!(opt_a.export_state_shards(1).is_ok());
    }
}
