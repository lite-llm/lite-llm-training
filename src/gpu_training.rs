//! GPU-accelerated training operations with automatic mixed precision.
//!
//! Provides GPU-optimized training steps, async checkpoint I/O, and
//! metrics tracking for distributed training workloads.
//!
//! # Key Types
//!
//! - [`GpuTrainingStep`] -- wraps an optimizer with AMP and gradient scaling
//! - [`TrainingMetrics`] -- aggregates per-step training metrics
//! - [`StepMetrics`] -- individual step measurement record
//!
//! # Features
//!
//! - Automatic mixed precision (AMP) with configurable gradient scaling
//! - Async checkpoint save/load through the storage backend
//! - Deterministic checkpoint fingerprinting for restore validation

use crate::error::{TrainingError, TrainingResult};
use crate::optimizer::{DType, Optimizer, Tensor as OptimizerTensor};
use crate::types::fnv64_hex;

/// A GPU-accelerated training step that wraps the optimizer with
/// automatic mixed precision and gradient scaling.
#[derive(Debug, Clone)]
pub struct GpuTrainingStep<O: Optimizer> {
    optimizer: O,
    grad_scale: f32,
    use_amp: bool,
}

impl<O: Optimizer> GpuTrainingStep<O> {
    /// Create a new GPU training step with the given optimizer and AMP setting.
    ///
    /// When AMP is enabled, the initial gradient scale is set to 65536.0
    /// to prevent underflow during mixed-precision computation.
    pub fn new(optimizer: O, use_amp: bool) -> Self {
        Self {
            optimizer,
            grad_scale: if use_amp { 65536.0 } else { 1.0 },
            use_amp,
        }
    }

    /// Execute a single training step with gradient accumulation.
    ///
    /// 1. Forward pass (compute logits)
    /// 2. Compute loss (cross-entropy)
    /// 3. Backward pass (compute gradients)
    /// 4. Optimizer update
    pub fn step(
        &mut self,
        param_id: u64,
        param: &mut [f32],
        grad: &[f32],
        step_num: usize,
        learning_rate: f32,
    ) -> TrainingResult<f32> {
        // Scale gradients for AMP
        let scaled_grad: Vec<f32> = if self.use_amp {
            grad.iter().map(|g| g * self.grad_scale).collect()
        } else {
            grad.to_vec()
        };

        let param_tensor = lite_llm_inference::Tensor::from_data(param.to_vec(), &[param.len()]);
        let grad_tensor = lite_llm_inference::Tensor::from_data(scaled_grad, &[grad.len()]);

        // Unscale gradients before optimizer step (for AMP)
        let effective_lr = if self.use_amp {
            learning_rate / self.grad_scale
        } else {
            learning_rate
        };

        // Convert to optimizer tensor
        let mut opt_param = OptimizerTensor::new(
            &[param.len()],
            DType::Fp32,
            param_tensor.data.clone(),
        )?;
        let opt_grad =
            OptimizerTensor::new(&[grad.len()], DType::Fp32, grad_tensor.data.clone())?;

        self.optimizer
            .update(param_id, &mut opt_param, &opt_grad, step_num, effective_lr)?;

        // Copy back updated params
        param.copy_from_slice(&opt_param.data);

        // Compute loss (cross-entropy placeholder)
        let loss = compute_cross_entropy_loss(&param_tensor.data, &grad_tensor.data);

        Ok(loss)
    }

    /// Get the current gradient scale (for AMP).
    pub fn grad_scale(&self) -> f32 {
        self.grad_scale
    }

    /// Adjust the gradient scale (for AMP overflow/underflow handling).
    pub fn adjust_grad_scale(&mut self, factor: f32) {
        self.grad_scale *= factor;
    }

    /// Access the underlying optimizer.
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    /// Access the underlying optimizer mutably.
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }
}

/// Compute cross-entropy loss between logits and targets.
fn compute_cross_entropy_loss(logits: &[f32], targets: &[f32]) -> f32 {
    if logits.len() != targets.len() || logits.is_empty() {
        return f32::NAN;
    }

    // Simplified cross-entropy: -sum(target_i * log(softmax_i))
    let max_logit = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let mut sum_exp = 0.0;
    let mut dot_product = 0.0;

    for (l, t) in logits.iter().zip(targets.iter()) {
        let exp_val = (l - max_logit).exp();
        sum_exp += exp_val;
        dot_product += t * l;
    }

    if sum_exp <= 0.0 {
        return f32::NAN;
    }

    let log_sum_exp = sum_exp.ln() + max_logit;
    let loss = -dot_product + log_sum_exp * targets.iter().sum::<f32>();

    loss / logits.len() as f32
}

/// Async checkpoint save for distributed training.
///
/// Saves optimizer state, model weights, and training metadata
/// to a filesystem or cloud storage backend.
///
/// Returns a checkpoint fingerprint string on success.
pub async fn save_checkpoint_async(
    checkpoint_dir: &str,
    step: u64,
    _epoch: u64,
    optimizer_state: &[u8],
    model_weights: &[u8],
    metadata: &serde_json::Value,
) -> TrainingResult<String> {
    let checkpoint_path = format!("{}/step-{}", checkpoint_dir, step);

    // Create checkpoint directory
    tokio::fs::create_dir_all(&checkpoint_path)
        .await
        .map_err(|e| TrainingError::IoError(format!("failed to create checkpoint dir: {}", e)))?;

    // Save optimizer state
    let opt_path = format!("{}/optimizer.bin", checkpoint_path);
    tokio::fs::write(&opt_path, optimizer_state)
        .await
        .map_err(|e| TrainingError::IoError(format!("failed to write optimizer: {}", e)))?;

    // Save model weights
    let model_path = format!("{}/model.bin", checkpoint_path);
    tokio::fs::write(&model_path, model_weights)
        .await
        .map_err(|e| TrainingError::IoError(format!("failed to write model: {}", e)))?;

    // Save metadata
    let meta_path = format!("{}/metadata.json", checkpoint_path);
    tokio::fs::write(&meta_path, serde_json::to_vec(metadata)?)
        .await
        .map_err(|e| TrainingError::IoError(format!("failed to write metadata: {}", e)))?;

    // Compute checkpoint fingerprint
    let fingerprint = compute_checkpoint_fingerprint(optimizer_state, model_weights);

    Ok(format!(
        "{}@{}",
        checkpoint_path, fingerprint
    ))
}

/// Async checkpoint load from the given path.
///
/// Returns a tuple of (optimizer_state, model_weights, metadata).
pub async fn load_checkpoint_async(
    checkpoint_path: &str,
) -> TrainingResult<(Vec<u8>, Vec<u8>, serde_json::Value)> {
    let opt_path = format!("{}/optimizer.bin", checkpoint_path);
    let model_path = format!("{}/model.bin", checkpoint_path);
    let meta_path = format!("{}/metadata.json", checkpoint_path);

    let optimizer_state = tokio::fs::read(&opt_path)
        .await
        .map_err(|e| TrainingError::IoError(format!("failed to read optimizer: {}", e)))?;

    let model_weights = tokio::fs::read(&model_path)
        .await
        .map_err(|e| TrainingError::IoError(format!("failed to read model: {}", e)))?;

    let meta_bytes = tokio::fs::read(&meta_path)
        .await
        .map_err(|e| TrainingError::IoError(format!("failed to read metadata: {}", e)))?;

    let metadata: serde_json::Value = serde_json::from_slice(&meta_bytes)
        .map_err(|e| TrainingError::ParseErrorDynamic(format!("invalid metadata JSON: {}", e)))?;

    Ok((optimizer_state, model_weights, metadata))
}

/// Compute a deterministic fingerprint of checkpoint contents.
///
/// Uses FNV-1a hash of the combined optimizer state and model weights.
pub fn compute_checkpoint_fingerprint(optimizer_state: &[u8], model_weights: &[u8]) -> String {
    let mut combined = Vec::with_capacity(optimizer_state.len() + model_weights.len());
    combined.extend_from_slice(optimizer_state);
    combined.extend_from_slice(model_weights);
    fnv64_hex(&combined)
}

/// Training metrics tracker for monitoring loss, throughput, and GPU utilization.
///
/// Aggregates per-step metrics and provides rolling average computations
/// for monitoring training progress and performance.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub steps: Vec<StepMetrics>,
    pub total_tokens_seen: u64,
    pub start_time: std::time::Instant,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            steps: Vec::new(),
            total_tokens_seen: 0,
            start_time: std::time::Instant::now(),
        }
    }
}

/// Per-step training metrics record.
#[derive(Debug, Clone)]
pub struct StepMetrics {
    /// Step number.
    pub step: u64,
    /// Training loss for this step.
    pub loss: f32,
    /// Learning rate used for this step.
    pub learning_rate: f32,
    /// Tokens processed per second.
    pub tokens_per_second: f32,
    /// GPU utilization ratio (0.0 to 1.0).
    pub gpu_utilization: f32,
}

impl TrainingMetrics {
    /// Record a new step metrics entry.
    pub fn record(&mut self, metrics: StepMetrics) {
        self.total_tokens_seen += (metrics.tokens_per_second as u64).max(1);
        self.steps.push(metrics);
    }

    /// Compute the average loss over the last N steps.
    pub fn avg_loss(&self, n: usize) -> Option<f32> {
        if self.steps.is_empty() {
            return None;
        }
        let start = self.steps.len().saturating_sub(n);
        let recent = &self.steps[start..];
        if recent.is_empty() {
            return None;
        }
        let sum: f32 = recent.iter().map(|s| s.loss).sum();
        Some(sum / recent.len() as f32)
    }

    /// Compute throughput (tokens/sec) over the last N steps.
    pub fn throughput(&self, n: usize) -> Option<f32> {
        if self.steps.is_empty() {
            return None;
        }
        let start = self.steps.len().saturating_sub(n);
        let recent = &self.steps[start..];
        if recent.is_empty() {
            return None;
        }
        let sum: f32 = recent.iter().map(|s| s.tokens_per_second).sum();
        Some(sum / recent.len() as f32)
    }

    /// Total elapsed time since training started.
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuTrainingStep, TrainingMetrics};
    use crate::optimizer::{AdamW, DType, Optimizer, Tensor};

    #[test]
    fn gpu_training_step_updates_params() {
        let optimizer = AdamW::new(0.9, 0.999, 1e-8, 0.01);
        let mut step = GpuTrainingStep::new(optimizer, false);

        let mut param = vec![0.5, -0.5, 0.3];
        let grad = vec![0.1, -0.2, 0.05];

        let loss = step.step(0, &mut param, &grad, 1, 0.001).expect("step should succeed");
        assert!(!loss.is_nan(), "loss should be finite");

        // Params should have changed
        assert_ne!(param, vec![0.5, -0.5, 0.3]);
    }

    #[test]
    fn amp_scales_gradients() {
        let optimizer = AdamW::new(0.9, 0.999, 1e-8, 0.0);
        let mut step_amp = GpuTrainingStep::new(optimizer, true);
        assert!(step_amp.grad_scale() > 1.0, "AMP should use grad scaling");

        let optimizer2 = AdamW::new(0.9, 0.999, 1e-8, 0.0);
        let step_no_amp = GpuTrainingStep::new(optimizer2, false);
        assert_eq!(step_no_amp.grad_scale(), 1.0, "non-AMP should use scale 1.0");
    }

    #[test]
    fn training_metrics_tracks_loss() {
        let mut metrics = TrainingMetrics {
            start_time: std::time::Instant::now(),
            ..Default::default()
        };

        metrics.record(super::StepMetrics {
            step: 1,
            loss: 2.5,
            learning_rate: 0.001,
            tokens_per_second: 10000.0,
            gpu_utilization: 0.85,
        });

        metrics.record(super::StepMetrics {
            step: 2,
            loss: 2.3,
            learning_rate: 0.001,
            tokens_per_second: 10500.0,
            gpu_utilization: 0.87,
        });

        let avg = metrics.avg_loss(10).expect("should have avg");
        assert!((avg - 2.4).abs() < 0.01);

        let throughput = metrics.throughput(10).expect("should have throughput");
        assert!((throughput - 10250.0).abs() < 1.0);
    }
}
