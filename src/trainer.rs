#[allow(unused_imports)]
use rand::Rng;

#[derive(Debug, Clone)]
pub struct TrainerConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs: usize,
    pub seq_length: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            batch_size: 4,
            epochs: 10,
            seq_length: 32,
        }
    }
}

pub struct Trainer {
    config: TrainerConfig,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> Self {
        Self { config }
    }

    pub fn train(&self, _texts: &[String]) {
        println!(
            "Training started: {} epochs, batch_size={}, lr={}",
            self.config.epochs, self.config.batch_size, self.config.learning_rate
        );
    }
}

pub fn cross_entropy_loss(predictions: &[f32], targets: &[u32]) -> f32 {
    let mut loss = 0.0f32;
    for (pred, &target) in predictions.iter().zip(targets.iter()) {
        let _pred = pred.max(1e-7);
        let idx = target as usize;
        if idx < predictions.len() {
            loss -= predictions[idx].ln();
        }
    }
    loss / targets.len() as f32
}

pub fn compute_gradients(output: &[f32], target: u32) -> Vec<f32> {
    let len = output.len();
    let mut gradients = output.to_vec();
    let idx = target as usize;
    if idx < len {
        gradients[idx] -= 1.0;
    }
    let scale = len as f32;
    for g in &mut gradients {
        *g /= scale;
    }
    gradients
}

pub fn adam_update(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    t: usize,
    lr: f32,
) {
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;

    for (i, (p, g)) in param.iter_mut().zip(grad.iter()).enumerate() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

        let m_hat = m[i] / (1.0 - beta1.powi(t as i32));
        let v_hat = v[i] / (1.0 - beta2.powi(t as i32));

        *p -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy() {
        let pred = vec![0.1, 0.7, 0.2];
        let target = vec![1];
        let loss = cross_entropy_loss(&pred, &target);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_gradients() {
        let output = vec![0.2, 0.5, 0.3];
        let grad = compute_gradients(&output, 1);
        assert_eq!(grad.len(), output.len());
    }
}
