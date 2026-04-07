use std::collections::{BTreeMap, BTreeSet};

use crate::error::{TrainingError, TrainingResult};
use crate::types::fnv64_hex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Fp32,
    Bf16,
    Fp16,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: &[usize], dtype: DType, data: Vec<f32>) -> TrainingResult<Self> {
        let expected = shape.iter().copied().product::<usize>();
        if expected != data.len() {
            return Err(TrainingError::InvalidInput(
                "tensor data length must match shape product",
            ));
        }

        Ok(Self {
            shape: shape.to_vec(),
            dtype,
            data,
        })
    }

    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let len = shape.iter().copied().product::<usize>();
        Self {
            shape: shape.to_vec(),
            dtype,
            data: vec![0.0; len],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerStateShard {
    pub param_id: u64,
    pub state_name: String,
    pub shard_index: u32,
    pub shard_count: u32,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub version: u32,
    pub checksum_hex: String,
    pub values: Vec<f32>,
}

pub trait Optimizer {
    fn name(&self) -> &'static str;

    fn init_state(&mut self, param_id: u64, shape: &[usize], dtype: DType) -> TrainingResult<()>;

    fn update(
        &mut self,
        param_id: u64,
        param: &mut Tensor,
        grad: &Tensor,
        step: usize,
        learning_rate: f32,
    ) -> TrainingResult<()>;

    fn export_state_shards(&self, shard_count: u32) -> TrainingResult<Vec<OptimizerStateShard>>;

    fn load_state_shards(&mut self, shards: &[OptimizerStateShard]) -> TrainingResult<()>;

    fn save_state(&self, shard_count: u32) -> TrainingResult<Vec<OptimizerStateShard>> {
        self.export_state_shards(shard_count)
    }

    fn load_state(&mut self, shards: &[OptimizerStateShard]) -> TrainingResult<()> {
        self.load_state_shards(shards)
    }
}

#[derive(Debug, Clone)]
pub struct SgdMomentum {
    momentum: f32,
    version: u32,
    velocity: BTreeMap<u64, Tensor>,
}

impl SgdMomentum {
    pub fn new(momentum: f32) -> Self {
        Self {
            momentum,
            version: 1,
            velocity: BTreeMap::new(),
        }
    }
}

impl Optimizer for SgdMomentum {
    fn name(&self) -> &'static str {
        "sgd_momentum"
    }

    fn init_state(&mut self, param_id: u64, shape: &[usize], dtype: DType) -> TrainingResult<()> {
        self.velocity
            .entry(param_id)
            .or_insert_with(|| Tensor::zeros(shape, dtype));
        Ok(())
    }

    fn update(
        &mut self,
        param_id: u64,
        param: &mut Tensor,
        grad: &Tensor,
        _step: usize,
        learning_rate: f32,
    ) -> TrainingResult<()> {
        if param.len() != grad.len() {
            return Err(TrainingError::InvalidInput(
                "param and grad must have same length",
            ));
        }

        self.init_state(param_id, &param.shape, param.dtype)?;
        let velocity = self
            .velocity
            .get_mut(&param_id)
            .ok_or(TrainingError::InvalidState("missing momentum state"))?;

        for idx in 0..param.data.len() {
            velocity.data[idx] = self.momentum * velocity.data[idx] + grad.data[idx];
            param.data[idx] -= learning_rate * velocity.data[idx];
        }

        Ok(())
    }

    fn export_state_shards(&self, shard_count: u32) -> TrainingResult<Vec<OptimizerStateShard>> {
        export_state_map("velocity", &self.velocity, shard_count, self.version)
    }

    fn load_state_shards(&mut self, shards: &[OptimizerStateShard]) -> TrainingResult<()> {
        self.velocity = load_state_map(shards, "velocity")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AdamW {
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    version: u32,
    m: BTreeMap<u64, Tensor>,
    v: BTreeMap<u64, Tensor>,
}

impl AdamW {
    pub fn new(beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            beta1,
            beta2,
            eps,
            weight_decay,
            version: 1,
            m: BTreeMap::new(),
            v: BTreeMap::new(),
        }
    }
}

impl Optimizer for AdamW {
    fn name(&self) -> &'static str {
        "adamw"
    }

    fn init_state(&mut self, param_id: u64, shape: &[usize], dtype: DType) -> TrainingResult<()> {
        self.m
            .entry(param_id)
            .or_insert_with(|| Tensor::zeros(shape, dtype));
        self.v
            .entry(param_id)
            .or_insert_with(|| Tensor::zeros(shape, dtype));
        Ok(())
    }

    fn update(
        &mut self,
        param_id: u64,
        param: &mut Tensor,
        grad: &Tensor,
        step: usize,
        learning_rate: f32,
    ) -> TrainingResult<()> {
        if param.len() != grad.len() {
            return Err(TrainingError::InvalidInput(
                "param and grad must have same length",
            ));
        }

        self.init_state(param_id, &param.shape, param.dtype)?;
        let m = self
            .m
            .get_mut(&param_id)
            .ok_or(TrainingError::InvalidState("missing adam m state"))?;
        let v = self
            .v
            .get_mut(&param_id)
            .ok_or(TrainingError::InvalidState("missing adam v state"))?;

        let step_f = (step as f32).max(1.0);
        let bias_correction1 = 1.0 - self.beta1.powf(step_f);
        let bias_correction2 = 1.0 - self.beta2.powf(step_f);

        for idx in 0..param.data.len() {
            let g = grad.data[idx];
            m.data[idx] = self.beta1 * m.data[idx] + (1.0 - self.beta1) * g;
            v.data[idx] = self.beta2 * v.data[idx] + (1.0 - self.beta2) * g * g;

            let m_hat = m.data[idx] / bias_correction1.max(1e-8);
            let v_hat = v.data[idx] / bias_correction2.max(1e-8);
            let update = m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * param.data[idx];
            param.data[idx] -= learning_rate * update;
        }

        Ok(())
    }

    fn export_state_shards(&self, shard_count: u32) -> TrainingResult<Vec<OptimizerStateShard>> {
        let mut shards = export_state_map("m", &self.m, shard_count, self.version)?;
        let mut v_shards = export_state_map("v", &self.v, shard_count, self.version)?;
        shards.append(&mut v_shards);
        Ok(shards)
    }

    fn load_state_shards(&mut self, shards: &[OptimizerStateShard]) -> TrainingResult<()> {
        self.m = load_state_map(shards, "m")?;
        self.v = load_state_map(shards, "v")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Adafactor {
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    version: u32,
    row_moments: BTreeMap<u64, Tensor>,
    col_moments: BTreeMap<u64, Tensor>,
    full_second_moment: BTreeMap<u64, Tensor>,
}

impl Adafactor {
    pub fn new(beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            beta2,
            eps,
            weight_decay,
            version: 1,
            row_moments: BTreeMap::new(),
            col_moments: BTreeMap::new(),
            full_second_moment: BTreeMap::new(),
        }
    }
}

impl Optimizer for Adafactor {
    fn name(&self) -> &'static str {
        "adafactor"
    }

    fn init_state(&mut self, param_id: u64, shape: &[usize], dtype: DType) -> TrainingResult<()> {
        if shape.len() == 2 {
            let rows = shape[0];
            let cols = shape[1];
            self.row_moments
                .entry(param_id)
                .or_insert_with(|| Tensor::zeros(&[rows], dtype));
            self.col_moments
                .entry(param_id)
                .or_insert_with(|| Tensor::zeros(&[cols], dtype));
            self.full_second_moment.remove(&param_id);
        } else {
            self.full_second_moment
                .entry(param_id)
                .or_insert_with(|| Tensor::zeros(shape, dtype));
            self.row_moments.remove(&param_id);
            self.col_moments.remove(&param_id);
        }

        Ok(())
    }

    fn update(
        &mut self,
        param_id: u64,
        param: &mut Tensor,
        grad: &Tensor,
        _step: usize,
        learning_rate: f32,
    ) -> TrainingResult<()> {
        if param.len() != grad.len() {
            return Err(TrainingError::InvalidInput(
                "param and grad must have same length",
            ));
        }

        self.init_state(param_id, &param.shape, param.dtype)?;

        if param.shape.len() == 2 {
            let rows = param.shape[0];
            let cols = param.shape[1];
            if rows * cols != param.len() {
                return Err(TrainingError::InvalidInput(
                    "2D tensor shape must match flattened tensor length",
                ));
            }

            let row_state = self
                .row_moments
                .get_mut(&param_id)
                .ok_or(TrainingError::InvalidState("missing adafactor row moments"))?;
            let col_state = self
                .col_moments
                .get_mut(&param_id)
                .ok_or(TrainingError::InvalidState("missing adafactor col moments"))?;

            for row in 0..rows {
                let mut sum = 0.0_f32;
                for col in 0..cols {
                    let idx = row * cols + col;
                    let g = grad.data[idx];
                    sum += g * g;
                }
                let mean = sum / cols as f32;
                row_state.data[row] = self.beta2 * row_state.data[row] + (1.0 - self.beta2) * mean;
            }

            for col in 0..cols {
                let mut sum = 0.0_f32;
                for row in 0..rows {
                    let idx = row * cols + col;
                    let g = grad.data[idx];
                    sum += g * g;
                }
                let mean = sum / rows as f32;
                col_state.data[col] = self.beta2 * col_state.data[col] + (1.0 - self.beta2) * mean;
            }

            let row_mean = row_state.data.iter().sum::<f32>() / rows as f32;
            let norm = row_mean.max(self.eps);

            for idx in 0..param.data.len() {
                let row = idx / cols;
                let col = idx % cols;
                let variance = (row_state.data[row] * col_state.data[col] / norm).max(self.eps);
                let update = grad.data[idx] / variance.sqrt() + self.weight_decay * param.data[idx];
                param.data[idx] -= learning_rate * update;
            }

            return Ok(());
        }

        let second =
            self.full_second_moment
                .get_mut(&param_id)
                .ok_or(TrainingError::InvalidState(
                    "missing adafactor second moment",
                ))?;

        for idx in 0..param.data.len() {
            let g = grad.data[idx];
            second.data[idx] = self.beta2 * second.data[idx] + (1.0 - self.beta2) * g * g;
            let update =
                g / (second.data[idx].max(self.eps)).sqrt() + self.weight_decay * param.data[idx];
            param.data[idx] -= learning_rate * update;
        }

        Ok(())
    }

    fn export_state_shards(&self, shard_count: u32) -> TrainingResult<Vec<OptimizerStateShard>> {
        let mut shards = export_state_map("rows", &self.row_moments, shard_count, self.version)?;
        let mut col_shards =
            export_state_map("cols", &self.col_moments, shard_count, self.version)?;
        let mut full_shards =
            export_state_map("v", &self.full_second_moment, shard_count, self.version)?;
        shards.append(&mut col_shards);
        shards.append(&mut full_shards);
        Ok(shards)
    }

    fn load_state_shards(&mut self, shards: &[OptimizerStateShard]) -> TrainingResult<()> {
        self.row_moments = load_state_map(shards, "rows")?;
        self.col_moments = load_state_map(shards, "cols")?;
        self.full_second_moment = load_state_map(shards, "v")?;
        Ok(())
    }
}

pub fn optimizer_state_fingerprint(shards: &[OptimizerStateShard]) -> String {
    let mut canonical = String::new();

    let mut ordered = shards.to_vec();
    ordered.sort_by(|a, b| {
        (
            a.param_id,
            a.state_name.as_str(),
            a.shard_index,
            a.shard_count,
            a.version,
        )
            .cmp(&(
                b.param_id,
                b.state_name.as_str(),
                b.shard_index,
                b.shard_count,
                b.version,
            ))
    });

    for shard in &ordered {
        canonical.push_str(&format!(
            "{}|{}|{}|{}|{}|{}|{}|{}\n",
            shard.param_id,
            shard.state_name,
            shard.shard_index,
            shard.shard_count,
            shard
                .shape
                .iter()
                .map(|dim| dim.to_string())
                .collect::<Vec<String>>()
                .join(","),
            dtype_to_str(shard.dtype),
            shard.version,
            shard.checksum_hex
        ));
    }

    fnv64_hex(canonical.as_bytes())
}

fn export_state_map(
    state_name: &str,
    state: &BTreeMap<u64, Tensor>,
    shard_count: u32,
    version: u32,
) -> TrainingResult<Vec<OptimizerStateShard>> {
    if shard_count == 0 {
        return Err(TrainingError::InvalidConfig(
            "shard_count must be greater than zero",
        ));
    }

    let mut shards = Vec::new();

    for (param_id, tensor) in state {
        let splits = split_tensor(&tensor.data, shard_count as usize);

        for (shard_index, values) in splits.into_iter().enumerate() {
            let checksum_hex = values_checksum(&values);
            shards.push(OptimizerStateShard {
                param_id: *param_id,
                state_name: state_name.to_owned(),
                shard_index: shard_index as u32,
                shard_count,
                shape: tensor.shape.clone(),
                dtype: tensor.dtype,
                version,
                checksum_hex,
                values,
            });
        }
    }

    Ok(shards)
}

fn load_state_map(
    shards: &[OptimizerStateShard],
    state_name: &str,
) -> TrainingResult<BTreeMap<u64, Tensor>> {
    let mut grouped: BTreeMap<u64, Vec<&OptimizerStateShard>> = BTreeMap::new();

    for shard in shards {
        if shard.state_name == state_name {
            grouped.entry(shard.param_id).or_default().push(shard);
        }
    }

    let mut restored = BTreeMap::new();

    for (param_id, mut parts) in grouped {
        parts.sort_by(|a, b| a.shard_index.cmp(&b.shard_index));

        let first = parts
            .first()
            .ok_or(TrainingError::InvalidState("state shard group is empty"))?;
        let expected_shard_count = first.shard_count;
        let expected_shape = first.shape.clone();
        let expected_dtype = first.dtype;
        let expected_version = first.version;
        let expected_count = expected_shard_count as usize;

        if expected_count == 0 || parts.len() != expected_count {
            return Err(TrainingError::MissingShard(format!(
                "param={param_id}/{state_name}"
            )));
        }

        let mut seen = BTreeSet::new();
        for part in &parts {
            if part.shard_count != expected_shard_count
                || part.shape != expected_shape
                || part.dtype != expected_dtype
                || part.version != expected_version
            {
                return Err(TrainingError::InvalidState(
                    "inconsistent shard metadata for optimizer state",
                ));
            }

            if !seen.insert(part.shard_index) {
                return Err(TrainingError::InvalidState(
                    "duplicate optimizer shard index detected",
                ));
            }
        }

        for expected_idx in 0..expected_count as u32 {
            if !seen.contains(&expected_idx) {
                return Err(TrainingError::MissingShard(format!(
                    "param={param_id}/{state_name}#{expected_idx}"
                )));
            }
        }

        let mut values = Vec::new();
        for part in parts {
            let checksum = values_checksum(&part.values);
            if checksum != part.checksum_hex {
                return Err(TrainingError::ChecksumMismatch {
                    path: format!("param={param_id}/{state_name}#{}", part.shard_index),
                    expected: part.checksum_hex.clone(),
                    actual: checksum,
                });
            }
            values.extend_from_slice(&part.values);
        }

        restored.insert(
            param_id,
            Tensor::new(&expected_shape, expected_dtype, values)
                .map_err(|_| TrainingError::InvalidState("restored tensor shape mismatch"))?,
        );
    }

    Ok(restored)
}

fn split_tensor(values: &[f32], shard_count: usize) -> Vec<Vec<f32>> {
    if shard_count == 0 {
        return Vec::new();
    }

    let mut shards = Vec::with_capacity(shard_count);
    let len = values.len();

    for idx in 0..shard_count {
        let start = idx * len / shard_count;
        let end = (idx + 1) * len / shard_count;
        shards.push(values[start..end].to_vec());
    }

    shards
}

fn values_checksum(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fnv64_hex(&bytes)
}

fn dtype_to_str(dtype: DType) -> &'static str {
    match dtype {
        DType::Fp32 => "fp32",
        DType::Bf16 => "bf16",
        DType::Fp16 => "fp16",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        optimizer_state_fingerprint, Adafactor, AdamW, DType, Optimizer, SgdMomentum, Tensor,
    };

    #[test]
    fn sgd_momentum_update_is_deterministic() {
        let mut opt = SgdMomentum::new(0.9);

        let mut p1 = Tensor::new(&[3], DType::Fp32, vec![1.0, 2.0, 3.0]).expect("valid tensor");
        let mut p2 = p1.clone();
        let grad = Tensor::new(&[3], DType::Fp32, vec![0.1, 0.2, 0.3]).expect("valid tensor");

        opt.update(0, &mut p1, &grad, 1, 0.01)
            .expect("update should succeed");

        let mut opt2 = SgdMomentum::new(0.9);
        opt2.update(0, &mut p2, &grad, 1, 0.01)
            .expect("update should succeed");

        assert_eq!(p1, p2);
    }

    #[test]
    fn adam_state_shard_roundtrip_restores_state() {
        let mut opt = AdamW::new(0.9, 0.999, 1e-8, 0.01);
        let mut param = Tensor::new(&[2], DType::Fp32, vec![0.5, -0.5]).expect("valid tensor");
        let grad = Tensor::new(&[2], DType::Fp32, vec![0.2, -0.1]).expect("valid tensor");

        opt.update(7, &mut param, &grad, 1, 0.001)
            .expect("update should succeed");

        let shards = opt
            .export_state_shards(2)
            .expect("state export should succeed");

        let mut restored = AdamW::new(0.9, 0.999, 1e-8, 0.01);
        restored
            .load_state_shards(&shards)
            .expect("state load should succeed");

        let roundtrip = restored
            .export_state_shards(2)
            .expect("state export should succeed");

        assert_eq!(shards, roundtrip);
    }

    #[test]
    fn adafactor_update_is_deterministic() {
        let mut opt_a = Adafactor::new(0.999, 1e-8, 0.0);
        let mut opt_b = Adafactor::new(0.999, 1e-8, 0.0);

        let mut p1 = Tensor::new(&[2, 2], DType::Fp32, vec![1.0, 2.0, 3.0, 4.0]).expect("valid");
        let mut p2 = p1.clone();
        let grad = Tensor::new(&[2, 2], DType::Fp32, vec![0.1, 0.2, 0.3, 0.4]).expect("valid");

        opt_a
            .update(10, &mut p1, &grad, 1, 0.01)
            .expect("update should succeed");
        opt_b
            .update(10, &mut p2, &grad, 1, 0.01)
            .expect("update should succeed");

        assert_eq!(p1, p2);

        let shards_a = opt_a.export_state_shards(2).expect("export should succeed");
        let shards_b = opt_b.export_state_shards(2).expect("export should succeed");
        assert_eq!(
            optimizer_state_fingerprint(&shards_a),
            optimizer_state_fingerprint(&shards_b)
        );
    }

    #[test]
    fn load_state_rejects_corrupted_shards() {
        let mut opt = AdamW::new(0.9, 0.999, 1e-8, 0.01);
        let mut param = Tensor::new(&[2], DType::Fp32, vec![0.5, -0.5]).expect("valid tensor");
        let grad = Tensor::new(&[2], DType::Fp32, vec![0.2, -0.1]).expect("valid tensor");

        opt.update(7, &mut param, &grad, 1, 0.001)
            .expect("update should succeed");

        let mut shards = opt
            .export_state_shards(2)
            .expect("state export should succeed");

        if let Some(first) = shards.first_mut() {
            first.values[0] += 1.0;
        }

        let mut restored = AdamW::new(0.9, 0.999, 1e-8, 0.01);
        let result = restored.load_state_shards(&shards);
        assert!(result.is_err());
    }
}
