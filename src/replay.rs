use std::collections::BTreeMap;

use crate::curriculum::ExpansionPhase;
use crate::error::{TrainingError, TrainingResult};
use crate::optimizer::Tensor;
use crate::types::{fnv64_hex, TierId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayEventKind {
    MicroBatch {
        data_shard: String,
        seed: u64,
    },
    TierExpansion {
        tier: TierId,
        phase: ExpansionPhase,
        route_probability_bits: u32,
    },
    Collective {
        operation: String,
        rank_order: Vec<u32>,
        payload_checksum: String,
    },
    OptimizerStep {
        optimizer: String,
        parameter_checksum: String,
        learning_rate_bits: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayEvent {
    pub sequence: u64,
    pub update_step: u64,
    pub micro_batch_index: u32,
    pub kind: ReplayEventKind,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReplayContext {
    pub checkpoint_id: String,
    pub model_id: String,
    pub world_size: u32,
    pub events: Vec<ReplayEvent>,
}

impl ReplayContext {
    pub fn new(
        checkpoint_id: impl Into<String>,
        model_id: impl Into<String>,
        world_size: u32,
    ) -> Self {
        Self {
            checkpoint_id: checkpoint_id.into(),
            model_id: model_id.into(),
            world_size,
            events: Vec::new(),
        }
    }

    pub fn is_replayable(&self) -> bool {
        !self.checkpoint_id.is_empty() && !self.model_id.is_empty() && !self.events.is_empty()
    }

    pub fn push_event(&mut self, event: ReplayEvent) -> TrainingResult<()> {
        let expected_sequence = self.events.len() as u64;
        if event.sequence != expected_sequence {
            return Err(TrainingError::ReplayMismatch(
                "replay events must have contiguous sequence numbers",
            ));
        }

        if let Some(prev) = self.events.last() {
            if (event.update_step, event.micro_batch_index, event.sequence)
                < (prev.update_step, prev.micro_batch_index, prev.sequence)
            {
                return Err(TrainingError::ReplayMismatch(
                    "replay events must be appended in deterministic order",
                ));
            }
        }

        self.events.push(event);
        Ok(())
    }

    pub fn validate(&self) -> TrainingResult<()> {
        if self.checkpoint_id.trim().is_empty() || self.model_id.trim().is_empty() {
            return Err(TrainingError::InvalidConfig(
                "checkpoint_id and model_id are required for replay",
            ));
        }
        if self.world_size == 0 {
            return Err(TrainingError::InvalidConfig(
                "world_size must be greater than zero for replay",
            ));
        }

        for (idx, event) in self.events.iter().enumerate() {
            if event.sequence != idx as u64 {
                return Err(TrainingError::ReplayMismatch(
                    "event sequence indices must be contiguous",
                ));
            }
            if let ReplayEventKind::Collective { rank_order, .. } = &event.kind {
                if rank_order.len() != self.world_size as usize {
                    return Err(TrainingError::ReplayMismatch(
                        "collective rank order must cover full world size",
                    ));
                }
            }
        }

        Ok(())
    }

    pub fn to_canonical_string(&self) -> TrainingResult<String> {
        self.validate()?;

        let mut out = String::new();
        out.push_str(&format!("checkpoint_id={}\n", self.checkpoint_id));
        out.push_str(&format!("model_id={}\n", self.model_id));
        out.push_str(&format!("world_size={}\n", self.world_size));

        for event in &self.events {
            match &event.kind {
                ReplayEventKind::MicroBatch { data_shard, seed } => {
                    out.push_str(&format!(
                        "event|{}|{}|{}|micro_batch|{}|{}\n",
                        event.sequence,
                        event.update_step,
                        event.micro_batch_index,
                        data_shard,
                        seed
                    ));
                }
                ReplayEventKind::TierExpansion {
                    tier,
                    phase,
                    route_probability_bits,
                } => {
                    out.push_str(&format!(
                        "event|{}|{}|{}|tier_expansion|{}|{}|{}\n",
                        event.sequence,
                        event.update_step,
                        event.micro_batch_index,
                        tier,
                        phase_to_str(*phase),
                        route_probability_bits
                    ));
                }
                ReplayEventKind::Collective {
                    operation,
                    rank_order,
                    payload_checksum,
                } => {
                    let rank_csv = rank_order
                        .iter()
                        .map(|rank| rank.to_string())
                        .collect::<Vec<String>>()
                        .join(",");
                    out.push_str(&format!(
                        "event|{}|{}|{}|collective|{}|{}|{}\n",
                        event.sequence,
                        event.update_step,
                        event.micro_batch_index,
                        operation,
                        rank_csv,
                        payload_checksum
                    ));
                }
                ReplayEventKind::OptimizerStep {
                    optimizer,
                    parameter_checksum,
                    learning_rate_bits,
                } => {
                    out.push_str(&format!(
                        "event|{}|{}|{}|optimizer_step|{}|{}|{}\n",
                        event.sequence,
                        event.update_step,
                        event.micro_batch_index,
                        optimizer,
                        parameter_checksum,
                        learning_rate_bits
                    ));
                }
            }
        }

        out.push_str("end\n");
        Ok(out)
    }

    pub fn from_canonical_string(value: &str) -> TrainingResult<Self> {
        let mut checkpoint_id = None;
        let mut model_id = None;
        let mut world_size = None;
        let mut events = Vec::new();

        for raw in value.lines() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if line == "end" {
                break;
            }

            if let Some(rest) = line.strip_prefix("checkpoint_id=") {
                checkpoint_id = Some(rest.to_owned());
                continue;
            }
            if let Some(rest) = line.strip_prefix("model_id=") {
                model_id = Some(rest.to_owned());
                continue;
            }
            if let Some(rest) = line.strip_prefix("world_size=") {
                world_size = Some(
                    rest.parse::<u32>()
                        .map_err(|_| TrainingError::ParseError("invalid world_size"))?,
                );
                continue;
            }

            let fields: Vec<&str> = line.split('|').collect();
            if fields.len() < 5 || fields[0] != "event" {
                return Err(TrainingError::ParseError("invalid replay event record"));
            }

            let sequence = fields[1]
                .parse::<u64>()
                .map_err(|_| TrainingError::ParseError("invalid event sequence"))?;
            let update_step = fields[2]
                .parse::<u64>()
                .map_err(|_| TrainingError::ParseError("invalid update step"))?;
            let micro_batch_index = fields[3]
                .parse::<u32>()
                .map_err(|_| TrainingError::ParseError("invalid micro-batch index"))?;

            let kind = match fields[4] {
                "micro_batch" => {
                    if fields.len() != 7 {
                        return Err(TrainingError::ParseError(
                            "invalid micro_batch record width",
                        ));
                    }
                    ReplayEventKind::MicroBatch {
                        data_shard: fields[5].to_owned(),
                        seed: fields[6]
                            .parse::<u64>()
                            .map_err(|_| TrainingError::ParseError("invalid micro_batch seed"))?,
                    }
                }
                "tier_expansion" => {
                    if fields.len() != 8 {
                        return Err(TrainingError::ParseError(
                            "invalid tier_expansion record width",
                        ));
                    }
                    ReplayEventKind::TierExpansion {
                        tier: fields[5]
                            .parse::<u16>()
                            .map_err(|_| TrainingError::ParseError("invalid tier id"))?,
                        phase: str_to_phase(fields[6])?,
                        route_probability_bits: fields[7]
                            .parse::<u32>()
                            .map_err(|_| TrainingError::ParseError("invalid route probability"))?,
                    }
                }
                "collective" => {
                    if fields.len() != 8 {
                        return Err(TrainingError::ParseError("invalid collective record width"));
                    }
                    let rank_order = if fields[6].is_empty() {
                        Vec::new()
                    } else {
                        let mut parsed = Vec::new();
                        for token in fields[6].split(',') {
                            parsed.push(token.parse::<u32>().map_err(|_| {
                                TrainingError::ParseError("invalid collective rank")
                            })?);
                        }
                        parsed
                    };

                    ReplayEventKind::Collective {
                        operation: fields[5].to_owned(),
                        rank_order,
                        payload_checksum: fields[7].to_owned(),
                    }
                }
                "optimizer_step" => {
                    if fields.len() != 8 {
                        return Err(TrainingError::ParseError(
                            "invalid optimizer_step record width",
                        ));
                    }
                    ReplayEventKind::OptimizerStep {
                        optimizer: fields[5].to_owned(),
                        parameter_checksum: fields[6].to_owned(),
                        learning_rate_bits: fields[7]
                            .parse::<u32>()
                            .map_err(|_| TrainingError::ParseError("invalid learning rate bits"))?,
                    }
                }
                _ => return Err(TrainingError::ParseError("unknown replay event type")),
            };

            events.push(ReplayEvent {
                sequence,
                update_step,
                micro_batch_index,
                kind,
            });
        }

        let replay = Self {
            checkpoint_id: checkpoint_id
                .ok_or(TrainingError::ParseError("missing checkpoint_id"))?,
            model_id: model_id.ok_or(TrainingError::ParseError("missing model_id"))?,
            world_size: world_size.ok_or(TrainingError::ParseError("missing world_size"))?,
            events,
        };

        replay.validate()?;
        Ok(replay)
    }

    pub fn replay_hash(&self) -> TrainingResult<String> {
        let canonical = self.to_canonical_string()?;
        Ok(fnv64_hex(canonical.as_bytes()))
    }

    pub fn verify_against(&self, observed: &ReplayContext) -> TrainingResult<()> {
        self.validate()?;
        observed.validate()?;

        if self.checkpoint_id != observed.checkpoint_id
            || self.model_id != observed.model_id
            || self.world_size != observed.world_size
        {
            return Err(TrainingError::ReplayMismatch(
                "replay metadata differs from expected run",
            ));
        }

        if self.events.len() != observed.events.len() {
            return Err(TrainingError::ReplayMismatch(
                "replay event count differs from expected run",
            ));
        }

        for (expected, actual) in self.events.iter().zip(&observed.events) {
            if expected != actual {
                return Err(TrainingError::ReplayMismatch(
                    "replay event mismatch detected",
                ));
            }
        }

        Ok(())
    }
}

pub fn parameter_checksum(parameters: &BTreeMap<u64, Tensor>) -> String {
    let mut bytes = Vec::new();

    for (param_id, tensor) in parameters {
        bytes.extend_from_slice(&param_id.to_le_bytes());
        for dim in &tensor.shape {
            bytes.extend_from_slice(&(*dim as u64).to_le_bytes());
        }
        for value in &tensor.data {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }

    fnv64_hex(&bytes)
}

fn phase_to_str(phase: ExpansionPhase) -> &'static str {
    match phase {
        ExpansionPhase::Preparation => "preparation",
        ExpansionPhase::Isolation => "isolation",
        ExpansionPhase::Integration => "integration",
        ExpansionPhase::JointTraining => "joint_training",
        ExpansionPhase::Complete => "complete",
    }
}

fn str_to_phase(value: &str) -> TrainingResult<ExpansionPhase> {
    match value {
        "preparation" => Ok(ExpansionPhase::Preparation),
        "isolation" => Ok(ExpansionPhase::Isolation),
        "integration" => Ok(ExpansionPhase::Integration),
        "joint_training" => Ok(ExpansionPhase::JointTraining),
        "complete" => Ok(ExpansionPhase::Complete),
        _ => Err(TrainingError::ParseError("invalid expansion phase")),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{parameter_checksum, ReplayContext, ReplayEvent, ReplayEventKind};
    use crate::curriculum::ExpansionPhase;
    use crate::optimizer::{DType, Tensor};

    #[test]
    fn replay_roundtrip_is_lossless() {
        let mut replay = ReplayContext::new("ckpt-1", "lite-llm-base", 2);
        replay
            .push_event(ReplayEvent {
                sequence: 0,
                update_step: 10,
                micro_batch_index: 0,
                kind: ReplayEventKind::MicroBatch {
                    data_shard: "train-shard-0".to_owned(),
                    seed: 42,
                },
            })
            .expect("event append should succeed");
        replay
            .push_event(ReplayEvent {
                sequence: 1,
                update_step: 10,
                micro_batch_index: 1,
                kind: ReplayEventKind::TierExpansion {
                    tier: 7,
                    phase: ExpansionPhase::Integration,
                    route_probability_bits: (0.42_f32).to_bits(),
                },
            })
            .expect("event append should succeed");

        let canonical = replay
            .to_canonical_string()
            .expect("serialization should succeed");
        let parsed = ReplayContext::from_canonical_string(&canonical)
            .expect("deserialization should succeed");

        assert_eq!(replay, parsed);
        assert!(replay.replay_hash().is_ok());
    }

    #[test]
    fn replay_verifier_detects_mismatch() {
        let mut expected = ReplayContext::new("ckpt-1", "lite-llm-base", 1);
        expected
            .push_event(ReplayEvent {
                sequence: 0,
                update_step: 1,
                micro_batch_index: 0,
                kind: ReplayEventKind::MicroBatch {
                    data_shard: "a".to_owned(),
                    seed: 1,
                },
            })
            .expect("event append should succeed");

        let mut observed = expected.clone();
        observed.events[0].kind = ReplayEventKind::MicroBatch {
            data_shard: "a".to_owned(),
            seed: 2,
        };

        let verify = expected.verify_against(&observed);
        assert!(verify.is_err());
    }

    #[test]
    fn parameter_checksum_is_deterministic() {
        let mut params = BTreeMap::new();
        params.insert(
            1,
            Tensor::new(&[2], DType::Fp32, vec![1.0, 2.0]).expect("valid tensor"),
        );

        let a = parameter_checksum(&params);
        let b = parameter_checksum(&params);
        assert_eq!(a, b);
    }
}
