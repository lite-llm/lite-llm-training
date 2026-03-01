use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};

use crate::error::{TrainingError, TrainingResult};
use crate::optimizer::{DType, OptimizerStateShard};
use crate::types::fnv64_hex;

const MANIFEST_FILE: &str = "manifest.trainchk";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedBinaryShard {
    pub name: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RankCheckpointPayload {
    pub rank: u32,
    pub parameter_shards: Vec<NamedBinaryShard>,
    pub optimizer_state_shards: Vec<OptimizerStateShard>,
    pub router_state_shards: Vec<NamedBinaryShard>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointRequest {
    pub checkpoint_id: String,
    pub model_id: String,
    pub step: u64,
    pub world_size: u32,
    pub optimizer_name: String,
    pub metadata_version: u32,
}

impl CheckpointRequest {
    pub fn validate(&self) -> TrainingResult<()> {
        if self.checkpoint_id.trim().is_empty() {
            return Err(TrainingError::InvalidConfig(
                "checkpoint_id must not be empty",
            ));
        }
        if self.model_id.trim().is_empty() {
            return Err(TrainingError::InvalidConfig("model_id must not be empty"));
        }
        if self.world_size == 0 {
            return Err(TrainingError::InvalidConfig(
                "world_size must be greater than zero",
            ));
        }
        if self.optimizer_name.trim().is_empty() {
            return Err(TrainingError::InvalidConfig(
                "optimizer_name must not be empty",
            ));
        }
        if self.metadata_version == 0 {
            return Err(TrainingError::InvalidConfig(
                "metadata_version must be greater than zero",
            ));
        }
        Ok(())
    }
}

pub trait ChecksumVerifier {
    fn verify(&self, path: &str, expected_hex: &str, bytes: &[u8]) -> TrainingResult<()>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Fnv64ChecksumVerifier;

impl ChecksumVerifier for Fnv64ChecksumVerifier {
    fn verify(&self, path: &str, expected_hex: &str, bytes: &[u8]) -> TrainingResult<()> {
        let actual = fnv64_hex(bytes);
        if actual != expected_hex {
            return Err(TrainingError::ChecksumMismatch {
                path: path.to_owned(),
                expected: expected_hex.to_owned(),
                actual,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryShardRef {
    pub rank: u32,
    pub name: String,
    pub path: String,
    pub bytes: u64,
    pub checksum_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizerStateShardRef {
    pub rank: u32,
    pub path: String,
    pub param_id: u64,
    pub state_name: String,
    pub shard_index: u32,
    pub shard_count: u32,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub version: u32,
    pub value_checksum_hex: String,
    pub file_checksum_hex: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributedCheckpointManifest {
    pub metadata_version: u32,
    pub checkpoint_id: String,
    pub model_id: String,
    pub step: u64,
    pub world_size: u32,
    pub optimizer_name: String,
    pub parameter_shards: Vec<BinaryShardRef>,
    pub optimizer_state_shards: Vec<OptimizerStateShardRef>,
    pub router_state_shards: Vec<BinaryShardRef>,
}

impl DistributedCheckpointManifest {
    pub fn validate(&self) -> TrainingResult<()> {
        if self.metadata_version == 0 {
            return Err(TrainingError::InvalidConfig(
                "metadata_version must be greater than zero",
            ));
        }
        if self.world_size == 0 {
            return Err(TrainingError::InvalidConfig(
                "world_size must be greater than zero",
            ));
        }
        if self.checkpoint_id.trim().is_empty() || self.model_id.trim().is_empty() {
            return Err(TrainingError::InvalidConfig(
                "checkpoint_id and model_id must not be empty",
            ));
        }

        let expected_ranks: BTreeSet<u32> = (0..self.world_size).collect();
        let mut p = BTreeSet::new();
        let mut o = BTreeSet::new();
        let mut r = BTreeSet::new();

        for shard in &self.parameter_shards {
            if shard.rank >= self.world_size {
                return Err(TrainingError::InvalidState(
                    "parameter shard rank is outside world_size",
                ));
            }
            p.insert(shard.rank);
        }

        for shard in &self.optimizer_state_shards {
            if shard.rank >= self.world_size {
                return Err(TrainingError::InvalidState(
                    "optimizer shard rank is outside world_size",
                ));
            }
            if shard.shard_count == 0 {
                return Err(TrainingError::InvalidState(
                    "optimizer shard_count must be non-zero",
                ));
            }
            o.insert(shard.rank);
        }

        for shard in &self.router_state_shards {
            if shard.rank >= self.world_size {
                return Err(TrainingError::InvalidState(
                    "router shard rank is outside world_size",
                ));
            }
            r.insert(shard.rank);
        }

        if p != expected_ranks || o != expected_ranks || r != expected_ranks {
            return Err(TrainingError::MissingShard(
                "checkpoint must include parameter, optimizer and router shards for all ranks"
                    .to_owned(),
            ));
        }

        Ok(())
    }

    pub fn to_canonical_string(&self) -> TrainingResult<String> {
        self.validate()?;

        let mut out = String::new();
        out.push_str(&format!("metadata_version={}\n", self.metadata_version));
        out.push_str(&format!("checkpoint_id={}\n", self.checkpoint_id));
        out.push_str(&format!("model_id={}\n", self.model_id));
        out.push_str(&format!("step={}\n", self.step));
        out.push_str(&format!("world_size={}\n", self.world_size));
        out.push_str(&format!("optimizer_name={}\n", self.optimizer_name));

        for shard in &self.parameter_shards {
            out.push_str(&format!(
                "param|{}|{}|{}|{}|{}\n",
                shard.rank, shard.path, shard.name, shard.bytes, shard.checksum_hex
            ));
        }

        for shard in &self.optimizer_state_shards {
            let shape = shard
                .shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<String>>()
                .join(",");
            out.push_str(&format!(
                "optim|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n",
                shard.rank,
                shard.path,
                shard.param_id,
                shard.state_name,
                shard.shard_index,
                shard.shard_count,
                shape,
                dtype_to_str(shard.dtype),
                shard.version,
                shard.value_checksum_hex,
                shard.file_checksum_hex,
                shard.bytes
            ));
        }

        for shard in &self.router_state_shards {
            out.push_str(&format!(
                "router|{}|{}|{}|{}|{}\n",
                shard.rank, shard.path, shard.name, shard.bytes, shard.checksum_hex
            ));
        }

        out.push_str("end\n");
        Ok(out)
    }
}
impl DistributedCheckpointManifest {
    pub fn from_canonical_string(value: &str) -> TrainingResult<Self> {
        let mut metadata_version = None;
        let mut checkpoint_id = None;
        let mut model_id = None;
        let mut step = None;
        let mut world_size = None;
        let mut optimizer_name = None;
        let mut parameter_shards = Vec::new();
        let mut optimizer_state_shards = Vec::new();
        let mut router_state_shards = Vec::new();

        for raw in value.lines() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if line == "end" {
                break;
            }

            if let Some(rest) = line.strip_prefix("metadata_version=") {
                metadata_version = Some(
                    rest.parse::<u32>()
                        .map_err(|_| TrainingError::ParseError("invalid metadata_version"))?,
                );
                continue;
            }
            if let Some(rest) = line.strip_prefix("checkpoint_id=") {
                checkpoint_id = Some(rest.to_owned());
                continue;
            }
            if let Some(rest) = line.strip_prefix("model_id=") {
                model_id = Some(rest.to_owned());
                continue;
            }
            if let Some(rest) = line.strip_prefix("step=") {
                step = Some(
                    rest.parse::<u64>()
                        .map_err(|_| TrainingError::ParseError("invalid step"))?,
                );
                continue;
            }
            if let Some(rest) = line.strip_prefix("world_size=") {
                world_size = Some(
                    rest.parse::<u32>()
                        .map_err(|_| TrainingError::ParseError("invalid world_size"))?,
                );
                continue;
            }
            if let Some(rest) = line.strip_prefix("optimizer_name=") {
                optimizer_name = Some(rest.to_owned());
                continue;
            }

            let mut parts = line.split('|');
            let tag = parts
                .next()
                .ok_or(TrainingError::ParseError("missing line tag"))?;

            match tag {
                "param" => {
                    parameter_shards.push(BinaryShardRef {
                        rank: parts
                            .next()
                            .ok_or(TrainingError::ParseError("param rank missing"))?
                            .parse::<u32>()
                            .map_err(|_| TrainingError::ParseError("invalid param rank"))?,
                        path: parts
                            .next()
                            .ok_or(TrainingError::ParseError("param path missing"))?
                            .to_owned(),
                        name: parts
                            .next()
                            .ok_or(TrainingError::ParseError("param name missing"))?
                            .to_owned(),
                        bytes: parts
                            .next()
                            .ok_or(TrainingError::ParseError("param bytes missing"))?
                            .parse::<u64>()
                            .map_err(|_| TrainingError::ParseError("invalid param bytes"))?,
                        checksum_hex: parts
                            .next()
                            .ok_or(TrainingError::ParseError("param checksum missing"))?
                            .to_owned(),
                    });
                }
                "optim" => {
                    let fields: Vec<&str> = line.split('|').collect();
                    if fields.len() != 13 {
                        return Err(TrainingError::ParseError("invalid optim record width"));
                    }

                    let shape = if fields[7].is_empty() {
                        Vec::new()
                    } else {
                        let mut dims = Vec::new();
                        for dim in fields[7].split(',') {
                            dims.push(dim.parse::<usize>().map_err(|_| {
                                TrainingError::ParseError("invalid optim shape dimension")
                            })?);
                        }
                        dims
                    };

                    optimizer_state_shards.push(OptimizerStateShardRef {
                        rank: fields[1]
                            .parse::<u32>()
                            .map_err(|_| TrainingError::ParseError("invalid optim rank"))?,
                        path: fields[2].to_owned(),
                        param_id: fields[3]
                            .parse::<u64>()
                            .map_err(|_| TrainingError::ParseError("invalid optim param_id"))?,
                        state_name: fields[4].to_owned(),
                        shard_index: fields[5]
                            .parse::<u32>()
                            .map_err(|_| TrainingError::ParseError("invalid optim shard_index"))?,
                        shard_count: fields[6]
                            .parse::<u32>()
                            .map_err(|_| TrainingError::ParseError("invalid optim shard_count"))?,
                        shape,
                        dtype: str_to_dtype(fields[8])?,
                        version: fields[9]
                            .parse::<u32>()
                            .map_err(|_| TrainingError::ParseError("invalid optim version"))?,
                        value_checksum_hex: fields[10].to_owned(),
                        file_checksum_hex: fields[11].to_owned(),
                        bytes: fields[12]
                            .parse::<u64>()
                            .map_err(|_| TrainingError::ParseError("invalid optim bytes"))?,
                    });
                }
                "router" => {
                    router_state_shards.push(BinaryShardRef {
                        rank: parts
                            .next()
                            .ok_or(TrainingError::ParseError("router rank missing"))?
                            .parse::<u32>()
                            .map_err(|_| TrainingError::ParseError("invalid router rank"))?,
                        path: parts
                            .next()
                            .ok_or(TrainingError::ParseError("router path missing"))?
                            .to_owned(),
                        name: parts
                            .next()
                            .ok_or(TrainingError::ParseError("router name missing"))?
                            .to_owned(),
                        bytes: parts
                            .next()
                            .ok_or(TrainingError::ParseError("router bytes missing"))?
                            .parse::<u64>()
                            .map_err(|_| TrainingError::ParseError("invalid router bytes"))?,
                        checksum_hex: parts
                            .next()
                            .ok_or(TrainingError::ParseError("router checksum missing"))?
                            .to_owned(),
                    });
                }
                _ => {}
            }
        }

        let manifest = Self {
            metadata_version: metadata_version
                .ok_or(TrainingError::ParseError("missing metadata_version"))?,
            checkpoint_id: checkpoint_id
                .ok_or(TrainingError::ParseError("missing checkpoint_id"))?,
            model_id: model_id.ok_or(TrainingError::ParseError("missing model_id"))?,
            step: step.ok_or(TrainingError::ParseError("missing step"))?,
            world_size: world_size.ok_or(TrainingError::ParseError("missing world_size"))?,
            optimizer_name: optimizer_name
                .ok_or(TrainingError::ParseError("missing optimizer_name"))?,
            parameter_shards,
            optimizer_state_shards,
            router_state_shards,
        };

        manifest.validate()?;
        Ok(manifest)
    }
}

#[derive(Debug, Clone)]
pub struct DistributedCheckpointRepository {
    root: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RestoredDistributedCheckpoint {
    pub manifest: DistributedCheckpointManifest,
    pub payloads: Vec<RankCheckpointPayload>,
}

impl DistributedCheckpointRepository {
    pub fn new(root: impl Into<PathBuf>) -> TrainingResult<Self> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn commit(
        &self,
        request: &CheckpointRequest,
        payloads: &[RankCheckpointPayload],
    ) -> TrainingResult<DistributedCheckpointManifest> {
        request.validate()?;
        validate_rank_payloads(request.world_size, payloads)?;

        let lock_path = self.lock_path(&request.checkpoint_id);
        let _lock = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&lock_path)
            .map_err(|err| {
                if err.kind() == std::io::ErrorKind::AlreadyExists {
                    TrainingError::IoError("checkpoint lock already held".to_owned())
                } else {
                    TrainingError::from(err)
                }
            })?;

        let temp_dir = self.temp_checkpoint_dir(&request.checkpoint_id);
        let final_dir = self.checkpoint_dir(&request.checkpoint_id);

        if temp_dir.exists() {
            fs::remove_dir_all(&temp_dir)?;
        }
        fs::create_dir_all(&temp_dir)?;

        let result = self
            .write_payloads(&temp_dir, request, payloads)
            .and_then(|manifest| {
                let canonical = manifest.to_canonical_string()?;
                fs::write(temp_dir.join(MANIFEST_FILE), canonical.as_bytes())?;

                if final_dir.exists() {
                    fs::remove_dir_all(&final_dir)?;
                }
                fs::rename(&temp_dir, &final_dir)?;
                Ok(manifest)
            });

        if result.is_err() {
            let _ = fs::remove_dir_all(&temp_dir);
        }

        let _ = fs::remove_file(lock_path);
        result
    }
    pub fn restore(
        &self,
        checkpoint_id: &str,
        verifier: &dyn ChecksumVerifier,
    ) -> TrainingResult<RestoredDistributedCheckpoint> {
        let checkpoint_dir = self.checkpoint_dir(checkpoint_id);
        let canonical = fs::read_to_string(checkpoint_dir.join(MANIFEST_FILE))?;
        let manifest = DistributedCheckpointManifest::from_canonical_string(&canonical)?;

        let mut by_rank: BTreeMap<u32, RankCheckpointPayload> = BTreeMap::new();
        for rank in 0..manifest.world_size {
            by_rank.insert(
                rank,
                RankCheckpointPayload {
                    rank,
                    parameter_shards: Vec::new(),
                    optimizer_state_shards: Vec::new(),
                    router_state_shards: Vec::new(),
                },
            );
        }

        for shard in &manifest.parameter_shards {
            let bytes = fs::read(logical_path_to_fs(&checkpoint_dir, &shard.path))?;
            if bytes.len() as u64 != shard.bytes {
                return Err(TrainingError::ChecksumMismatch {
                    path: shard.path.clone(),
                    expected: shard.bytes.to_string(),
                    actual: bytes.len().to_string(),
                });
            }
            verifier.verify(&shard.path, &shard.checksum_hex, &bytes)?;
            by_rank
                .get_mut(&shard.rank)
                .ok_or(TrainingError::InvalidState("missing rank during restore"))?
                .parameter_shards
                .push(NamedBinaryShard {
                    name: shard.name.clone(),
                    bytes,
                });
        }

        for shard in &manifest.optimizer_state_shards {
            let bytes = fs::read(logical_path_to_fs(&checkpoint_dir, &shard.path))?;
            if bytes.len() as u64 != shard.bytes {
                return Err(TrainingError::ChecksumMismatch {
                    path: shard.path.clone(),
                    expected: shard.bytes.to_string(),
                    actual: bytes.len().to_string(),
                });
            }
            verifier.verify(&shard.path, &shard.file_checksum_hex, &bytes)?;
            let decoded =
                decode_optimizer_state_shard(std::str::from_utf8(&bytes).map_err(|_| {
                    TrainingError::ParseError("optimizer state shard is not valid utf-8")
                })?)?;

            if decoded.param_id != shard.param_id
                || decoded.state_name != shard.state_name
                || decoded.shard_index != shard.shard_index
                || decoded.shard_count != shard.shard_count
                || decoded.shape != shard.shape
                || decoded.dtype != shard.dtype
                || decoded.version != shard.version
                || decoded.checksum_hex != shard.value_checksum_hex
            {
                return Err(TrainingError::ReplayMismatch(
                    "optimizer shard metadata mismatch during restore",
                ));
            }

            by_rank
                .get_mut(&shard.rank)
                .ok_or(TrainingError::InvalidState("missing rank during restore"))?
                .optimizer_state_shards
                .push(decoded);
        }

        for shard in &manifest.router_state_shards {
            let bytes = fs::read(logical_path_to_fs(&checkpoint_dir, &shard.path))?;
            if bytes.len() as u64 != shard.bytes {
                return Err(TrainingError::ChecksumMismatch {
                    path: shard.path.clone(),
                    expected: shard.bytes.to_string(),
                    actual: bytes.len().to_string(),
                });
            }
            verifier.verify(&shard.path, &shard.checksum_hex, &bytes)?;
            by_rank
                .get_mut(&shard.rank)
                .ok_or(TrainingError::InvalidState("missing rank during restore"))?
                .router_state_shards
                .push(NamedBinaryShard {
                    name: shard.name.clone(),
                    bytes,
                });
        }

        let mut payloads = by_rank
            .into_values()
            .collect::<Vec<RankCheckpointPayload>>();
        for payload in &mut payloads {
            payload
                .parameter_shards
                .sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
            payload
                .router_state_shards
                .sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
            payload.optimizer_state_shards.sort_by(|a, b| {
                (a.param_id, a.state_name.as_str(), a.shard_index).cmp(&(
                    b.param_id,
                    b.state_name.as_str(),
                    b.shard_index,
                ))
            });
        }

        Ok(RestoredDistributedCheckpoint { manifest, payloads })
    }

    pub fn abort_staging(&self, checkpoint_id: &str) -> TrainingResult<()> {
        let temp_dir = self.temp_checkpoint_dir(checkpoint_id);
        if temp_dir.exists() {
            fs::remove_dir_all(temp_dir)?;
        }
        Ok(())
    }

    pub fn list_checkpoints(&self) -> TrainingResult<Vec<String>> {
        let mut ids = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".tmp") {
                continue;
            }
            if entry.path().join(MANIFEST_FILE).exists() {
                ids.push(name);
            }
        }
        ids.sort();
        Ok(ids)
    }

    fn write_payloads(
        &self,
        temp_dir: &Path,
        request: &CheckpointRequest,
        payloads: &[RankCheckpointPayload],
    ) -> TrainingResult<DistributedCheckpointManifest> {
        let mut ordered = payloads.to_vec();
        ordered.sort_by_key(|payload| payload.rank);

        let mut parameter_refs = Vec::new();
        let mut optimizer_refs = Vec::new();
        let mut router_refs = Vec::new();

        for payload in &ordered {
            write_named_shards(
                temp_dir,
                "params",
                payload.rank,
                &payload.parameter_shards,
                &mut parameter_refs,
            )?;
            write_named_shards(
                temp_dir,
                "router",
                payload.rank,
                &payload.router_state_shards,
                &mut router_refs,
            )?;

            let mut optim = payload.optimizer_state_shards.clone();
            optim.sort_by(|a, b| {
                (a.param_id, a.state_name.as_str(), a.shard_index).cmp(&(
                    b.param_id,
                    b.state_name.as_str(),
                    b.shard_index,
                ))
            });

            for shard in &optim {
                let computed_value_checksum = values_checksum(&shard.values);
                if computed_value_checksum != shard.checksum_hex {
                    return Err(TrainingError::ChecksumMismatch {
                        path: format!(
                            "optim_state/param={} state={} shard={}",
                            shard.param_id, shard.state_name, shard.shard_index
                        ),
                        expected: shard.checksum_hex.clone(),
                        actual: computed_value_checksum,
                    });
                }

                let name = format!(
                    "p{}-{}-s{:04}-of-{:04}.state",
                    shard.param_id,
                    sanitize_component(shard.state_name.as_str()),
                    shard.shard_index,
                    shard.shard_count
                );
                let logical_path = format!("optim_state/rank-{}/{name}", rank_folder(payload.rank));
                let fs_path = logical_path_to_fs(temp_dir, &logical_path);
                if let Some(parent) = fs_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let encoded = encode_optimizer_state_shard(shard);
                fs::write(&fs_path, encoded.as_bytes())?;

                optimizer_refs.push(OptimizerStateShardRef {
                    rank: payload.rank,
                    path: logical_path,
                    param_id: shard.param_id,
                    state_name: shard.state_name.clone(),
                    shard_index: shard.shard_index,
                    shard_count: shard.shard_count,
                    shape: shard.shape.clone(),
                    dtype: shard.dtype,
                    version: shard.version,
                    value_checksum_hex: shard.checksum_hex.clone(),
                    file_checksum_hex: fnv64_hex(encoded.as_bytes()),
                    bytes: encoded.len() as u64,
                });
            }
        }

        parameter_refs.sort_by(|a, b| (a.rank, a.name.as_str()).cmp(&(b.rank, b.name.as_str())));
        optimizer_refs.sort_by(|a, b| {
            (a.rank, a.param_id, a.state_name.as_str(), a.shard_index).cmp(&(
                b.rank,
                b.param_id,
                b.state_name.as_str(),
                b.shard_index,
            ))
        });
        router_refs.sort_by(|a, b| (a.rank, a.name.as_str()).cmp(&(b.rank, b.name.as_str())));

        let manifest = DistributedCheckpointManifest {
            metadata_version: request.metadata_version,
            checkpoint_id: request.checkpoint_id.clone(),
            model_id: request.model_id.clone(),
            step: request.step,
            world_size: request.world_size,
            optimizer_name: request.optimizer_name.clone(),
            parameter_shards: parameter_refs,
            optimizer_state_shards: optimizer_refs,
            router_state_shards: router_refs,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    fn checkpoint_dir(&self, checkpoint_id: &str) -> PathBuf {
        self.root.join(checkpoint_id)
    }

    fn temp_checkpoint_dir(&self, checkpoint_id: &str) -> PathBuf {
        self.root.join(format!("{checkpoint_id}.tmp"))
    }

    fn lock_path(&self, checkpoint_id: &str) -> PathBuf {
        self.root.join(format!("{checkpoint_id}.lock"))
    }
}
pub fn encode_optimizer_state_shard(shard: &OptimizerStateShard) -> String {
    let shape = shard
        .shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<String>>()
        .join(",");
    let values_bits = shard
        .values
        .iter()
        .map(|v| format!("{:08x}", v.to_bits()))
        .collect::<Vec<String>>()
        .join(",");

    format!(
        "param_id={}\nstate_name={}\nshard_index={}\nshard_count={}\nshape={}\ndtype={}\nversion={}\nvalue_checksum={}\nvalues_bits={}\nend\n",
        shard.param_id,
        shard.state_name,
        shard.shard_index,
        shard.shard_count,
        shape,
        dtype_to_str(shard.dtype),
        shard.version,
        shard.checksum_hex,
        values_bits
    )
}

pub fn decode_optimizer_state_shard(value: &str) -> TrainingResult<OptimizerStateShard> {
    let mut param_id = None;
    let mut state_name = None;
    let mut shard_index = None;
    let mut shard_count = None;
    let mut shape = None;
    let mut dtype = None;
    let mut version = None;
    let mut checksum_hex = None;
    let mut values = None;

    for raw in value.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line == "end" {
            break;
        }
        if let Some(rest) = line.strip_prefix("param_id=") {
            param_id = Some(
                rest.parse::<u64>()
                    .map_err(|_| TrainingError::ParseError("invalid param_id"))?,
            );
            continue;
        }
        if let Some(rest) = line.strip_prefix("state_name=") {
            state_name = Some(rest.to_owned());
            continue;
        }
        if let Some(rest) = line.strip_prefix("shard_index=") {
            shard_index = Some(
                rest.parse::<u32>()
                    .map_err(|_| TrainingError::ParseError("invalid shard_index"))?,
            );
            continue;
        }
        if let Some(rest) = line.strip_prefix("shard_count=") {
            shard_count = Some(
                rest.parse::<u32>()
                    .map_err(|_| TrainingError::ParseError("invalid shard_count"))?,
            );
            continue;
        }
        if let Some(rest) = line.strip_prefix("shape=") {
            let parsed = if rest.is_empty() {
                Vec::new()
            } else {
                let mut dims = Vec::new();
                for dim in rest.split(',') {
                    dims.push(
                        dim.parse::<usize>()
                            .map_err(|_| TrainingError::ParseError("invalid shape dimension"))?,
                    );
                }
                dims
            };
            shape = Some(parsed);
            continue;
        }
        if let Some(rest) = line.strip_prefix("dtype=") {
            dtype = Some(str_to_dtype(rest)?);
            continue;
        }
        if let Some(rest) = line.strip_prefix("version=") {
            version = Some(
                rest.parse::<u32>()
                    .map_err(|_| TrainingError::ParseError("invalid version"))?,
            );
            continue;
        }
        if let Some(rest) = line.strip_prefix("value_checksum=") {
            checksum_hex = Some(rest.to_owned());
            continue;
        }
        if let Some(rest) = line.strip_prefix("values_bits=") {
            let parsed = if rest.is_empty() {
                Vec::new()
            } else {
                let mut out = Vec::new();
                for token in rest.split(',') {
                    let bits = u32::from_str_radix(token, 16)
                        .map_err(|_| TrainingError::ParseError("invalid value bits"))?;
                    out.push(f32::from_bits(bits));
                }
                out
            };
            values = Some(parsed);
            continue;
        }
    }

    let shard = OptimizerStateShard {
        param_id: param_id.ok_or(TrainingError::ParseError("missing param_id"))?,
        state_name: state_name.ok_or(TrainingError::ParseError("missing state_name"))?,
        shard_index: shard_index.ok_or(TrainingError::ParseError("missing shard_index"))?,
        shard_count: shard_count.ok_or(TrainingError::ParseError("missing shard_count"))?,
        shape: shape.ok_or(TrainingError::ParseError("missing shape"))?,
        dtype: dtype.ok_or(TrainingError::ParseError("missing dtype"))?,
        version: version.ok_or(TrainingError::ParseError("missing version"))?,
        checksum_hex: checksum_hex.ok_or(TrainingError::ParseError("missing value_checksum"))?,
        values: values.ok_or(TrainingError::ParseError("missing values_bits"))?,
    };

    let computed = values_checksum(&shard.values);
    if computed != shard.checksum_hex {
        return Err(TrainingError::ChecksumMismatch {
            path: "optimizer_state_shard".to_owned(),
            expected: shard.checksum_hex,
            actual: computed,
        });
    }

    Ok(shard)
}

fn validate_rank_payloads(
    world_size: u32,
    payloads: &[RankCheckpointPayload],
) -> TrainingResult<()> {
    if payloads.len() != world_size as usize {
        return Err(TrainingError::MissingShard(
            "payload count must equal world_size".to_owned(),
        ));
    }

    let mut seen = BTreeSet::new();
    for payload in payloads {
        if payload.rank >= world_size {
            return Err(TrainingError::InvalidInput(
                "payload rank exceeds world_size",
            ));
        }
        if !seen.insert(payload.rank) {
            return Err(TrainingError::InvalidInput("duplicate payload rank"));
        }
    }

    let expected: BTreeSet<u32> = (0..world_size).collect();
    if seen != expected {
        return Err(TrainingError::MissingShard(
            "payload ranks must cover all ranks".to_owned(),
        ));
    }

    Ok(())
}

fn write_named_shards(
    root: &Path,
    subdir: &str,
    rank: u32,
    shards: &[NamedBinaryShard],
    out: &mut Vec<BinaryShardRef>,
) -> TrainingResult<()> {
    let mut ordered = shards.to_vec();
    ordered.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));

    for (index, shard) in ordered.iter().enumerate() {
        let name = format!(
            "{:04}-{}.bin",
            index,
            sanitize_component(shard.name.as_str())
        );
        let logical_path = format!("{subdir}/rank-{}/{name}", rank_folder(rank));
        let fs_path = logical_path_to_fs(root, &logical_path);

        if let Some(parent) = fs_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&fs_path, &shard.bytes)?;

        out.push(BinaryShardRef {
            rank,
            name: shard.name.clone(),
            path: logical_path,
            bytes: shard.bytes.len() as u64,
            checksum_hex: fnv64_hex(&shard.bytes),
        });
    }

    Ok(())
}

fn logical_path_to_fs(root: &Path, logical_path: &str) -> PathBuf {
    let mut path = root.to_path_buf();
    for part in logical_path.split('/') {
        path.push(part);
    }
    path
}

fn rank_folder(rank: u32) -> String {
    format!("{rank:04}")
}

fn sanitize_component(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '.' || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "blob".to_owned()
    } else {
        out
    }
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

fn str_to_dtype(value: &str) -> TrainingResult<DType> {
    match value {
        "fp32" => Ok(DType::Fp32),
        "bf16" => Ok(DType::Bf16),
        "fp16" => Ok(DType::Fp16),
        _ => Err(TrainingError::ParseError("invalid dtype")),
    }
}
#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        decode_optimizer_state_shard, encode_optimizer_state_shard, CheckpointRequest,
        DistributedCheckpointManifest, DistributedCheckpointRepository, Fnv64ChecksumVerifier,
        NamedBinaryShard, RankCheckpointPayload,
    };
    use crate::optimizer::{DType, OptimizerStateShard};

    fn unique_temp_root() -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("lite-llm-training-checkpoint-tests-{nanos}"))
    }

    fn value_checksum(values: &[f32]) -> String {
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        crate::types::fnv64_hex(&bytes)
    }

    fn sample_payload(rank: u32) -> RankCheckpointPayload {
        RankCheckpointPayload {
            rank,
            parameter_shards: vec![NamedBinaryShard {
                name: format!("weights-r{rank}"),
                bytes: vec![rank as u8, 1, 2, 3],
            }],
            optimizer_state_shards: vec![OptimizerStateShard {
                param_id: 100 + rank as u64,
                state_name: "m".to_owned(),
                shard_index: 0,
                shard_count: 1,
                shape: vec![2],
                dtype: DType::Fp32,
                version: 1,
                checksum_hex: value_checksum(&[0.1 + rank as f32, 0.2 + rank as f32]),
                values: vec![0.1 + rank as f32, 0.2 + rank as f32],
            }],
            router_state_shards: vec![NamedBinaryShard {
                name: format!("router-r{rank}"),
                bytes: vec![9, rank as u8],
            }],
        }
    }

    #[test]
    fn optimizer_state_codec_roundtrip_is_lossless() {
        let shard = sample_payload(0).optimizer_state_shards.remove(0);
        let encoded = encode_optimizer_state_shard(&shard);
        let decoded = decode_optimizer_state_shard(&encoded).expect("decode should succeed");
        assert_eq!(shard, decoded);
    }

    #[test]
    fn distributed_checkpoint_roundtrip_restores_payloads() {
        let root = unique_temp_root();
        let repo = DistributedCheckpointRepository::new(&root).expect("repo should initialize");

        let request = CheckpointRequest {
            checkpoint_id: "ckpt-1".to_owned(),
            model_id: "lite-llm-base".to_owned(),
            step: 321,
            world_size: 2,
            optimizer_name: "adamw".to_owned(),
            metadata_version: 1,
        };

        let manifest = repo
            .commit(&request, &[sample_payload(1), sample_payload(0)])
            .expect("commit should succeed");

        let verifier = Fnv64ChecksumVerifier;
        let restored = repo
            .restore("ckpt-1", &verifier)
            .expect("restore should succeed");

        assert_eq!(restored.manifest, manifest);
        assert_eq!(restored.payloads.len(), 2);
        assert_eq!(restored.payloads[0].rank, 0);
        assert_eq!(restored.payloads[1].rank, 1);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn commit_rejects_missing_rank_payloads() {
        let root = unique_temp_root();
        let repo = DistributedCheckpointRepository::new(&root).expect("repo should initialize");

        let request = CheckpointRequest {
            checkpoint_id: "ckpt-2".to_owned(),
            model_id: "lite-llm-base".to_owned(),
            step: 321,
            world_size: 2,
            optimizer_name: "adamw".to_owned(),
            metadata_version: 1,
        };

        let result = repo.commit(&request, &[sample_payload(0)]);
        assert!(result.is_err());
        assert!(!root.join("ckpt-2.tmp").exists());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn restore_detects_shard_corruption() {
        let root = unique_temp_root();
        let repo = DistributedCheckpointRepository::new(&root).expect("repo should initialize");

        let request = CheckpointRequest {
            checkpoint_id: "ckpt-3".to_owned(),
            model_id: "lite-llm-base".to_owned(),
            step: 321,
            world_size: 1,
            optimizer_name: "adamw".to_owned(),
            metadata_version: 1,
        };

        repo.commit(&request, &[sample_payload(0)])
            .expect("commit should succeed");

        let canonical = std::fs::read_to_string(root.join("ckpt-3").join("manifest.trainchk"))
            .expect("manifest should be readable");
        let manifest = DistributedCheckpointManifest::from_canonical_string(&canonical)
            .expect("manifest should parse");

        let target = manifest
            .parameter_shards
            .first()
            .expect("must contain parameter shard");
        let mut path = root.join("ckpt-3");
        for part in target.path.split('/') {
            path.push(part);
        }
        std::fs::write(path, b"CORRUPTED").expect("must write corruption");

        let verifier = Fnv64ChecksumVerifier;
        let restored = repo.restore("ckpt-3", &verifier);
        assert!(restored.is_err());

        let _ = std::fs::remove_dir_all(root);
    }
}
