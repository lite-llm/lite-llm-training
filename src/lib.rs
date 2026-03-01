pub mod accumulation;
pub mod checkpoint;
pub mod curriculum;
pub mod error;
pub mod load_balancing;
pub mod optimizer;
pub mod precision;
pub mod replay;
pub mod starvation;
pub mod types;
pub mod versioning;

pub use accumulation::{
    apply_updates_deterministic, seed_for_micro_batch, AccumulationConfig, AccumulationState,
    MicroBatchSchedule,
};
pub use checkpoint::{
    decode_optimizer_state_shard, encode_optimizer_state_shard, BinaryShardRef, CheckpointRequest,
    ChecksumVerifier, DistributedCheckpointManifest, DistributedCheckpointRepository,
    Fnv64ChecksumVerifier, NamedBinaryShard, OptimizerStateShardRef, RankCheckpointPayload,
    RestoredDistributedCheckpoint,
};
pub use curriculum::{
    CurriculumScheduler, ExpansionDirective, ExpansionPhase, ExpansionWindow, IntegrationSchedule,
    TierExpansionPlan,
};
pub use error::{TrainingError, TrainingResult};
pub use load_balancing::{
    hierarchical_load_balancing_loss, HierarchicalRoutingUsage, LoadBalancingConfig,
    LoadBalancingMetrics,
};
pub use optimizer::{
    optimizer_state_fingerprint, Adafactor, AdamW, DType, Optimizer, OptimizerStateShard,
    SgdMomentum, Tensor,
};
pub use precision::{
    cast_values, LossScaler, LossScalingPolicy, MixedPrecisionPolicy, PrecisionMode,
};
pub use replay::{parameter_checksum, ReplayContext, ReplayEvent, ReplayEventKind};
pub use starvation::{
    analyze_starvation, no_update_probability_upper_bound, recommend_interventions,
    StarvationConfig, StarvationIntervention, StarvationReport, TierAssignmentStats,
};
pub use types::{fnv64_hex, ExpertKey, TierId};
pub use versioning::{
    assert_loadable, assess_compatibility, build_migration_plan, CompatibilityLevel,
    CompatibilityReport, EvolutionKind, MigrationPlan, ModelIdentifier, ModelVersion,
};
