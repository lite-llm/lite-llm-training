# lite-llm-training

Training runtime crate for Lite LLM (`SPEC-031` to `SPEC-040`).

## Overview
Implements deterministic training primitives including curriculum scheduling, load balancing, optimization, precision management, distributed checkpointing, and GPU-accelerated training with automatic mixed precision.

This crate provides the complete training stack: curriculum tier expansion scheduling with deterministic phases, hierarchical load-balancing with starvation controls, optimizer abstractions (AdamW, Adafactor, SGD momentum) with precision policies, accumulation ordering guarantees, sharded optimizer state and distributed checkpointing, deterministic replay model for auditability, model evolution and compatibility checks, GPU training ops with AMP and gradient scaling, and async checkpoint I/O with deterministic fingerprinting.

## Features

### Feature Flag: `default` (empty)
No optional features. GPU backend is controlled by `lite-llm-inference`'s `cuda` feature flag.

## Dependencies
| Crate | Version | Purpose |
|-------|---------|---------|
| rand | 0.8 | Seeded tie-breaking and deterministic sampling |
| tokio | 1 | Async runtime for checkpoint I/O |
| serde | 1.0 | Serialization for checkpoint metadata |
| serde_json | 1.0 | JSON encoding for checkpoint state |
| lite-llm-inference | 0.1.0 (path) | GPU tensor backend for training ops |

## Key Modules
- `curriculum` — tier expansion scheduler, integration schedules, expansion windows
- `load_balancing` — hierarchical load-balancing loss with routing usage tracking
- `starvation` — expert starvation analysis, intervention recommendations
- `optimizer` — AdamW, Adafactor, SGD momentum with state shards and fingerprints
- `precision` — mixed precision policy, loss scaling, type casting
- `accumulation` — gradient accumulation with deterministic micro-batch scheduling
- `checkpoint` — distributed checkpoint repository with rank payloads and checksums
- `gpu_training` — GPU training step with AMP, async checkpoint I/O, metrics tracking
- `replay` — deterministic replay context with parameter checksums
- `versioning` — model evolution, compatibility assessment, migration planning
- `trainer` — core trainer with forward/backward/loss computation
- `types` — shared type contracts (`ExpertKey`, `TierId`)
- `error` — training error model

## Public API
### Core Types
- `CurriculumScheduler` — deterministic tier expansion with phases and windows
- `TierExpansionPlan` — expansion plan with new tier, window, and integration schedule
- `ExpansionWindow` — preparation, isolation, integration, joint training step counts
- `IntegrationSchedule` — linear or exponential integration weighting
- `HierarchicalRoutingUsage` — routing usage metrics for load balancing
- `LoadBalancingConfig` / `LoadBalancingMetrics` — load balancing configuration
- `StarvationConfig` / `StarvationReport` / `StarvationIntervention` — starvation analysis
- `AdamW` / `Adafactor` / `SgdMomentum` — optimizer implementations
- `OptimizerStateShard` — sharded optimizer state with checksums
- `MixedPrecisionPolicy` / `LossScaler` / `PrecisionMode` — precision management
- `AccumulationConfig` / `AccumulationState` / `MicroBatchSchedule` — gradient accumulation
- `DistributedCheckpointManifest` / `DistributedCheckpointRepository` — distributed checkpointing
- `GpuTrainingStep` — GPU training step with AMP and gradient scaling
- `TrainingMetrics` / `StepMetrics` — per-step training metrics with rolling averages
- `ReplayContext` / `ReplayEvent` / `ReplayEventKind` — deterministic replay
- `ModelVersion` / `CompatibilityReport` / `MigrationPlan` — model evolution
- `Trainer` / `TrainerConfig` — core training loop
- `Tensor` (re-exported from inference) — tensor type for optimizer state

### Core Functions
- `hierarchical_load_balancing_loss()` — compute load-balancing loss
- `analyze_starvation()` / `recommend_interventions()` — starvation analysis
- `optimizer_state_fingerprint()` — deterministic optimizer state fingerprint
- `compute_checkpoint_fingerprint()` — deterministic checkpoint fingerprint
- `save_checkpoint_async()` / `load_checkpoint_async()` — async checkpoint I/O
- `apply_updates_deterministic()` — deterministic gradient application
- `seed_for_micro_batch()` — deterministic micro-batch seeding
- `parameter_checksum()` — parameter checksum for replay validation
- `cross_entropy_loss()` / `compute_gradients()` / `adam_update()` — training primitives
- `assert_loadable()` / `assess_compatibility()` / `build_migration_plan()` — version checks

### Traits
- `Optimizer` — optimizer interface with state shard management

## Quick Start
```rust
use lite_llm_training::{
    CurriculumScheduler, TierExpansionPlan, ExpansionWindow,
    IntegrationSchedule, AdamW, Trainer, TrainerConfig,
    AccumulationConfig, AccumulationState,
};

// Create curriculum scheduler
let scheduler = CurriculumScheduler::new(
    TierExpansionPlan {
        new_tier: 2,
        window: ExpansionWindow {
            start_step: 0,
            preparation_steps: 10,
            isolation_steps: 5,
            integration_steps: 20,
            joint_training_steps: 100,
        },
        integration_schedule: IntegrationSchedule::Linear,
        deterministic_seed: 42,
    },
    world_size: 4,
);

// Create optimizer
let optimizer = AdamW::new(
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    weight_decay: 0.01,
);

// Create trainer
let config = TrainerConfig {
    learning_rate: 0.001,
    max_steps: 1000,
};
let mut trainer = Trainer::new(config, optimizer);

// Run training step
let loss = trainer.step(&gradients, step_num, learning_rate)?;
```

## Running Tests
```bash
cargo fmt
cargo test
```

## Architecture
This crate implements the training layer for the lite-llm platform. GPU training ops (`gpu_training` module) wrap optimizers with automatic mixed precision and gradient scaling, enabling efficient mixed-precision training. Async checkpoint I/O (`save_checkpoint_async`, `load_checkpoint_async`) persists training state through the async storage backend. The distributed checkpoint repository integrates with `lite-llm-storage` for persistent checkpoint storage. It integrates with `lite-llm-inference` for model evaluation and with `lite-llm` orchestrator for training mode entrypoints.

## Changelog
See `CHANGELOG.md`.

## License
See `LICENSE`.
