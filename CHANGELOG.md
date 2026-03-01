# Changelog

All notable changes to `lite-llm-training` are documented in this file.

## [0.1.0] - 2026-03-01
### Added
- Curriculum scheduler and tier expansion phase model.
- Hierarchical load-balancing loss and starvation analysis/intervention controls.
- Optimizer abstraction (`SgdMomentum`, `AdamW`, `Adafactor`) and mixed precision policies.
- Deterministic gradient accumulation and update ordering.
- Sharded optimizer-state codec and distributed checkpoint repository.
- Deterministic replay context model and verification helpers.
- Model evolution/versioning compatibility helpers.
