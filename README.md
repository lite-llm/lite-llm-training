# lite-llm-training

Training runtime crate for Lite LLM (`SPEC-031` to `SPEC-040`).

## Scope
Implements deterministic training primitives:

- curriculum tier expansion scheduling (`SPEC-031`)
- hierarchical load-balancing and starvation controls (`SPEC-032`, `SPEC-033`)
- optimizer abstraction, precision policy, and accumulation ordering (`SPEC-034`..`SPEC-036`)
- sharded optimizer state and distributed checkpointing (`SPEC-037`, `SPEC-038`)
- deterministic replay model (`SPEC-039`)
- model evolution and compatibility checks (`SPEC-040`)

## Modules
- `src/curriculum.rs`
- `src/load_balancing.rs`
- `src/starvation.rs`
- `src/optimizer.rs`
- `src/precision.rs`
- `src/accumulation.rs`
- `src/checkpoint.rs`
- `src/replay.rs`
- `src/versioning.rs`
- `src/types.rs`
- `src/error.rs`

## Build and Test
```bash
cargo fmt
cargo test
```

## Documentation
- System docs: `../lite-llm-docs/README.md`
- Testing guidance: `../lite-llm-docs/operations/testing-and-ci.md`

## Changelog
See `CHANGELOG.md`.

## License
See `LICENSE`.
