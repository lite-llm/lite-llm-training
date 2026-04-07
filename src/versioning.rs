use std::collections::BTreeSet;

use crate::error::{TrainingError, TrainingResult};
use crate::types::TierId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModelVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ModelVersion {
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    pub fn as_string(self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIdentifier {
    pub name: String,
    pub version: ModelVersion,
}

impl ModelIdentifier {
    pub fn parse(value: &str) -> TrainingResult<Self> {
        let (name, version_raw) = value.rsplit_once("-v").ok_or(TrainingError::ParseError(
            "model identifier must include -v<semver>",
        ))?;

        if name.trim().is_empty() {
            return Err(TrainingError::ParseError("model name must not be empty"));
        }

        let parts: Vec<&str> = version_raw.split('.').collect();
        if parts.len() != 3 {
            return Err(TrainingError::ParseError(
                "model version must use major.minor.patch",
            ));
        }

        let version = ModelVersion {
            major: parts[0]
                .parse::<u32>()
                .map_err(|_| TrainingError::ParseError("invalid major version"))?,
            minor: parts[1]
                .parse::<u32>()
                .map_err(|_| TrainingError::ParseError("invalid minor version"))?,
            patch: parts[2]
                .parse::<u32>()
                .map_err(|_| TrainingError::ParseError("invalid patch version"))?,
        };

        Ok(Self {
            name: name.to_owned(),
            version,
        })
    }

    pub fn as_string(&self) -> String {
        format!("{}-v{}", self.name, self.version.as_string())
    }

    pub fn evolve(&self, kind: EvolutionKind) -> Self {
        let version = kind.bump(self.version);
        Self {
            name: self.name.clone(),
            version,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvolutionKind {
    TierExpansion,
    ArchitectureChange,
    HyperparameterAdjustment,
    BugFix,
}

impl EvolutionKind {
    pub fn bump(self, current: ModelVersion) -> ModelVersion {
        match self {
            Self::TierExpansion => ModelVersion::new(current.major, current.minor + 1, 0),
            Self::ArchitectureChange => ModelVersion::new(current.major + 1, 0, 0),
            Self::HyperparameterAdjustment => current,
            Self::BugFix => ModelVersion::new(current.major, current.minor, current.patch + 1),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityLevel {
    FullyCompatible,
    ForwardCompatible,
    NeedsMigration,
    Incompatible,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatibilityReport {
    pub level: CompatibilityLevel,
    pub reason: String,
    pub ignored_tiers: Vec<TierId>,
    pub migration_required: bool,
}

pub fn assess_compatibility(
    runtime: &ModelIdentifier,
    checkpoint: &ModelIdentifier,
    checkpoint_tiers: &[TierId],
    runtime_supported_tiers: &[TierId],
) -> CompatibilityReport {
    if runtime.name != checkpoint.name {
        return CompatibilityReport {
            level: CompatibilityLevel::Incompatible,
            reason: "model names differ".to_owned(),
            ignored_tiers: Vec::new(),
            migration_required: false,
        };
    }

    if checkpoint.version.major > runtime.version.major {
        return CompatibilityReport {
            level: CompatibilityLevel::Incompatible,
            reason: "checkpoint major version is newer than runtime".to_owned(),
            ignored_tiers: Vec::new(),
            migration_required: false,
        };
    }

    if checkpoint.version.major < runtime.version.major {
        return CompatibilityReport {
            level: CompatibilityLevel::NeedsMigration,
            reason: "checkpoint major version is older than runtime".to_owned(),
            ignored_tiers: Vec::new(),
            migration_required: true,
        };
    }

    let supported: BTreeSet<TierId> = runtime_supported_tiers.iter().copied().collect();
    let mut ignored_tiers = checkpoint_tiers
        .iter()
        .copied()
        .filter(|tier| !supported.contains(tier))
        .collect::<Vec<TierId>>();
    ignored_tiers.sort();

    if checkpoint.version.minor > runtime.version.minor {
        return CompatibilityReport {
            level: CompatibilityLevel::ForwardCompatible,
            reason: "checkpoint minor version is newer; unknown tiers may be ignored".to_owned(),
            ignored_tiers,
            migration_required: false,
        };
    }

    CompatibilityReport {
        level: CompatibilityLevel::FullyCompatible,
        reason: "checkpoint is loadable without migration".to_owned(),
        ignored_tiers,
        migration_required: false,
    }
}

pub fn assert_loadable(report: &CompatibilityReport) -> TrainingResult<()> {
    match report.level {
        CompatibilityLevel::FullyCompatible | CompatibilityLevel::ForwardCompatible => Ok(()),
        CompatibilityLevel::NeedsMigration | CompatibilityLevel::Incompatible => {
            Err(TrainingError::VersionIncompatible(report.reason.clone()))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationPlan {
    pub from: ModelVersion,
    pub to: ModelVersion,
    pub steps: Vec<String>,
}

pub fn build_migration_plan(from: ModelVersion, to: ModelVersion) -> MigrationPlan {
    let mut steps = Vec::new();

    if from.major != to.major {
        steps.push("migrate checkpoint schema for major-version architecture changes".to_owned());
    }
    if from.minor != to.minor {
        steps.push("reconcile TierSet and ignore or initialize unknown tiers".to_owned());
    }
    if from.patch != to.patch {
        steps.push("apply deterministic patch-level compatibility checks".to_owned());
    }

    MigrationPlan { from, to, steps }
}

#[cfg(test)]
mod tests {
    use super::{
        assert_loadable, assess_compatibility, CompatibilityLevel, EvolutionKind, ModelIdentifier,
        ModelVersion,
    };

    #[test]
    fn model_identifier_parse_and_format_roundtrip() {
        let id = ModelIdentifier::parse("lite-llm-base-v2.1.3").expect("parse should succeed");
        assert_eq!(id.name, "lite-llm-base");
        assert_eq!(id.version, ModelVersion::new(2, 1, 3));
        assert_eq!(id.as_string(), "lite-llm-base-v2.1.3");
    }

    #[test]
    fn evolution_bumps_versions_as_expected() {
        let base = ModelIdentifier::parse("lite-llm-base-v1.2.3").expect("parse should succeed");

        assert_eq!(
            base.evolve(EvolutionKind::TierExpansion).version,
            ModelVersion::new(1, 3, 0)
        );
        assert_eq!(
            base.evolve(EvolutionKind::ArchitectureChange).version,
            ModelVersion::new(2, 0, 0)
        );
        assert_eq!(
            base.evolve(EvolutionKind::BugFix).version,
            ModelVersion::new(1, 2, 4)
        );
    }

    #[test]
    fn compatibility_reports_forward_minor_with_ignored_tiers() {
        let runtime = ModelIdentifier::parse("lite-llm-base-v1.2.0").expect("parse should succeed");
        let checkpoint =
            ModelIdentifier::parse("lite-llm-base-v1.3.0").expect("parse should succeed");

        let report = assess_compatibility(&runtime, &checkpoint, &[1, 2, 3], &[1, 2]);
        assert_eq!(report.level, CompatibilityLevel::ForwardCompatible);
        assert_eq!(report.ignored_tiers, vec![3]);
        assert!(assert_loadable(&report).is_ok());
    }

    #[test]
    fn compatibility_rejects_major_mismatch() {
        let runtime = ModelIdentifier::parse("lite-llm-base-v2.0.0").expect("parse should succeed");
        let checkpoint =
            ModelIdentifier::parse("lite-llm-base-v1.9.9").expect("parse should succeed");

        let report = assess_compatibility(&runtime, &checkpoint, &[1], &[1]);
        assert_eq!(report.level, CompatibilityLevel::NeedsMigration);
        assert!(assert_loadable(&report).is_err());
    }
}
