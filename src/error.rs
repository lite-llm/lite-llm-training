use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingError {
    InvalidConfig(&'static str),
    InvalidState(&'static str),
    InvalidInput(&'static str),
    ParseError(&'static str),
    IoError(String),
    ParseErrorDynamic(String),
    InvalidConfigDynamic(String),
    MissingShard(String),
    ChecksumMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    VersionIncompatible(String),
    ReplayMismatch(&'static str),
}

pub type TrainingResult<T> = Result<T, TrainingError>;

impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::InvalidState(msg) => write!(f, "invalid state: {msg}"),
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
            Self::ParseErrorDynamic(msg) => write!(f, "parse error: {msg}"),
            Self::InvalidConfigDynamic(msg) => write!(f, "invalid config: {msg}"),
            Self::IoError(msg) => write!(f, "io error: {msg}"),
            Self::MissingShard(path) => write!(f, "missing shard: {path}"),
            Self::ChecksumMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "checksum mismatch for {path}: expected {expected}, got {actual}"
            ),
            Self::VersionIncompatible(msg) => write!(f, "version incompatible: {msg}"),
            Self::ReplayMismatch(msg) => write!(f, "replay mismatch: {msg}"),
        }
    }
}

impl Error for TrainingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl From<std::io::Error> for TrainingError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err.to_string())
    }
}

impl From<serde_json::Error> for TrainingError {
    fn from(err: serde_json::Error) -> Self {
        Self::ParseErrorDynamic(err.to_string())
    }
}
