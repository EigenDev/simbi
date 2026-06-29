// =============================================================================
// error.rs
//
// the I/O layer's structured error type. replaces the `Result<(), String>`
// pattern that pervaded `sim::checkpoint` with a proper enum, so callers can
// match on the failure mode (missing path? type mismatch? backend specific?)
// instead of grep-string-comparing.
// =============================================================================

use std::fmt;

/// I/O layer errors. one variant per recoverable failure mode plus a
/// catch-all `Backend` for whatever the underlying writer surfaces.
#[derive(Debug)]
pub enum IoError {
    /// the underlying backend (HDF5, JSON, ...) raised a native error.
    /// the string is the backend's own message verbatim — no parsing.
    Backend(String),
    /// a required path is missing in the file. e.g., `load_checkpoint` on
    /// a file with no `level_0/conserved` group.
    MissingPath(String),
    /// the requested type doesn't match what the file holds. e.g., reading
    /// a `time` attribute as `i64` when it was written as `f64`.
    TypeMismatch {
        path: String,
        expected: &'static str,
        actual: &'static str,
    },
    /// a dataset's shape disagreed with the destination buffer.
    ShapeMismatch {
        path: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// a path tried to overwrite an existing node.
    AlreadyExists(String),
    /// raw I/O (filesystem-level).
    Io(std::io::Error),
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(m) => write!(f, "backend error: {m}"),
            Self::MissingPath(p) => write!(f, "missing path: '{p}'"),
            Self::TypeMismatch {
                path,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "type mismatch at '{path}': expected {expected}, got {actual}"
                )
            }
            Self::ShapeMismatch {
                path,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "shape mismatch at '{path}': expected {expected:?}, got {actual:?}"
                )
            }
            Self::AlreadyExists(p) => write!(f, "node already exists at '{p}'"),
            Self::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for IoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

impl From<std::io::Error> for IoError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, IoError>;
