// =============================================================================
// attr.rs
//
// `Attr` — a typed scalar metadata value. unifies the things that today the
// checkpoint writer takes as `(&str, &str)` — gamma is f64, iteration is u64,
// regime is a String — into one enum. coupled with `Metadata::with(k, v)`,
// callers write `extras.with("gm", 1.0).with("problem", "kepler")` and never
// `let x_s = x.to_string()` again.
// =============================================================================

/// a typed metadata value. construction goes through `From<T> for Attr` for
/// every common primitive, so call sites can pass naked literals.
#[derive(Debug, Clone, PartialEq)]
pub enum Attr {
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    Str(String),
}

impl Attr {
    /// a `&'static str` discriminant — handy for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::I64(_)  => "i64",
            Self::U64(_)  => "u64",
            Self::F64(_)  => "f64",
            Self::Str(_)  => "str",
        }
    }
}

// ---- ergonomic From impls so call sites pass naked values ----

impl From<bool> for Attr { fn from(v: bool) -> Self { Self::Bool(v) } }
impl From<i32>  for Attr { fn from(v: i32)  -> Self { Self::I64(v as i64) } }
impl From<i64>  for Attr { fn from(v: i64)  -> Self { Self::I64(v) } }
impl From<u32>  for Attr { fn from(v: u32)  -> Self { Self::U64(v as u64) } }
impl From<u64>  for Attr { fn from(v: u64)  -> Self { Self::U64(v) } }
impl From<usize> for Attr { fn from(v: usize) -> Self { Self::U64(v as u64) } }
impl From<f32>  for Attr { fn from(v: f32)  -> Self { Self::F64(v as f64) } }
impl From<f64>  for Attr { fn from(v: f64)  -> Self { Self::F64(v) } }
impl From<&str> for Attr { fn from(v: &str) -> Self { Self::Str(v.to_string()) } }
impl From<String> for Attr { fn from(v: String) -> Self { Self::Str(v) } }
impl From<&String> for Attr { fn from(v: &String) -> Self { Self::Str(v.clone()) } }

// ---- typed extractors with proper TypeMismatch errors ----

use crate::error::{IoError, Result};

impl Attr {
    pub fn as_f64(&self, path: &str) -> Result<f64> {
        match self {
            Self::F64(v) => Ok(*v),
            Self::I64(v) => Ok(*v as f64), // safe widening
            Self::U64(v) => Ok(*v as f64),
            other => Err(IoError::TypeMismatch {
                path: path.into(), expected: "f64", actual: other.type_name(),
            }),
        }
    }
    pub fn as_i64(&self, path: &str) -> Result<i64> {
        match self {
            Self::I64(v) => Ok(*v),
            Self::U64(v) => Ok(*v as i64),
            other => Err(IoError::TypeMismatch {
                path: path.into(), expected: "i64", actual: other.type_name(),
            }),
        }
    }
    pub fn as_u64(&self, path: &str) -> Result<u64> {
        match self {
            Self::U64(v) => Ok(*v),
            Self::I64(v) if *v >= 0 => Ok(*v as u64),
            other => Err(IoError::TypeMismatch {
                path: path.into(), expected: "u64", actual: other.type_name(),
            }),
        }
    }
    pub fn as_bool(&self, path: &str) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            other => Err(IoError::TypeMismatch {
                path: path.into(), expected: "bool", actual: other.type_name(),
            }),
        }
    }
    pub fn as_str<'a>(&'a self, path: &str) -> Result<&'a str> {
        match self {
            Self::Str(s) => Ok(s.as_str()),
            other => Err(IoError::TypeMismatch {
                path: path.into(), expected: "str", actual: other.type_name(),
            }),
        }
    }
}

// =============================================================================
// `Metadata` — typed key/value bag built via fluent `.with(key, value)`. the
// answer to "no more to_string in examples". any value that impls `Into<Attr>`
// (every common primitive does) goes in unchanged.
// =============================================================================

/// builder-style typed metadata bag. order-preserving so the resulting file
/// has a deterministic attribute layout.
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    entries: Vec<(String, Attr)>,
}

impl Metadata {
    pub fn new() -> Self { Self::default() }

    /// append (key, value). returns `self` for chaining:
    /// ```ignore
    /// let extras = Metadata::new()
    ///     .with("problem", "kepler")
    ///     .with("ring_r0", 1.0)
    ///     .with("gm", 1.0);
    /// ```
    pub fn with(mut self, key: impl Into<String>, value: impl Into<Attr>) -> Self {
        self.entries.push((key.into(), value.into()));
        self
    }

    /// in-place push variant; equivalent to `with(..)` but for already-owned bags.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<Attr>) {
        self.entries.push((key.into(), value.into()));
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Attr)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn get(&self, key: &str) -> Option<&Attr> {
        self.entries.iter().find_map(|(k, v)| (k == key).then_some(v))
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}

impl<'a> IntoIterator for &'a Metadata {
    type Item = (&'a str, &'a Attr);
    type IntoIter = std::iter::Map<
        std::slice::Iter<'a, (String, Attr)>,
        fn(&'a (String, Attr)) -> (&'a str, &'a Attr),
    >;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }
}
