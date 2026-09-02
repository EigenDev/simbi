// =============================================================================
// algebra.rs
//
// the carrier constitution lives in the foundation crate `symbi-carrier`;
// this module re-exports it so compiler-side code keeps the
// `symbi_ir::algebra::*` paths, and holds the render-policy vocabulary the
// emit backends select on.
//
// usage:
//  use symbi_ir::algebra::Scalar;      // == symbi_carrier::Scalar
//  let policy = RenderPolicy::default();
// =============================================================================

pub use symbi_carrier::{Mask, Scalar, Selectable, SourceLoc, laws, source_loc};

// =============================================================================
// section 6 — RenderPolicy: natural transformation across Target homomorphisms.
//
// the same IR renders multiple ways. each render is a homomorphism into a
// distinct source-code algebra. RenderPolicy selects which.
// =============================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RenderPolicy {
    /// production — minified: anonymous temporaries, comment-free output. fastest
    /// downstream compile, smallest binary. the default for `cargo build --release`.
    Minified,
    /// debug / audit — preserves names, source-loc comments, section headers.
    /// for inspection only.
    Audit,
    /// reserved. graph -> LaTeX is a non-trivial pretty-printer awaiting a
    /// documentation-pipeline consumer. emit returns
    /// `Err(RenderPolicyNotImplemented)` for this variant until a consumer
    /// earns it (rent test). production paths select an implemented variant.
    Latex,
}

/// returned by an emitter when the caller selects a `RenderPolicy` that is
/// still reserved (`RenderPolicy::Latex`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RenderPolicyNotImplemented {
    pub policy: RenderPolicy,
}

impl Default for RenderPolicy {
    fn default() -> Self {
        RenderPolicy::Minified
    }
}

