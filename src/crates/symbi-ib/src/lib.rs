// =============================================================================
// lib.rs
//
// immersed boundary body system. provides discrete physical objects
// (black holes, planets, rigid spheres) embedded in a fluid grid,
// with per-cell source terms for gravity, accretion, and rigid forcing.
//
// usage:
//   use symbi_ib::{Body, BodyCollection, CellGeometry, grav_source};
//   let body = Body::black_hole(0, pos, vel, mass, radius, ...);
//   let (src, delta) = grav_source(&body, &prim, &metric, &cell, gamma);
// =============================================================================

pub mod body;
pub mod body_delta;
pub mod collection;
pub mod effects;
pub mod sink;
pub mod motion;
pub mod diagnostics;

pub use body::{Body, BodyKind};
pub use body_delta::BodyDelta;
pub use collection::{BodyCollection, BinaryParams, ReferenceFrame, MAX_BODIES};
pub use effects::{CellGeometry, grav_source, accretion_source, rigid_source};
pub use sink::{accretion_coefficient, sink_weight, WeightedSums, SinkProperties, compute_sink_properties};
pub use motion::{rotate_2d, rotate_3d, advance_binary, apply_body_deltas, keplerian_binary};
pub use diagnostics::DiagnosticAccumulator;
