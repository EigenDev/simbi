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
pub mod bondi;
pub mod history;
pub mod penalize;
pub mod sdf;
pub mod body_delta;
pub mod collection;
pub mod drain;
pub mod effects;
pub mod sink;
pub mod motion;
pub mod diagnostics;

pub use body::{Body, BodyKind};
pub use bondi::{bondi_profile, mdot_bondi, sonic_radius, BondiState};
pub use history::BodyHistory;
pub use penalize::{penalize_cell, BodyKin, Property, Relax};
pub use sdf::SdfExpr;
pub use body_delta::BodyDelta;
pub use drain::{drain_body_cell, drain_cell, drain_mask, drain_timescale, sound_speed_from_cons};
pub use collection::{BodyCollection, BinaryParams, ReferenceFrame, MAX_BODIES};
pub use effects::{CellGeometry, grav_source, rigid_source};
pub use sink::accretion_coefficient;
pub use motion::{rotate_2d, rotate_3d, advance_binary, apply_body_deltas, keplerian_binary};
pub use diagnostics::DiagnosticAccumulator;
