// =============================================================================
// lib.rs
//
// immersed boundary body system: discrete physical objects (black holes,
// planets, rigid spheres) embedded in a fluid grid. body kinematics live on
// `Body`/`BodyKind`; surface physics is the volume-penalized property algebra
// (`penalize`) over exact signed-distance geometry (`sdf`).
//
// usage:
//   use symbi_ib::{Body, BodyCollection, Property, Relax, penalize_cell};
//   let body = Body::black_hole(0, pos, vel, mass, radius, ...);
// =============================================================================

pub mod body;
pub mod bondi;
pub mod excise;
pub mod history;
pub mod penalize;
pub mod sdf;
pub mod body_delta;
pub mod collection;
pub mod drain;
pub mod motion;
pub mod diagnostics;
pub mod shell_flux;

pub use body::{Body, BodyKind, BodySpec, MagneticSpec, SurfaceSpec};
pub use bondi::{accretion_coefficient, bondi_profile, mdot_bondi, sonic_radius, BondiState};
pub use excise::{onion_fill_cell, onion_pass_count};
pub use history::BodyHistory;
pub use penalize::{moment, omega_cross, penalize_cell, BodyKin, Property, Relax};
pub use sdf::SdfExpr;
pub use body_delta::BodyDelta;
pub use drain::{drain_body_cell, drain_cell, drain_mask, drain_timescale, sound_speed_from_cons};
pub use collection::{BodyCollection, BinaryParams, ReferenceFrame, MAX_BODIES};
pub use motion::{rotate_2d, rotate_3d, advance_binary, apply_body_deltas, keplerian_binary};
pub use diagnostics::DiagnosticAccumulator;
pub use shell_flux::{shell_accretion, FaceFlux};
