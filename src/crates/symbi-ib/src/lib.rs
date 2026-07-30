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
pub mod body_delta;
pub mod bond;
pub mod bondi;
pub mod collection;
pub mod contact;
pub mod diagnostics;
pub mod drain;
pub mod excise;
pub mod gravity;
pub mod history;
pub mod motion;
pub mod penalize;
pub mod sdf;
pub mod shell_flux;

pub use body::{Body, BodyKind, BodySpec, MagneticSpec, SurfaceSpec};
pub use body_delta::BodyDelta;
pub use bond::{
    Bond, BondMaterial, ExternalLoad, FragmentPhysics, advance_bonded, bond_potential_energy,
};
pub use bondi::{BondiState, accretion_coefficient, bondi_profile, mdot_bondi, sonic_radius};
pub use collection::{BinaryParams, BodyCollection, MAX_SOURCE_BODIES, ReferenceFrame};
pub use contact::{ContactMaterial, Contacts};
pub use diagnostics::DiagnosticAccumulator;
pub use drain::{drain_body_cell, drain_cell, drain_mask, drain_timescale, sound_speed_from_cons};
pub use gravity::{MutualGravity, gravitational_potential_energy};
pub use history::BodyHistory;
pub use motion::{advance_binary, apply_body_deltas, keplerian_binary, rotate_2d, rotate_3d};
pub use penalize::{BodyKin, Property, Relax, moment, omega_cross, penalize_cell};
pub use sdf::SdfExpr;
pub use shell_flux::{FaceFlux, shell_accretion};
