// =============================================================================
// symbi-sim/src/lib.rs
//
// the simulation-state core crate: the `FieldStore` /
// `SimState` SoA containers, the sim<->substrate seam (KernelSet / RegimeSubstrate
// traits + the Solver/RegimeKind classification enums), config, the free field
// helpers, checkpoint I/O, and example CLI parsing.
//
// this is the HUB the whole orchestration revolves around, lifted below the
// substrate (`regimes`) and the integrator (`evolve`) so they depend DOWN on it
// — `state.rs` carries no upward edge into `regimes`, which is what lets the
// hub be its own crate.
//
// dependency floor only: algebra / grid / ir / hydro / geometry / xpu / io. it
// names NO concrete kernelset and NO executor — those live above it.
//
// usage:
//  use symbi_sim::state::SimStateGeneric;
//  use symbi_sim::substrate_seam::{KernelSet, RegimeSubstrate, Solver};
// =============================================================================

pub mod stage;
pub mod tracers;
pub mod state;
pub mod substrate_seam;
pub mod driver;
pub mod config;
pub mod hydro_ops;
pub mod run_args;
pub mod checkpoint;
pub mod decomp;
