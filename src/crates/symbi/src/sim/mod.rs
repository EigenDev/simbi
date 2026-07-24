// =============================================================================
// sim/mod.rs
//
// simulation framework:
//   state / substrate_seam / config / hydro_ops / run_args / checkpoint
//                 — re-exported from the `symbi-sim` crate: the FieldStore hub +
//                   the sim<->substrate seam, lifted below the substrate.
//   evolve        — the evolve driver (drives the KernelSet trait); stays here, it depends
//                   UP on the substrate (regimes) + the executor.
//   refinement    — static mesh refinement (SMR) hierarchy; stays here for the same reason.
// =============================================================================

// the sim-state core lives in `symbi-sim`; re-exported at the `crate::sim::*`
// paths so every downstream caller (regimes, dispatch, prelude, tests) is untouched.
pub use symbi_sim::{
    checkpoint, config, decomp, driver, hydro_ops, run_args, state, substrate_seam, tracers,
};

// SMR lives in the `symbi-amr` crate; re-exported at the `crate::sim::refinement`
// path so downstream callers (tests, examples) are untouched.
pub use symbi_amr::refinement;

pub mod evolve;
