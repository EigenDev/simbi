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
// paths, which is where regimes, dispatch, the prelude, and tests name it.
pub use symbi_sim::{
    checkpoint, decomp, driver, hydro_ops, state, substrate_seam, tracers,
};

// SMR lives in the `symbi-amr` crate; re-exported at the `crate::sim::refinement`
// path, which is where tests and examples name it.
pub use symbi_amr::refinement;

pub mod evolve;
