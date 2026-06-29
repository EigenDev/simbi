// =============================================================================
// amr/mod.rs
//
// static mesh refinement (SMR) over the kernel-native engine (docs/design/21, 22).
// a Hierarchy owns one SimStateGeneric + KernelSet per refinement level and
// composes the EXISTING single-level stage loop per level; nothing here
// modifies the single-level engine.
//
// single-coverage cap: ONE refined box per level, ONE FluxRegister per
// coarse-fine level-pair, no patch graph, no clustering. this is SMR, not
// multi-patch adaptive AMR (deferred — docs/design/21_amr.md). the `amr` module
// spelling is a code-symbol naming debt, not an adaptivity claim.
//
//   hierarchy     — Hierarchy + LevelData + the berger-oliger subcycle driver
//   transfer      — cf ghost slabs + prolong/restrict kernel dispatch
//   flux_register — conservative reflux accumulator
//
// usage:
//  let mut hier = Hierarchy::with_refinement(sim, kernels, &regions, order, make)?;
//  hier.evolve(t_final)?;
// =============================================================================

pub mod emf_register;
pub mod flux_register;
pub mod hierarchy;
pub mod transfer;

pub use emf_register::EmfRegister;
pub use flux_register::FluxRegister;
pub use hierarchy::{
    evolve_hierarchy_decomposed, fine_subgrid, FineSubgrid, Hierarchy, LevelData, RefinementRegion,
};
pub use transfer::ProlongOrder;
