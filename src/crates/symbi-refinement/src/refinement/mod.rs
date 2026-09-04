// =============================================================================
// refinement/mod.rs
//
// fixed mesh refinement over the kernel-native engine.
// a Hierarchy owns one SimStateGeneric + KernelSet per refinement level and
// composes the existing single-level stage loop per level; the single-level
// engine stays intact.
//
// single-coverage cap: one refined box per level, declared at setup and held for
// the run, with one FluxRegister per coarse-fine level-pair. multi-patch
// solution-adaptive refinement lies outside this implementation.
//
//   hierarchy     — Hierarchy + LevelData + the berger-oliger subcycle driver
//   transfer      — cf ghost slabs + prolong/restrict kernel dispatch
//   flux_register — conservative reflux accumulator
//   equilibrium   — the numerical flux of a stationary target, per level, so the reflux
//                   differences deviations from that target rather than the state itself
//
// usage:
//  let mut hier = Hierarchy::with_refinement(sim, kernels, &regions, order, make)?;
//  hier.evolve(t_final)?;
// =============================================================================

pub mod emf_register;
pub mod equilibrium;
pub mod flux_register;
pub mod hierarchy;
pub mod tracer_interface;
pub mod transfer;

pub use emf_register::EmfRegister;
pub use equilibrium::EquilibriumFlux;
pub use flux_register::FluxRegister;
pub use hierarchy::{
    FineSubgrid, Hierarchy, LevelData, RefinementRegion, evolve_hierarchy_decomposed, fine_subgrid,
    gather_decomposed_hierarchy_tracers, seed_decomposed_fine_from_coarse,
    seed_decomposed_hierarchy_tracers,
};
pub use tracer_interface::{
    InterfaceFace, InterfaceTransfer, interface_faces, interface_mass_transfers,
    interface_transport_kernels,
};
pub use transfer::ProlongOrder;
