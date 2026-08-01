// =============================================================================
// amr/mod.rs
//
// static mesh refinement (SMR) over the kernel-native engine.
// a Hierarchy owns one SimStateGeneric + KernelSet per refinement level and
// composes the EXISTING single-level stage loop per level; nothing here
// modifies the single-level engine.
//
// single-coverage cap: ONE refined box per level, ONE FluxRegister per
// coarse-fine level-pair, no patch graph, no clustering. this is SMR;
// multi-patch adaptive AMR is deferred. the `amr` module
// spelling is a code-symbol naming debt and makes no adaptivity claim.
//
//   hierarchy     — Hierarchy + LevelData + the berger-oliger subcycle driver
//   transfer      — cf ghost slabs + prolong/restrict kernel dispatch
//   flux_register — conservative reflux accumulator
//   equilibrium   — the numerical flux of a stationary target, per level, so the reflux
//                   differences DEVIATIONS from that target rather than the state itself
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
    gather_decomposed_hierarchy_tracers, seed_decomposed_hierarchy_tracers,
};
pub use tracer_interface::{
    InterfaceFace, InterfaceTransfer, interface_faces, interface_mass_transfers,
    interface_transport_kernels,
};
pub use transfer::ProlongOrder;
