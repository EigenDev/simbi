// =============================================================================
// symbi-refinement/src/lib.rs
//
// the mesh-refinement crate: a fixed per-level hierarchy declared at setup — the
// per-level `SimStateGeneric` + `KernelSet` tree, the coarse-fine flux/EMF registers,
// and the conservative restriction/prolongation transfer operators.
//
// sits above `symbi-sim` (per-level sims + the shared driver primitives it reuses)
// and `symbi-substrate` (it dispatches each level's KernelSet); the `symbi` top crate
// drives it. it is a sibling of the single-grid `evolve` driver — both consume the
// same `symbi_sim::driver` primitives dry, neither depends on the other.
//
// usage:
//  use symbi_refinement::refinement::Hierarchy;
// =============================================================================

// the fixed per-level refinement hierarchy over per-level KernelSets.
pub mod refinement;
