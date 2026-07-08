// =============================================================================
// symbi-amr/src/lib.rs
//
// the static mesh refinement (SMR) crate (docs/design/41): the per-level
// `SimStateGeneric` + `KernelSet` hierarchy, the coarse-fine flux/EMF registers,
// and the conservative restriction/prolongation transfer operators.
//
// sits ABOVE `symbi-sim` (per-level sims + the shared driver primitives it reuses)
// and `symbi-substrate` (it dispatches each level's KernelSet); the `symbi` top crate
// drives it. it is a sibling of the single-grid `evolve` driver — both consume the
// same `symbi_sim::driver` primitives DRY, neither depends on the other.
//
// usage:
//  use symbi_amr::refinement::SmrHierarchy;
// =============================================================================

// static mesh refinement (SMR) hierarchy over per-level KernelSets.
pub mod refinement;
