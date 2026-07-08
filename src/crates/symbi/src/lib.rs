// =============================================================================
// symbi — the user-facing crate.
//
// re-exports the core crates and provides the simulation framework.
// the user writes:
//   let mut sim = SimState::build(Newtonian, eos, Cartesian)
//       .cells([n]).spacing([dx]).boundaries(BoundaryType::Outflow)
//       .allocate()?.set_initial(|x| prim_at(x)).build();
//   let sub = sim.substrate();
//   evolve(&mut sim, &sub, t_final)?;
// =============================================================================

// allow kernel macro to emit ::symbi:: paths within this crate
extern crate self as symbi;

// simulation framework (state, evolution driver, GPU ops)
pub mod sim;

// one-import convenience surface for users: `use symbi::prelude::*;`
pub mod prelude;

// tiled parallel dispatch (DomainForEach trait for #[symbi::kernel(coord)])
pub mod dispatch;

// the substrate (KernelSets + regime->KernelSet map + dispatch) and the kernel
// support utilities live in the `symbi-substrate` crate (docs/design/41);
// re-exported at the `crate::regimes` / `crate::kernels` paths so evolve,
// refinement, the prelude, and downstream callers are untouched.
pub use symbi_substrate::{kernels, regimes};

// ---- re-exports ----
pub use symbi_algebra::*;
pub use dispatch::{DomainForEach, KernelInfo, parallel_reduce_1d, parallel_reduce_2d, parallel_reduce_3d};
pub use symbi_geometry;
pub use symbi_hydro;
pub use symbi_grid;
pub use symbi_xpu;
