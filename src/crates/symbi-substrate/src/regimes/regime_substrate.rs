// =============================================================================
// regimes/regime_substrate.rs
//
// `RegimeSubstrate` — the regime -> matched-KernelSet map, so `SimState::substrate()` builds the
// right godunov / constrained-transport KernelSet from the sim alone (no hand-matching the long
// `NewtonianMhdSubstrateKernelSet::<Mem, Sc, D>::new(gamma, cfl, theta, &geom.allocated)` to the
// regime at 100+ call sites). the EOS parameter (gamma for ideal-gas regimes, cs for isothermal)
// comes from `Eos::substrate_param`; CFL + the allocated domain come from the sim.
//
// PURELY ADDITIVE — the explicit `*SubstrateKernelSet::new(..)` path is untouched. each regime
// maps to exactly ONE KernelSet (the `IsoNewtonian` / `IsothermalMhd` regimes are distinct types,
// so the mapping is one-to-one). theta defaults to 1.0 (minmod) via `substrate()`; tune it with
// the `.theta(..)` builder on the returned set.
//
// usage:
//   let sub = sim.substrate().with_solver(Solver::Hlld);   // matched KernelSet, eos/cfl/alloc from sim
//   let sub = sim.substrate().theta(1.5);                  // sharper MC limiter
// =============================================================================

use symbi_algebra::{Domain, OrderedNumeric};
use symbi_ir::algebra::Scalar;
use symbi_xpu::{ExecutionSpace, MemorySpace};

use symbi_geometry::Metric;
use symbi_hydro::eos::Eos;
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::isothermal_mhd::IsothermalMhd;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::regime::Regime;
use symbi_hydro::rhd::Rhd;
use symbi_hydro::rmhd::Rmhd;

use crate::regimes::substrate::IsoSubstrateKernelSet;
use crate::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet;
use crate::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use crate::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use crate::regimes::substrate_rhd::RhdSubstrateKernelSet;
use crate::regimes::substrate_rmhd::RmhdSubstrateKernelSet;
use symbi_sim::state::SimStateGeneric;

/// map a regime to its substrate `KernelSet`, constructed from the run scalars (EOS param, CFL,
/// theta) + the allocated domain. lives in the substrate layer (this crate):
/// the per-regime impls below name concrete kernelsets, and the orphan rule requires the trait
/// that maps the foreign `Regime` types to be local here. `SimSubstrate::substrate()` is the
/// ergonomic front door.
pub trait RegimeSubstrate<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize>:
    Regime<Sc, D>
{
    /// the matched substrate KernelSet type for this regime.
    type KernelSet;
    /// build it. `eos_param` is gamma (ideal-gas regimes) or cs (isothermal), per
    /// `Eos::substrate_param`; hydro sets take theta via the `.theta()` builder, MHD positionally.
    fn make_substrate(eos_param: f64, cfl: f64, theta: f64, alloc: &Domain<D>) -> Self::KernelSet;
}

impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> RegimeSubstrate<Mem, Sc, D>
    for Newtonian
{
    type KernelSet = AdiabaticSubstrateKernelSet<Mem, Sc, D>;
    fn make_substrate(g: f64, cfl: f64, theta: f64, alloc: &Domain<D>) -> Self::KernelSet {
        AdiabaticSubstrateKernelSet::<Mem, Sc, D>::new(g, cfl, alloc).theta(theta)
    }
}

impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> RegimeSubstrate<Mem, Sc, D>
    for IsoNewtonian
{
    type KernelSet = IsoSubstrateKernelSet<Mem, Sc, D>;
    fn make_substrate(cs: f64, cfl: f64, theta: f64, alloc: &Domain<D>) -> Self::KernelSet {
        IsoSubstrateKernelSet::<Mem, Sc, D>::new(cs, cfl, alloc).theta(theta)
    }
}

impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> RegimeSubstrate<Mem, Sc, D>
    for Rhd
{
    type KernelSet = RhdSubstrateKernelSet<Mem, Sc, D>;
    fn make_substrate(g: f64, cfl: f64, theta: f64, alloc: &Domain<D>) -> Self::KernelSet {
        RhdSubstrateKernelSet::<Mem, Sc, D>::new(g, cfl, alloc).theta(theta)
    }
}

impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> RegimeSubstrate<Mem, Sc, D>
    for NewtonianMhd
{
    type KernelSet = NewtonianMhdSubstrateKernelSet<Mem, Sc, D>;
    fn make_substrate(g: f64, cfl: f64, theta: f64, alloc: &Domain<D>) -> Self::KernelSet {
        NewtonianMhdSubstrateKernelSet::<Mem, Sc, D>::new(g, cfl, theta, alloc)
    }
}

impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> RegimeSubstrate<Mem, Sc, D>
    for IsothermalMhd
{
    type KernelSet = IsothermalMhdSubstrateKernelSet<Mem, Sc, D>;
    fn make_substrate(cs: f64, cfl: f64, theta: f64, alloc: &Domain<D>) -> Self::KernelSet {
        IsothermalMhdSubstrateKernelSet::<Mem, Sc, D>::new(cs, cfl, theta, alloc)
    }
}

impl<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize> RegimeSubstrate<Mem, Sc, D>
    for Rmhd
{
    type KernelSet = RmhdSubstrateKernelSet<Mem, Sc, D>;
    fn make_substrate(g: f64, cfl: f64, theta: f64, alloc: &Domain<D>) -> Self::KernelSet {
        RmhdSubstrateKernelSet::<Mem, Sc, D>::new(g, cfl, theta, alloc)
    }
}

// =============================================================================
// SimSubstrate — the `sim.substrate()` ergonomic front door
// =============================================================================

/// `sim.substrate()` builds the matched substrate `KernelSet` from the sim alone — the EOS param
/// (gamma for ideal-gas regimes, cs for isothermal), CFL, and allocated domain are pulled off the
/// sim, so the long `*SubstrateKernelSet::<Mem, Sc, D>::new(..)` need not be hand-matched to the
/// regime. theta defaults to 1.0 (minmod) — tune with the returned set's `.theta(..)`.
///
/// an extension trait (not an inherent method) because `SimStateGeneric` lives in `symbi-sim`:
/// `sim.substrate()` works wherever this trait is in scope (it is in the prelude).
///
/// usage:
///   let sub = sim.substrate().with_solver(Solver::Hlld);   // matched KernelSet, eos/cfl/alloc from sim
///   let sub = sim.substrate().theta(1.5);                  // sharper MC limiter
pub trait SimSubstrate<Mem: MemorySpace, Sc: Scalar + OrderedNumeric, const D: usize>
where
    Self: Sized,
{
    /// the matched substrate KernelSet type for this sim's regime.
    type KernelSet;
    /// build it from the sim's eos param / cfl / allocated domain (theta = 1.0 minmod).
    fn substrate(&self) -> Self::KernelSet;
}

impl<R, const D: usize, const DOF: usize, M, E, S, Mem, Sc> SimSubstrate<Mem, Sc, D>
    for SimStateGeneric<R, D, DOF, M, E, S, Mem, Sc>
where
    R: RegimeSubstrate<Mem, Sc, D>,
    M: Metric<Sc, D>,
    E: Eos<Sc>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    type KernelSet = <R as RegimeSubstrate<Mem, Sc, D>>::KernelSet;
    fn substrate(&self) -> Self::KernelSet {
        <R as RegimeSubstrate<Mem, Sc, D>>::make_substrate(
            self.physics.eos.substrate_param(),
            self.cfl,
            1.0,
            &self.geom.allocated,
        )
    }
}
