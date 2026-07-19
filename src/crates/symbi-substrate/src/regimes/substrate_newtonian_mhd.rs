// =============================================================================
// regimes/substrate_newtonian_mhd.rs
//
// the Newtonian-MHD KernelSet is now the `R = NewtonianMhd` instance of the unified
// `MhdSubstrateKernelSet`. thin back-compat alias; the
// implementation (algebraic c2p, inline-magnetosonic HLLE/HLLC/HLLD flux, the shared
// CT stack) lives once in `substrate_mhd` + `mhd_substrate`, driven by `NewtonianMhd::SPEC`
// (`has_energy = true`, `materializes_wave_speeds = false`).
//
// usage:
//  let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(gamma, cfl, theta, &alloc);
// =============================================================================

use symbi_hydro::newtonian_mhd::NewtonianMhd;

use crate::regimes::substrate_mhd::MhdSubstrateKernelSet;

/// a D-dimensional Newtonian-MHD `KernelSet` — the `NewtonianMhd` instance of [`MhdSubstrateKernelSet`].
pub type NewtonianMhdSubstrateKernelSet<Mem, Sc, const D: usize> =
    MhdSubstrateKernelSet<NewtonianMhd, Mem, Sc, D>;

/// back-compat alias for the 3D KernelSet (the original `*3D` name).
pub type NewtonianMhdSubstrateKernelSet3D<Mem, Sc = f64> = NewtonianMhdSubstrateKernelSet<Mem, Sc, 3>;
