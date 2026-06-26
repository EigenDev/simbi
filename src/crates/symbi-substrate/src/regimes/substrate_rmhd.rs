// =============================================================================
// regimes/substrate_rmhd.rs
//
// the RMHD KernelSet is now the `R = Rmhd` instance of the unified
// `MhdSubstrateKernelSet` (docs/design/35 R1). this module is a thin back-compat
// alias; the implementation (KKC c2p, quartic-wave-speed HLLE/HLLC flux, the CT
// stack) lives once in `substrate_mhd` + `mhd_substrate`, driven by `Rmhd::SPEC`.
//
// usage:
//  let sub = RmhdSubstrateKernelSet::<HostMemory, f64, 3>::new(gamma, cfl, theta, &alloc);
// =============================================================================

use symbi_hydro::rmhd::Rmhd;

use crate::regimes::substrate_mhd::MhdSubstrateKernelSet;

/// a D-dimensional RMHD `KernelSet` — the `Rmhd` instance of [`MhdSubstrateKernelSet`].
pub type RmhdSubstrateKernelSet<Mem, Sc, const D: usize> = MhdSubstrateKernelSet<Rmhd, Mem, Sc, D>;

/// back-compat alias for the 3D KernelSet (the original `*3D` name).
pub type RmhdSubstrateKernelSet3D<Mem, Sc = f64> = RmhdSubstrateKernelSet<Mem, Sc, 3>;
