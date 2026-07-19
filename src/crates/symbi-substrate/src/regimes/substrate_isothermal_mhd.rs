// =============================================================================
// regimes/substrate_isothermal_mhd.rs
//
// the isothermal-MHD KernelSet is now the `R = IsothermalMhd` instance of the unified
// `MhdSubstrateKernelSet`. thin back-compat alias; the
// implementation (trivial c2p, HLLE/HLLD flux with `p = cs^2 rho`, the shared CT stack)
// lives once in `substrate_mhd` + `mhd_substrate`, driven by `IsothermalMhd::SPEC`
// (`has_energy = false` -> the pre/nrg field rows are dropped; `cs` is the EOS scalar).
//
// usage:
//  let sub = IsothermalMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(cs, cfl, theta, &alloc);
// =============================================================================

use symbi_hydro::isothermal_mhd::IsothermalMhd;

use crate::regimes::substrate_mhd::MhdSubstrateKernelSet;

/// a D-dimensional isothermal-MHD `KernelSet` — the `IsothermalMhd` instance of [`MhdSubstrateKernelSet`].
pub type IsothermalMhdSubstrateKernelSet<Mem, Sc, const D: usize> =
    MhdSubstrateKernelSet<IsothermalMhd, Mem, Sc, D>;

/// back-compat alias for the 3D KernelSet (the original `*3D` name).
pub type IsothermalMhdSubstrateKernelSet3D<Mem, Sc = f64> = IsothermalMhdSubstrateKernelSet<Mem, Sc, 3>;
