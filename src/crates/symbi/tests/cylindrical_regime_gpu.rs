// =============================================================================
// cylindrical_regime_gpu.rs
//
// GPU<->CPU validation for the cylindrical-natural hydro cells: the
// cyl-1D (radial) and cyl-3D (r,phi,z) "_cyl" godunov + wave-speed maps for all three
// EOS regimes (iso / newton-adiabatic / rhd). builds host (AOT CPU) + unified (NVRTC)
// sims, runs godunov_euler + cfl on each, diffs < 1e-9. (cyl-2D r-phi is already
// covered by cylindrical_disk_gpu for newton; iso/rhd 2D share the same IR builders.)
// proves the matrix fill compiles to PTX + runs on device.
//
// run on the host (CUDA 13.2 + g++-15): cargo test -p symbi --features cuda \
//     --test cylindrical_regime_gpu
// =============================================================================

#![cfg(feature = "cuda")]

use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::evolve::KernelSet;
use symbi::sim::state::*;
use symbi_algebra::Domain;
use symbi_geometry::{Cylindrical, Metric};
use symbi_grid::Field;
use symbi_hydro::eos::{Eos, IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::regime::Regime;
use symbi_hydro::rhd::Rhd;
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};

const GAMMA: f64 = 1.4;
const CS: f64 = 1.0;
const TWO_PI: f64 = std::f64::consts::TAU;

fn cmp<const D: usize, MH: MemorySpace, MD: MemorySpace>(
    dom: &Domain<D>,
    host: &Field<f64, D, MH>,
    dev: &Field<f64, D, MD>,
    what: &str,
) {
    for c in dom.iter() {
        let (h, g) = (*host.view().at(c), *dev.view().at(c));
        assert!(g.is_finite(), "{what} at {c:?} non-finite on GPU: {g}");
        let rel = (g - h).abs() / h.abs().max(1.0);
        assert!(
            rel < 1e-9,
            "{what} at {c:?}: gpu {g} != cpu {h} (rel {rel:e})"
        );
    }
}

// diff godunov_euler (all cons fields) + cfl dt between an already-IC'd host + dev sim.
// hks / dks are distinct types (the kernelset is generic over the memory space).
fn diff_godunov_cfl<const D: usize, R, E, HKS, DKS>(
    host: &SimStateGeneric<R, D, D, Cylindrical, E, CpuSpace, HostMemory>,
    dev: &SimStateGeneric<R, D, D, Cylindrical, E, CudaSpace, UnifiedMemory>,
    hset: &HKS,
    dset: &DKS,
    has_energy: bool,
    tag: &str,
) where
    R: Regime<f64, D>,
    E: Eos<f64>,
    Cylindrical: Metric<f64, D>,
    HKS: KernelSet<D, D, HostMemory, f64>,
    DKS: KernelSet<D, D, UnifiedMemory, f64>,
{
    let interior = &host.geom.interior;
    // c2p populates prim (rho/vel/pre) from the cons IC — the wave-speed map + godunov
    // reconstruction read prim, so this must run first.
    hset.c2p(host);
    dset.c2p(dev);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
    let (hdt, ddt) = (hset.cfl(host), dset.cfl(dev));
    assert!(
        hdt > 0.0 && hdt.is_finite() && ddt.is_finite(),
        "{tag} bad dt: cpu {hdt} gpu {ddt}"
    );
    assert!(
        (hdt - ddt).abs() / hdt < 1e-9,
        "{tag} cfl dt: cpu {hdt} != gpu {ddt}"
    );

    hset.godunov_stage(host, 0.002, 0.0, 1.0);
    dset.godunov_stage(dev, 0.002, 0.0, 1.0);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
    cmp(
        interior,
        &host.fields.cons.den,
        &dev.fields.cons.den,
        &format!("{tag} cons.den"),
    );
    for k in 0..D {
        cmp(
            interior,
            &host.fields.cons.mom[k],
            &dev.fields.cons.mom[k],
            &format!("{tag} cons.mom"),
        );
    }
    if has_energy {
        let (h, g) = (
            host.fields.cons.nrg_field().unwrap(),
            dev.fields.cons.nrg_field().unwrap(),
        );
        cmp(interior, h, g, &format!("{tag} cons.nrg"));
    }
}

// ----- cyl 1D (radial) ------------------------------------------------------------

fn build_cyl_1d<R: Regime<f64, 1>, E: Eos<f64>, S: ExecutionSpace, Mem: MemorySpace>(
    regime: R,
    eos: E,
) -> SimStateGeneric<R, 1, 1, Cylindrical, E, S, Mem> {
    let (nr, r_lo, dr) = (32usize, 1.0_f64, 1.0 / 32.0);
    let sim = SimStateGeneric::<R, 1, 1, Cylindrical, E, S, Mem>::build(regime, eos, Cylindrical)
        .cells([nr])
        .origin([r_lo])
        .spacing([dr])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.3)
        .finish()
        .expect("cyl 1D ctor");
    let cnrg = sim.fields.cons.nrg_field();
    for c in sim.geom.interior.iter() {
        let r = r_lo + (c[0] as f64 + 0.5) * dr;
        let g = (-((r - 1.5) / 0.2).powi(2)).exp();
        let rho = 1.0 + 0.3 * g;
        sim.fields.cons.den.view_mut().set(c, rho);
        sim.fields.cons.mom[0].view_mut().set(c, rho * 0.1 * g);
        if let Some(nrg) = cnrg {
            // adiabatic: E = p/(g-1)+0.5 rho v^2; rhd at near-rest: tau ~ p/(g-1).
            nrg.view_mut().set(
                c,
                (1.0 + 0.5 * g) / (GAMMA - 1.0) + 0.5 * rho * (0.1 * g).powi(2),
            );
        }
    }
    sim
}

#[test]
fn cyl_1d_all_regimes_gpu_match_cpu() {
    // newton
    let h = build_cyl_1d::<_, _, CpuSpace, HostMemory>(Newtonian, IdealGas { gamma: GAMMA });
    let d = build_cyl_1d::<_, _, CudaSpace, UnifiedMemory>(Newtonian, IdealGas { gamma: GAMMA });
    let hs = AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.3, &h.geom.allocated);
    let ds =
        AdiabaticSubstrateKernelSet::<UnifiedMemory, f64, 1>::new(GAMMA, 0.3, &d.geom.allocated);
    diff_godunov_cfl(&h, &d, &hs, &ds, true, "newton cyl1d");
    // iso
    let h = build_cyl_1d::<_, _, CpuSpace, HostMemory>(IsoNewtonian, Isothermal { cs: CS });
    let d = build_cyl_1d::<_, _, CudaSpace, UnifiedMemory>(IsoNewtonian, Isothermal { cs: CS });
    let hs = IsoSubstrateKernelSet::<HostMemory, f64, 1>::new(CS, 0.3, &h.geom.allocated);
    let ds = IsoSubstrateKernelSet::<UnifiedMemory, f64, 1>::new(CS, 0.3, &d.geom.allocated);
    diff_godunov_cfl(&h, &d, &hs, &ds, false, "iso cyl1d");
    // rhd
    let h = build_cyl_1d::<_, _, CpuSpace, HostMemory>(Rhd, IdealGas { gamma: GAMMA });
    let d = build_cyl_1d::<_, _, CudaSpace, UnifiedMemory>(Rhd, IdealGas { gamma: GAMMA });
    let hs = RhdSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.3, &h.geom.allocated);
    let ds = RhdSubstrateKernelSet::<UnifiedMemory, f64, 1>::new(GAMMA, 0.3, &d.geom.allocated);
    diff_godunov_cfl(&h, &d, &hs, &ds, true, "rhd cyl1d");
}

// ----- cyl 3D (r,phi,z) -----------------------------------------------------------

fn build_cyl_3d<R: Regime<f64, 3>, E: Eos<f64>, S: ExecutionSpace, Mem: MemorySpace>(
    regime: R,
    eos: E,
) -> SimStateGeneric<R, 3, 3, Cylindrical, E, S, Mem> {
    let (nr, nphi, nz, r_lo, dr) = (12usize, 12usize, 4usize, 1.0_f64, 1.0 / 12.0);
    let (dphi, dz) = (TWO_PI / nphi as f64, 0.5 / nz as f64);
    let sim = SimStateGeneric::<R, 3, 3, Cylindrical, E, S, Mem>::build(regime, eos, Cylindrical)
        .cells([nr, nphi, nz])
        .origin([r_lo, 0.0, 0.0])
        .spacing([dr, dphi, dz])
        .boundaries(Boundaries::per_axis([
            [BoundaryType::Outflow, BoundaryType::Outflow],
            [BoundaryType::Periodic, BoundaryType::Periodic],
            [BoundaryType::Periodic, BoundaryType::Periodic],
        ]))
        .cfl(0.3)
        .finish()
        .expect("cyl 3D ctor");
    let cnrg = sim.fields.cons.nrg_field();
    for c in sim.geom.interior.iter() {
        let r = r_lo + (c[0] as f64 + 0.5) * dr;
        let g = (-((r - 1.5) / 0.3).powi(2)).exp();
        let rho = 1.0 + 0.3 * g;
        let vphi = 0.3 * g;
        sim.fields.cons.den.view_mut().set(c, rho);
        sim.fields.cons.mom[0].view_mut().set(c, 0.0);
        sim.fields.cons.mom[1].view_mut().set(c, rho * vphi);
        sim.fields.cons.mom[2].view_mut().set(c, 0.0);
        if let Some(nrg) = cnrg {
            nrg.view_mut()
                .set(c, (1.0 + 0.5 * g) / (GAMMA - 1.0) + 0.5 * rho * vphi * vphi);
        }
    }
    sim
}

#[test]
fn cyl_3d_all_regimes_gpu_match_cpu() {
    // newton
    let h = build_cyl_3d::<_, _, CpuSpace, HostMemory>(Newtonian, IdealGas { gamma: GAMMA });
    let d = build_cyl_3d::<_, _, CudaSpace, UnifiedMemory>(Newtonian, IdealGas { gamma: GAMMA });
    let hs = AdiabaticSubstrateKernelSet::<HostMemory, f64, 3>::new(GAMMA, 0.3, &h.geom.allocated);
    let ds =
        AdiabaticSubstrateKernelSet::<UnifiedMemory, f64, 3>::new(GAMMA, 0.3, &d.geom.allocated);
    diff_godunov_cfl(&h, &d, &hs, &ds, true, "newton cyl3d");
    // iso
    let h = build_cyl_3d::<_, _, CpuSpace, HostMemory>(IsoNewtonian, Isothermal { cs: CS });
    let d = build_cyl_3d::<_, _, CudaSpace, UnifiedMemory>(IsoNewtonian, Isothermal { cs: CS });
    let hs = IsoSubstrateKernelSet::<HostMemory, f64, 3>::new(CS, 0.3, &h.geom.allocated);
    let ds = IsoSubstrateKernelSet::<UnifiedMemory, f64, 3>::new(CS, 0.3, &d.geom.allocated);
    diff_godunov_cfl(&h, &d, &hs, &ds, false, "iso cyl3d");
    // rhd
    let h = build_cyl_3d::<_, _, CpuSpace, HostMemory>(Rhd, IdealGas { gamma: GAMMA });
    let d = build_cyl_3d::<_, _, CudaSpace, UnifiedMemory>(Rhd, IdealGas { gamma: GAMMA });
    let hs = RhdSubstrateKernelSet::<HostMemory, f64, 3>::new(GAMMA, 0.3, &h.geom.allocated);
    let ds = RhdSubstrateKernelSet::<UnifiedMemory, f64, 3>::new(GAMMA, 0.3, &d.geom.allocated);
    diff_godunov_cfl(&h, &d, &hs, &ds, true, "rhd cyl3d");
}
