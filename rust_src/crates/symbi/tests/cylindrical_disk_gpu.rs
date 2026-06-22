// =============================================================================
// cylindrical_disk_gpu.rs
//
// the GPU<->CPU validation for the cylindrical r-phi DISK family (DOF == NDIM == 2):
// the natural cylindrical plane, distinct from the r-z axisymmetric set
// (cylindrical_swirl_gpu.rs, DOF=3). builds two identical sims — host (AOT CPU) and
// unified (render IR -> NVRTC -> launch) — drives the DOF-generic
// AdiabaticSubstrateKernelSet on each, and diffs every kernel the disk dispatches
// GPU == CPU to rel < 1e-9. the geometry-dependent "_cyl" instances here are the
// godunov (area-weighted r-phi divergence + centrifugal/coriolis source + hoop p/r)
// and the CFL wave-speed map (physical r·dphi widths); the flux / c2p / snapshot /
// ghost reuse the cartesian ncomp=2 instances (DOF == NDIM), confirmed to dispatch
// correctly for an r-phi sim. proves the disk-evolve hydro runs on device.
//
// run on the host (CUDA 13.2 + g++-15): cargo test -p symbi --features cuda \
//     --test cylindrical_disk_gpu
// =============================================================================

#![cfg(feature = "cuda")]

use symbi::kernels::support::FaceDomain;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::KernelSet;
use symbi::sim::state::*;
use symbi_algebra::{Domain, Tensor};
use symbi_geometry::Cylindrical;
use symbi_grid::Field;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};

const NR: usize = 16;
const NPHI: usize = 24;
const CFL: f64 = 0.3;
const GAMMA: f64 = 1.4;
const TWO_PI: f64 = std::f64::consts::TAU;

fn cmp<MH: MemorySpace, MD: MemorySpace>(
    dom: &Domain<2>,
    host: &Field<f64, 2, MH>,
    dev: &Field<f64, 2, MD>,
    what: &str,
) {
    for c in dom.iter() {
        let (h, g) = (*host.view().at(c), *dev.view().at(c));
        assert!(g.is_finite(), "{what} at {c:?} went non-finite on GPU: {g}");
        let rel = (g - h).abs() / h.abs().max(1.0);
        assert!(rel < 1e-9, "{what} at {c:?}: gpu {g} != cpu {h} (rel {rel:e})");
    }
}

// a rotating disk (r in [0.5,1.5], phi in [0,2*pi)) with an azimuthal density/pressure
// gaussian + a sheared rotation profile and a small v_r — so the phi-flux, the
// area-weighted r-phi divergence, and the centrifugal/coriolis source are all exercised
// non-trivially (no exact equilibrium; we diff CPU vs GPU, not against a steady state).
fn build_disk<S: ExecutionSpace, Mem: MemorySpace>(
) -> SimStateGeneric<Newtonian, 2, 2, Cylindrical, IdealGas<f64>, S, Mem> {
    let (r_lo, r_hi) = (0.5_f64, 1.5_f64);
    let dr = (r_hi - r_lo) / NR as f64;
    let dphi = TWO_PI / NPHI as f64;
    SimStateGeneric::<Newtonian, 2, 2, Cylindrical, IdealGas<f64>, S, Mem>::build(
        Newtonian,
        IdealGas { gamma: GAMMA },
        Cylindrical,
    )
    .cells([NR, NPHI])
    .origin([r_lo, 0.0])
    .spacing([dr, dphi])
    .cfl(CFL)
    .boundaries(Boundaries::per_axis([
        [BoundaryType::Outflow, BoundaryType::Outflow],
        [BoundaryType::Periodic, BoundaryType::Periodic],
    ]))
    .allocate()
    .expect("cyl disk sim construction failed")
    .set_initial(|[r, phi]| {
        let g = (-((r - 1.0) / 0.2).powi(2)).exp() * (0.5 * phi).sin().powi(2);
        let rho = 1.0 + 0.4 * g;
        let pre = 1.0 + 0.3 * g;
        let vphi = (1.0_f64 / r).sqrt() * (1.0 + 0.05 * g); // sheared near-keplerian
        let vr = 0.03 * g;
        Prim { rho, vel: Tensor::new([vr, vphi]), pre }
    })
    .build()
}

#[test]
fn cyl_disk_gpu_matches_cpu() {
    let host = build_disk::<CpuSpace, HostMemory>();
    let dev = build_disk::<CudaSpace, UnifiedMemory>();
    let hset = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, CFL, &host.geom.allocated);
    let dset = AdiabaticSubstrateKernelSet::<UnifiedMemory, f64, 2>::new(GAMMA, CFL, &dev.geom.allocated);
    let alloc = &host.geom.allocated;
    let interior = &host.geom.interior;
    let (hnrg, dnrg) = (host.fields.cons.nrg_field().unwrap(), dev.fields.cons.nrg_field().unwrap());
    let (hpre, dpre) = (host.fields.prim.pre_field().unwrap(), dev.fields.prim.pre_field().unwrap());

    // snapshot / c2p / ghost (cartesian ncomp=2 instances, reused since DOF == NDIM).
    hset.snapshot(&host);
    dset.snapshot(&dev);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
    cmp(alloc, &host.workspace.u_n.den, &dev.workspace.u_n.den, "u_n.den");
    for k in 0..2 {
        cmp(alloc, &host.workspace.u_n.mom[k], &dev.workspace.u_n.mom[k], "u_n.mom");
    }

    hset.c2p(&host);
    dset.c2p(&dev);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
    cmp(interior, &host.fields.prim.rho, &dev.fields.prim.rho, "prim.rho");
    for k in 0..2 {
        cmp(interior, &host.fields.prim.vel[k], &dev.fields.prim.vel[k], "prim.vel");
    }
    cmp(interior, hpre, dpre, "prim.pre");

    hset.ghost_fill(&host);
    dset.ghost_fill(&dev);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
    cmp(alloc, &host.fields.prim.rho, &dev.fields.prim.rho, "prim.rho (ghosts)");
    for k in 0..2 {
        cmp(alloc, &host.fields.prim.vel[k], &dev.fields.prim.vel[k], "prim.vel (ghosts)");
    }

    // cfl (iso_wave_speed_map_cyl_2d, ncomp=2): physical r·dphi widths + device reduce.
    let (hdt, ddt) = (hset.cfl(&host), dset.cfl(&dev));
    assert!(hdt > 0.0 && hdt.is_finite() && ddt.is_finite(), "bad dt: cpu {hdt} gpu {ddt}");
    assert!((hdt - ddt).abs() / hdt < 1e-9, "cfl dt: cpu {hdt} != gpu {ddt}");

    // flux (adiabatic_face_flux_2d_{dir}, cartesian): per sweep dir, den + 2 mom + nrg.
    for dir in 0..2 {
        hset.flux(&host, dir);
        dset.flux(&dev, dir);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
        let f = interior.face_domain(dir);
        cmp(&f, &host.fields.flux[dir].den, &dev.fields.flux[dir].den, "flux.den");
        for k in 0..2 {
            cmp(&f, &host.fields.flux[dir].mom[k], &dev.fields.flux[dir].mom[k], "flux.mom");
        }
        let (hfn, dfn) = (host.fields.flux[dir].nrg_field().unwrap(), dev.fields.flux[dir].nrg_field().unwrap());
        cmp(&f, hfn, dfn, "flux.nrg");
    }

    // godunov (adiabatic_godunov_{euler,rk2}_cyl_2d, ncomp=2): the in-place r-phi update
    // with the area-weighted divergence + centrifugal/coriolis source + hoop p/r — the
    // NEW disk kernels.
    hset.godunov_stage(&host, 0.002, 0.0, 1.0);
    dset.godunov_stage(&dev, 0.002, 0.0, 1.0);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
    cmp(interior, &host.fields.cons.den, &dev.fields.cons.den, "cons.den (euler)");
    for k in 0..2 {
        cmp(interior, &host.fields.cons.mom[k], &dev.fields.cons.mom[k], "cons.mom (euler)");
    }
    cmp(interior, hnrg, dnrg, "cons.nrg (euler)");

    hset.godunov_stage(&host, 0.002, 0.5, 0.5);
    dset.godunov_stage(&dev, 0.002, 0.5, 0.5);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
    for k in 0..2 {
        cmp(interior, &host.fields.cons.mom[k], &dev.fields.cons.mom[k], "cons.mom (rk2)");
    }
    cmp(interior, hnrg, dnrg, "cons.nrg (rk2)");
}
