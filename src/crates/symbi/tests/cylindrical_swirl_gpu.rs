// =============================================================================
// cylindrical_swirl_gpu.rs
//
// the GPU<->CPU validation for the DOF != NDIM kernels: the cylindrical r-z
// axisymmetric adiabatic family (ncomp = 3 on a 2-axis grid, swirl v_phi). builds
// two identical sims — host (CpuSpace/HostMemory, AOT CPU fn) and unified
// (CudaSpace/UnifiedMemory, render IR -> NVRTC -> launch) — drives the same
// DOF-generic AdiabaticSubstrateKernelSet on each, and diffs every _cyl kernel
// (snapshot / c2p / ghost_fill / cfl / flux per dir / godunov_euler / rk2) GPU == CPU
// to rel < 1e-9 (modulo nvcc FMA fusion). proves the metadata-driven dispatch +
// the _cyl ncomp=3 kernels run correctly on device.
//
// run on the host (CUDA 13.2 + g++-15): cargo test -p symbi --features cuda \
//     --test cylindrical_swirl_gpu
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
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};

const NR: usize = 16;
const NZ: usize = 8;
const CFL: f64 = 0.4;
const GAMMA: f64 = 1.4;
const V0: f64 = 1.0; // swirl speed.

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
        assert!(
            rel < 1e-9,
            "{what} at {c:?}: gpu {g} != cpu {h} (rel {rel:e})"
        );
    }
}

// a swirling annulus (r in [1,2], z in [0,1]) with a radial gaussian on rho/p and a
// 3-component velocity — so every cyl kernel (the phi-momentum advection, the
// centrifugal source, the per-axis-role flux) is exercised non-trivially.
fn build_swirl<S: ExecutionSpace, Mem: MemorySpace>()
-> SimStateGeneric<Newtonian, 2, 3, Cylindrical, IdealGas<f64>, S, Mem> {
    let (r_lo, r_hi) = (1.0_f64, 2.0_f64);
    let dr = (r_hi - r_lo) / NR as f64;
    let dz = 1.0 / NZ as f64;
    SimStateGeneric::<Newtonian, 2, 3, Cylindrical, IdealGas<f64>, S, Mem>::build(
        Newtonian,
        IdealGas { gamma: GAMMA },
        Cylindrical,
    )
    .cells([NR, NZ])
    .origin([r_lo, 0.0])
    .spacing([dr, dz])
    .cfl(CFL)
    .boundaries(Boundaries::per_axis([
        [BoundaryType::Outflow, BoundaryType::Outflow],
        [BoundaryType::Periodic, BoundaryType::Periodic],
    ]))
    .allocate()
    .expect("cyl swirl sim construction failed")
    .set_initial(|[r, _z]| {
        let g = (-((r - 1.5) / 0.15).powi(2)).exp();
        let rho = 1.0 + 0.3 * g;
        let pre = 1.0 + 0.5 * g;
        let (vr, vphi, vz) = (0.02, V0 * (1.0 + 0.1 * g), 0.01);
        Prim::adiabatic(Density(rho), Tensor::new([vr, vphi, vz]), Pressure(pre))
    })
    .build()
}

#[test]
fn cyl_swirl_gpu_matches_cpu() {
    let host = build_swirl::<CpuSpace, HostMemory>();
    let dev = build_swirl::<CudaSpace, UnifiedMemory>();
    let hset =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, CFL, &host.geom.allocated);
    let dset =
        AdiabaticSubstrateKernelSet::<UnifiedMemory, f64, 2>::new(GAMMA, CFL, &dev.geom.allocated);
    let alloc = &host.geom.allocated;
    let interior = &host.geom.interior;
    let (hnrg, dnrg) = (
        host.fields.cons.nrg_field().unwrap(),
        dev.fields.cons.nrg_field().unwrap(),
    );
    let (hpre, dpre) = (
        host.fields.prim.pre_field().unwrap(),
        dev.fields.prim.pre_field().unwrap(),
    );

    // snapshot (adiabatic_snapshot_cyl_2d): u_n = cons, all 3 components.
    hset.snapshot(&host);
    dset.snapshot(&dev);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
    cmp(
        alloc,
        &host.workspace.u_n.den,
        &dev.workspace.u_n.den,
        "u_n.den",
    );
    for k in 0..3 {
        cmp(
            alloc,
            &host.workspace.u_n.mom[k],
            &dev.workspace.u_n.mom[k],
            "u_n.mom",
        );
    }

    // c2p (adiabatic_c2p_cyl_2d): recovers rho + 3 velocities + pre.
    hset.c2p(&host);
    dset.c2p(&dev);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
    cmp(
        interior,
        &host.fields.prim.rho,
        &dev.fields.prim.rho,
        "prim.rho",
    );
    for k in 0..3 {
        cmp(
            interior,
            &host.fields.prim.vel[k],
            &dev.fields.prim.vel[k],
            "prim.vel",
        );
    }
    cmp(interior, hpre, dpre, "prim.pre");

    // ghost_fill (iso_ghost_fill_cyl_2d): the ncomp/axis-role pullback, all 3 vels.
    hset.ghost_fill(&host);
    dset.ghost_fill(&dev);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
    cmp(
        alloc,
        &host.fields.prim.rho,
        &dev.fields.prim.rho,
        "prim.rho (ghosts)",
    );
    for k in 0..3 {
        cmp(
            alloc,
            &host.fields.prim.vel[k],
            &dev.fields.prim.vel[k],
            "prim.vel (ghosts)",
        );
    }

    // cfl (iso_wave_speed_map_cyl_2d): the per-cell wave speed + device reduce.
    let (hdt, ddt) = (hset.cfl(&host), dset.cfl(&dev));
    assert!(
        hdt > 0.0 && hdt.is_finite() && ddt.is_finite(),
        "bad dt: cpu {hdt} gpu {ddt}"
    );
    assert!(
        (hdt - ddt).abs() / hdt < 1e-9,
        "cfl dt: cpu {hdt} != gpu {ddt}"
    );

    // flux (adiabatic_face_flux_cyl_2d_{dir}): per sweep dir, den + 3 mom + nrg.
    for dir in 0..2 {
        hset.flux(&host, dir);
        dset.flux(&dev, dir);
        symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
        let f = interior.face_domain(dir);
        cmp(
            &f,
            &host.fields.flux[dir].den,
            &dev.fields.flux[dir].den,
            "flux.den",
        );
        for k in 0..3 {
            cmp(
                &f,
                &host.fields.flux[dir].mom[k],
                &dev.fields.flux[dir].mom[k],
                "flux.mom",
            );
        }
        let (hfn, dfn) = (
            host.fields.flux[dir].nrg_field().unwrap(),
            dev.fields.flux[dir].nrg_field().unwrap(),
        );
        cmp(&f, hfn, dfn, "flux.nrg");
    }

    // godunov (adiabatic_godunov_{euler,rk2}_cyl_2d): the in-place update with the
    // area-weighted divergence + centrifugal source.
    hset.godunov_stage(&host, 0.005, 0.0, 1.0);
    dset.godunov_stage(&dev, 0.005, 0.0, 1.0);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
    cmp(
        interior,
        &host.fields.cons.den,
        &dev.fields.cons.den,
        "cons.den (euler)",
    );
    for k in 0..3 {
        cmp(
            interior,
            &host.fields.cons.mom[k],
            &dev.fields.cons.mom[k],
            "cons.mom (euler)",
        );
    }
    cmp(interior, hnrg, dnrg, "cons.nrg (euler)");

    hset.godunov_stage(&host, 0.005, 0.5, 0.5);
    dset.godunov_stage(&dev, 0.005, 0.5, 0.5);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
    for k in 0..3 {
        cmp(
            interior,
            &host.fields.cons.mom[k],
            &dev.fields.cons.mom[k],
            "cons.mom (rk2)",
        );
    }
    cmp(interior, hnrg, dnrg, "cons.nrg (rk2)");
}
