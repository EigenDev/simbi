// =============================================================================
// substrate_hydro_gpu.rs
//
// the D-generic hydro analog of substrate_rmhd_gpu.rs: every iso / Newton / RHD
// SubstrateKernelSet method runs on the GPU through the production dispatch path, at
// 1D, 2D AND 3D. build TWO identical sims — host (CpuSpace/HostMemory) and unified
// (CudaSpace/UnifiedMemory) — and drive the SAME `<Mem,f64,const D>` kernelset on
// each. unified `Mem` routes every kernel to run_gpu (render neutral IR -> NVRTC ->
// launch); host routes to the AOT CPU fn. the cfl exercises the device block-reduce
// (field_reduce_device) — the only device->host crossing.
//
// a deterministic SINGLE-PASS pipeline (snapshot, c2p, ghost_fill, cfl, flux per dir,
// godunov_euler, godunov_rk2) — every kernel run + diffed on the SAME input state, so
// there is no cfl-clamped step-count sensitivity. GPU vs CPU agree modulo nvcc FMA
// fusion (project_fma_discipline): ULP-bounded, rel < 1e-9.
//
// runs on a CUDA GPU (NVRTC needs no nvcc). in the symbi-cuda distrobox:
//   distrobox enter symbi-cuda -- cargo test -p symbi --features cuda \
//       --test substrate_hydro_gpu
// =============================================================================

#![cfg(feature = "cuda")]

use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::kernels::support::FaceDomain;
use symbi::sim::evolve::KernelSet;
use symbi::sim::state::*;
use symbi_algebra::{Domain, Tensor};
use symbi_geometry::{Cartesian, Metric};
use symbi_grid::Field;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::rhd::Rhd;
use symbi_hydro::state::{Prim, PrimG};
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};

const N: usize = 8;
const CFL: f64 = 0.4;
const GAMMA: f64 = 1.4; // Newton
const GAMMA_RHD: f64 = 5.0 / 3.0;
const CS: f64 = 1.0; // isothermal sound speed

// GPU vs CPU agree modulo nvcc FMA fusion: ULP-bounded drift, per single kernel.
// sync once before reading device fields back to host (the per-launch
// `ctx_sync` removal means stream ordering alone doesn't make device writes
// host-visible).
fn cmp<const D: usize, MH: MemorySpace, MD: MemorySpace>(
    dom: &Domain<D>,
    host: &Field<f64, D, MH>,
    dev: &Field<f64, D, MD>,
    what: &str,
) {
    symbi_xpu::cuda::ctx_sync();
    for c in dom.iter() {
        let (h, g) = (*host.view().at(c), *dev.view().at(c));
        assert!(g.is_finite(), "{what} at {c:?} went non-finite on GPU: {g}");
        let rel = (g - h).abs() / h.abs().max(1.0);
        assert!(rel < 1e-9, "{what} at {c:?}: gpu {g} != cpu {h} (rel {rel:e})");
    }
}

fn dt_close(host: f64, dev: f64) {
    assert!(host > 0.0 && host.is_finite() && dev.is_finite(), "bad dt: cpu {host} gpu {dev}");
    let rel = (host - dev).abs() / host;
    assert!(rel < 1e-9, "cfl dt: cpu {host} != gpu {dev} (rel {rel:e})");
}

// the d-axis sweep face domain (interior extended +1 on the hi side of `dir`).
fn face<const D: usize>(sim_interior: &Domain<D>, dir: usize) -> Domain<D> {
    sim_interior.face_domain(dir)
}

// ---- Newton (adiabatic, ideal-gas Euler) -----------------------------------

fn build_adiabatic<S: ExecutionSpace, Mem: MemorySpace, const D: usize>(
) -> SimState<Newtonian, D, Cartesian, IdealGas<f64>, S, Mem>
where
    Cartesian: Metric<f64, D>,
{
    let dx = 1.0 / N as f64;
    SimState::<Newtonian, D, Cartesian, IdealGas<f64>, S, Mem>::build(
        Newtonian,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([N; D])
    .spacing([dx; D])
    .cfl(CFL)
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .allocate()
    .expect("adiabatic sim construction failed")
    // conserved den=1+0.3*exp(-r2/0.05), mom[k]=0.02*(k+1), nrg=pre/(gamma-1) (no kinetic term)
    // inverted to prim: vel[k]=mom[k]/rho, pre'=(gamma-1)*nrg - (gamma-1)*0.5*rho*|v|^2 so that
    // prim->cons restores the same conserved nrg.
    .set_initial(|x| {
        let r2: f64 = (0..D).map(|k| (x[k] - 0.5).powi(2)).sum();
        let rho = 1.0 + 0.3 * (-r2 / 0.05).exp();
        let nrg = (1.0 + 2.0 * (-r2 / 0.02).exp()) / (GAMMA - 1.0);
        let vel: [f64; D] = std::array::from_fn(|k| 0.02 * (k as f64 + 1.0) / rho);
        let vsq: f64 = (0..D).map(|k| vel[k] * vel[k]).sum();
        let pre = (GAMMA - 1.0) * (nrg - 0.5 * rho * vsq);
        Prim { rho, vel: Tensor::new(vel), pre }
    })
    .build()
}

fn check_adiabatic<const D: usize>()
where
    Cartesian: Metric<f64, D>,
{
    let host = build_adiabatic::<CpuSpace, HostMemory, D>();
    let dev = build_adiabatic::<CudaSpace, UnifiedMemory, D>();
    let hset = AdiabaticSubstrateKernelSet::<HostMemory, f64, D>::new(GAMMA, CFL, &host.geom.allocated);
    let dset = AdiabaticSubstrateKernelSet::<UnifiedMemory, f64, D>::new(GAMMA, CFL, &dev.geom.allocated);
    let alloc = &host.geom.allocated;
    let interior = &host.geom.interior;
    let (hnrg, dnrg) = (host.fields.cons.nrg_field().unwrap(), dev.fields.cons.nrg_field().unwrap());
    let (hpre, dpre) = (host.fields.prim.pre_field().unwrap(), dev.fields.prim.pre_field().unwrap());

    hset.snapshot(&host);
    dset.snapshot(&dev);
    cmp(alloc, &host.workspace.u_n.den, &dev.workspace.u_n.den, "u_n.den");

    hset.c2p(&host);
    dset.c2p(&dev);
    cmp(interior, &host.fields.prim.rho, &dev.fields.prim.rho, "prim.rho");
    for k in 0..D {
        cmp(interior, &host.fields.prim.vel[k], &dev.fields.prim.vel[k], "prim.vel");
    }
    cmp(interior, hpre, dpre, "prim.pre");

    hset.ghost_fill(&host);
    dset.ghost_fill(&dev);
    cmp(alloc, &host.fields.prim.rho, &dev.fields.prim.rho, "prim.rho (ghosts)");

    dt_close(hset.cfl(&host), dset.cfl(&dev));

    for dir in 0..D {
        hset.flux(&host, dir);
        dset.flux(&dev, dir);
        let f = face(interior, dir);
        let (hfn, dfn) = (host.fields.flux[dir].nrg_field().unwrap(), dev.fields.flux[dir].nrg_field().unwrap());
        cmp(&f, &host.fields.flux[dir].den, &dev.fields.flux[dir].den, "flux.den");
        for k in 0..D {
            cmp(&f, &host.fields.flux[dir].mom[k], &dev.fields.flux[dir].mom[k], "flux.mom");
        }
        cmp(&f, hfn, dfn, "flux.nrg");
    }

    hset.godunov_stage(&host, 0.01, 0.0, 1.0);
    dset.godunov_stage(&dev, 0.01, 0.0, 1.0);
    cmp(interior, &host.fields.cons.den, &dev.fields.cons.den, "cons.den (euler)");
    hset.godunov_stage(&host, 0.01, 0.5, 0.5);
    dset.godunov_stage(&dev, 0.01, 0.5, 0.5);
    cmp(interior, hnrg, dnrg, "cons.nrg (rk2)");
}

#[test] fn adiabatic_gpu_1d() { check_adiabatic::<1>(); }
#[test] fn adiabatic_gpu_2d() { check_adiabatic::<2>(); }
#[test] fn adiabatic_gpu_3d() { check_adiabatic::<3>(); }

// ---- isothermal Euler (substrate-owned pressure, no energy) -----------------

fn build_iso<S: ExecutionSpace, Mem: MemorySpace, const D: usize>(
) -> SimState<IsoNewtonian, D, Cartesian, Isothermal<f64>, S, Mem>
where
    Cartesian: Metric<f64, D>,
{
    let dx = 1.0 / N as f64;
    SimState::<IsoNewtonian, D, Cartesian, Isothermal<f64>, S, Mem>::build(
        IsoNewtonian,
        Isothermal { cs: CS },
        Cartesian,
    )
    .cells([N; D])
    .spacing([dx; D])
    .cfl(CFL)
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .allocate()
    .expect("iso sim construction failed")
    // conserved den=1+0.4*exp(-r2/0.05), mom[k]=0.02*(k+1) inverted to iso prim: vel[k]=mom[k]/rho.
    .set_initial(|x| {
        let r2: f64 = (0..D).map(|k| (x[k] - 0.5).powi(2)).sum();
        let rho = 1.0 + 0.4 * (-r2 / 0.05).exp();
        let vel: [f64; D] = std::array::from_fn(|k| 0.02 * (k as f64 + 1.0) / rho);
        PrimG::<f64, D, IsoModel> { rho, vel: Tensor::new(vel), pre: Default::default() }
    })
    .build()
}

fn check_iso<const D: usize>()
where
    Cartesian: Metric<f64, D>,
{
    let host = build_iso::<CpuSpace, HostMemory, D>();
    let dev = build_iso::<CudaSpace, UnifiedMemory, D>();
    let hset = IsoSubstrateKernelSet::<HostMemory, f64, D>::new(CS, CFL, &host.geom.allocated);
    let dset = IsoSubstrateKernelSet::<UnifiedMemory, f64, D>::new(CS, CFL, &dev.geom.allocated);
    let alloc = &host.geom.allocated;
    let interior = &host.geom.interior;

    hset.snapshot(&host);
    dset.snapshot(&dev);
    cmp(alloc, &host.workspace.u_n.den, &dev.workspace.u_n.den, "u_n.den");

    hset.c2p(&host);
    dset.c2p(&dev);
    cmp(interior, &host.fields.prim.rho, &dev.fields.prim.rho, "prim.rho");
    for k in 0..D {
        cmp(interior, &host.fields.prim.vel[k], &dev.fields.prim.vel[k], "prim.vel");
    }
    // iso pressure is the substrate-owned field on the kernelset.
    cmp(interior, &hset.pre, &dset.pre, "iso pre");

    hset.ghost_fill(&host);
    dset.ghost_fill(&dev);
    cmp(alloc, &host.fields.prim.rho, &dev.fields.prim.rho, "prim.rho (ghosts)");
    cmp(alloc, &hset.pre, &dset.pre, "iso pre (ghosts)");

    dt_close(hset.cfl(&host), dset.cfl(&dev));

    for dir in 0..D {
        hset.flux(&host, dir);
        dset.flux(&dev, dir);
        let f = face(interior, dir);
        cmp(&f, &host.fields.flux[dir].den, &dev.fields.flux[dir].den, "flux.den");
        for k in 0..D {
            cmp(&f, &host.fields.flux[dir].mom[k], &dev.fields.flux[dir].mom[k], "flux.mom");
        }
    }

    hset.godunov_stage(&host, 0.01, 0.0, 1.0);
    dset.godunov_stage(&dev, 0.01, 0.0, 1.0);
    cmp(interior, &host.fields.cons.den, &dev.fields.cons.den, "cons.den (euler)");
    for k in 0..D {
        cmp(interior, &host.fields.cons.mom[k], &dev.fields.cons.mom[k], "cons.mom (euler)");
    }
}

#[test] fn iso_gpu_1d() { check_iso::<1>(); }
#[test] fn iso_gpu_2d() { check_iso::<2>(); }
#[test] fn iso_gpu_3d() { check_iso::<3>(); }

// ---- RHD (special-relativistic Euler, iterative c2p + per-axis wave speeds) -

fn build_rhd<S: ExecutionSpace, Mem: MemorySpace, const D: usize>(
) -> SimState<Rhd, D, Cartesian, IdealGas<f64>, S, Mem>
where
    Cartesian: Metric<f64, D>,
{
    let dx = 1.0 / N as f64;
    SimState::<Rhd, D, Cartesian, IdealGas<f64>, S, Mem>::build(
        Rhd,
        IdealGas { gamma: GAMMA_RHD },
        Cartesian,
    )
    .cells([N; D])
    .spacing([dx; D])
    .cfl(CFL)
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .allocate()
    .expect("RHD sim construction failed")
    // mildly relativistic: v = 0 => W = 1, D = rho, S = 0, tau = rho*h - p - rho. the at-rest prim
    // (rho, v=0, pre) traces to exactly these conserved values via the RHD prim->cons.
    .set_initial(|x| {
        let r2: f64 = (0..D).map(|k| (x[k] - 0.5).powi(2)).sum();
        let rho = 1.0 + 0.3 * (-r2 / 0.05).exp();
        let pre = 1.0 + 2.0 * (-r2 / 0.02).exp();
        Prim { rho, vel: Tensor::new([0.0; D]), pre }
    })
    .build()
}

fn check_rhd<const D: usize>()
where
    Cartesian: Metric<f64, D>,
{
    let host = build_rhd::<CpuSpace, HostMemory, D>();
    let dev = build_rhd::<CudaSpace, UnifiedMemory, D>();
    let hset = RhdSubstrateKernelSet::<HostMemory, f64, D>::new(GAMMA_RHD, CFL, &host.geom.allocated);
    let dset = RhdSubstrateKernelSet::<UnifiedMemory, f64, D>::new(GAMMA_RHD, CFL, &dev.geom.allocated);
    let alloc = &host.geom.allocated;
    let interior = &host.geom.interior;
    let (hnrg, dnrg) = (host.fields.cons.nrg_field().unwrap(), dev.fields.cons.nrg_field().unwrap());
    let (hpre, dpre) = (host.fields.prim.pre_field().unwrap(), dev.fields.prim.pre_field().unwrap());

    hset.snapshot(&host);
    dset.snapshot(&dev);
    cmp(alloc, &host.workspace.u_n.den, &dev.workspace.u_n.den, "u_n.den");

    // the iterative masked-Newton c2p on-device.
    hset.c2p(&host);
    dset.c2p(&dev);
    cmp(interior, &host.fields.prim.rho, &dev.fields.prim.rho, "prim.rho");
    for k in 0..D {
        cmp(interior, &host.fields.prim.vel[k], &dev.fields.prim.vel[k], "prim.vel");
    }
    cmp(interior, hpre, dpre, "prim.pre");

    hset.ghost_fill(&host);
    dset.ghost_fill(&dev);
    cmp(alloc, &host.fields.prim.rho, &dev.fields.prim.rho, "prim.rho (ghosts)");

    // the per-axis relativistic wave-speed map + device reduce.
    dt_close(hset.cfl(&host), dset.cfl(&dev));

    for dir in 0..D {
        hset.flux(&host, dir);
        dset.flux(&dev, dir);
        let f = face(interior, dir);
        let (hfn, dfn) = (host.fields.flux[dir].nrg_field().unwrap(), dev.fields.flux[dir].nrg_field().unwrap());
        cmp(&f, &host.fields.flux[dir].den, &dev.fields.flux[dir].den, "flux.den");
        for k in 0..D {
            cmp(&f, &host.fields.flux[dir].mom[k], &dev.fields.flux[dir].mom[k], "flux.mom");
        }
        cmp(&f, hfn, dfn, "flux.nrg");
    }

    hset.godunov_stage(&host, 0.01, 0.0, 1.0);
    dset.godunov_stage(&dev, 0.01, 0.0, 1.0);
    cmp(interior, &host.fields.cons.den, &dev.fields.cons.den, "cons.den (euler)");
    cmp(interior, hnrg, dnrg, "cons.nrg (euler)");
}

#[test] fn rhd_gpu_1d() { check_rhd::<1>(); }
#[test] fn rhd_gpu_2d() { check_rhd::<2>(); }
#[test] fn rhd_gpu_3d() { check_rhd::<3>(); }
