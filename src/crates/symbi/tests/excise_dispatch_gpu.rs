// =============================================================================
// !!! CONTRACT NOTE: the excision fill is a DIRICHLET VACUUM SINK (rho = 1e-10,
// v = 0, p = 1e-12; gv_excise.rs), NOT the onion-sweep outward copy this file's
// prose describes — a uniform state inside the sphere is rewritten to the floor,
// not preserved. the cpu-vs-gpu relative assertions below remain valid, but this
// file is cfg-stripped on non-cuda hosts and has NOT been compile-checked since
// the redesign: the first cuda session must re-verify it against the cpu twin
// (excise_dispatch.rs, which asserts the floor bitwise).
// excise_dispatch_gpu.rs
//
// the device twin of excise_dispatch: horizon excision on an origin-containing
// cartesian kerr-schild box, run on device memory (NVRTC render of the AOT
// excise_fill / excise_writeback / excise_p2c kernels) and asserted BIT-CLOSE to
// the CPU pass. the onion-sweep fill + valencia rebuild + the source-CFL mask
// behave identically on device: the far field stays untouched and the excised
// sphere is rewritten the same way it is on host. these emit paths had never
// executed on a device before this gate.
//
// runs on the host GPU (NVRTC needs no nvcc). run:
//   cargo test -p symbi --features cuda --test excise_dispatch_gpu
// =============================================================================
#![cfg(feature = "cuda")]

use symbi::regimes::substrate_kernels::dispatch_excise;
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::evolve::KernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::SchwarzschildKSCartesian;
use symbi_hydro::Rhd;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::state::Prim;
use symbi_sim::substrate_seam::WithExcision;
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};

const N: usize = 48;
const L: f64 = 1.2;
const GAMMA: f64 = 4.0 / 3.0;
const MASS: f64 = 0.3; // r_+ = 0.6, well inside the box
const R_EXC: f64 = 0.35; // inside r_+, above the metric guard M/2 = 0.15

type Sim<S, Mem> = SimState<Rhd, 2, SchwarzschildKSCartesian<f64>, IdealGas<f64>, S, Mem>;

fn build<S: ExecutionSpace, Mem: MemorySpace>(
    init: impl Fn([f64; 2]) -> Prim<f64, 2>,
) -> Sim<S, Mem> {
    let dx = 2.0 * L / N as f64;
    let sim = Sim::<S, Mem>::build(
        Rhd,
        IdealGas { gamma: GAMMA },
        SchwarzschildKSCartesian { mass: MASS },
    )
    .cells([N, N])
    .origin([-L, -L])
    .spacing([dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
    .allocate()
    .expect("sim")
    .set_initial(|x| init(x))
    .build();
    // the builder stores CONSERVED state; the excision pass reads current prims (materialized
    // by c2p in production). populate the prim fields directly to model the post-c2p state.
    for c in sim.geom.interior.iter() {
        let lo = sim.geom.interior.spaces[0].lo;
        let x = sim.geom.x_lo[0] + ((c[0] - lo) as f64 + 0.5) * dx;
        let y = sim.geom.x_lo[1] + ((c[1] - lo) as f64 + 0.5) * dx;
        let p = init([x, y]);
        sim.fields.prim.rho.view_mut().set(c, p.rho);
        sim.fields.prim.vel[0].view_mut().set(c, p.vel[0]);
        sim.fields.prim.vel[1].view_mut().set(c, p.vel[1]);
        sim.fields
            .prim
            .pre_field()
            .unwrap()
            .view_mut()
            .set(c, p.pre);
    }
    sim
}

// the 8-field (prim rho/v0/v1/pre + cons den/m0/m1/nrg) snapshot over the interior.
fn snapshot<S: ExecutionSpace, Mem: MemorySpace>(sim: &Sim<S, Mem>) -> Vec<[f64; 8]> {
    sim.geom
        .interior
        .iter()
        .map(|c| {
            [
                *sim.fields.prim.rho.view().at(c),
                *sim.fields.prim.vel[0].view().at(c),
                *sim.fields.prim.vel[1].view().at(c),
                *sim.fields.prim.pre_field().unwrap().view().at(c),
                *sim.fields.cons.den.view().at(c),
                *sim.fields.cons.mom[0].view().at(c),
                *sim.fields.cons.mom[1].view().at(c),
                *sim.fields.cons.nrg_field().unwrap().view().at(c),
            ]
        })
        .collect()
}

#[test]
fn excise_dispatch_matches_cpu_on_device() {
    let init = |[x, y]: [f64; 2]| Prim {
        rho: 1.0 + 0.2 * (2.0 * x).sin() * (1.5 * y).cos(),
        vel: Tensor::new([0.08 * (x + y).cos(), -0.06 * (x - y).sin()]),
        pre: 0.05 + 0.01 * (x * y).cos(),
    };
    let h = build::<CpuSpace, HostMemory>(init);
    let d = build::<CudaSpace, UnifiedMemory>(init);

    dispatch_excise(&h, GAMMA, R_EXC);
    dispatch_excise(&d, GAMMA, R_EXC);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();

    let (sh, sd) = (snapshot(&h), snapshot(&d));
    let names = ["rho", "v0", "v1", "pre", "den", "m0", "m1", "nrg"];
    let mut gap = 0.0_f64;
    for (i, (a, b)) in sh.iter().zip(sd.iter()).enumerate() {
        for k in 0..8 {
            assert!(
                b[k].is_finite(),
                "non-finite device {} at cell {i}",
                names[k]
            );
            let rel = (a[k] - b[k]).abs() / a[k].abs().max(1.0);
            assert!(
                rel < 1e-9,
                "excise {} at cell {i}: cpu {} != gpu {} (rel {rel:e})",
                names[k],
                a[k],
                b[k],
            );
            gap = gap.max(rel);
        }
    }
    // the fill actually rewrote the sphere on device (not a no-op that would pass vacuously).
    let dx = 2.0 * L / N as f64;
    let before = build::<CudaSpace, UnifiedMemory>(init);
    let sb = snapshot(&before);
    let mut n_changed = 0usize;
    for (i, c) in d.geom.interior.iter().enumerate() {
        let lo = d.geom.interior.spaces[0].lo;
        let x = d.geom.x_lo[0] + ((c[0] - lo) as f64 + 0.5) * dx;
        let y = d.geom.x_lo[1] + ((c[1] - lo) as f64 + 0.5) * dx;
        if (x * x + y * y).sqrt() < R_EXC - 2.0 * dx && sd[i] != sb[i] {
            n_changed += 1;
        }
    }
    assert!(
        n_changed > 20,
        "device excise never rewrote the deep sphere (got {n_changed})"
    );
}

#[test]
fn excise_source_cfl_dt_matches_cpu_on_device() {
    // the GR source-admissibility CFL kernel (rhd_source_cfl) zeroes its rate on the excised
    // sphere so the horizon cells do not collapse the timestep — a past device failure mode.
    // a device mask that failed to zero the excised rate would collapse dt on device and diverge
    // from the host dt. this is the only gate that exercises the source-cfl kernel on device.
    let init = |[x, y]: [f64; 2]| Prim {
        rho: 1.0 + 0.2 * (2.0 * x).sin() * (1.5 * y).cos(),
        vel: Tensor::new([0.08 * (x + y).cos(), -0.06 * (x - y).sin()]),
        pre: 0.05 + 0.01 * (x * y).cos(),
    };
    let h = build::<CpuSpace, HostMemory>(init);
    let d = build::<CudaSpace, UnifiedMemory>(init);
    // fill the excised sphere first: the cfl reads the post-excise state.
    dispatch_excise(&h, GAMMA, R_EXC);
    dispatch_excise(&d, GAMMA, R_EXC);
    let hk = RhdSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.3, &h.geom.allocated)
        .with_excision(R_EXC);
    let dk = RhdSubstrateKernelSet::<UnifiedMemory, f64, 2>::new(GAMMA, 0.3, &d.geom.allocated)
        .with_excision(R_EXC);
    let hdt = hk.cfl(&h);
    let ddt = dk.cfl(&d);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
    assert!(
        hdt.is_finite() && hdt > 1e-6,
        "host excised dt bad (not a real timestep): {hdt:e}"
    );
    assert!(
        ddt.is_finite() && ddt > 1e-6,
        "device excised dt collapsed: {ddt:e}"
    );
    let rel = (hdt - ddt).abs() / hdt;
    assert!(
        rel < 1e-9,
        "excised cfl dt host {hdt:e} != device {ddt:e} (rel {rel:e})"
    );
}

#[test]
fn excise_uniform_state_matches_cpu_on_device() {
    // a uniform primitive state: the fill sweeps are the identity on prims; the conserved
    // rebuild recomputes the same arithmetic on both backends. host and device must agree.
    let init = |_: [f64; 2]| Prim {
        rho: 1.3,
        vel: Tensor::new([0.05, -0.04]),
        pre: 0.02,
    };
    let h = build::<CpuSpace, HostMemory>(init);
    let d = build::<CudaSpace, UnifiedMemory>(init);
    dispatch_excise(&h, GAMMA, R_EXC);
    dispatch_excise(&d, GAMMA, R_EXC);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
    let (sh, sd) = (snapshot(&h), snapshot(&d));
    for (i, (a, b)) in sh.iter().zip(sd.iter()).enumerate() {
        for k in 0..8 {
            let rel = (a[k] - b[k]).abs() / a[k].abs().max(1.0);
            assert!(
                rel < 1e-9,
                "uniform excise field {k} at cell {i}: cpu {} != gpu {} (rel {rel:e})",
                a[k],
                b[k]
            );
        }
    }
}
