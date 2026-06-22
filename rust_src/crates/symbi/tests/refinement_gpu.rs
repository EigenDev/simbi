// =============================================================================
// refine_gpu.rs
//
// the amr gpu gate: the FULL 2-level berger-oliger driver (subcycle, prolong,
// restrict, flux-register reflux, mhd staggered transfer + emf reflux) on the
// CUDA backend (CudaSpace / UnifiedMemory), asserting the same invariants the
// cpu gates pin AND bitwise-comparable agreement with a host run of the
// identical problem. on the device backend the snapshots run the field_copy
// kernel and the flux register the field_fill / field_axpy_shift /
// refine_acc_face kernels — no per-cell host touch during compute (the emf
// register's codim-2 edge skeleton excepted).
//
// run: cargo test -p symbi --features cuda --test refine_gpu
// =============================================================================

#![cfg(feature = "cuda")]

use std::f64::consts::PI;
use std::sync::atomic::Ordering;

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet3D;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::regime::Regime;
use symbi_hydro::state::{Cons, Prim};
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 16;

// =============================================================================
// hydro: 2-level blast conservation on device + host-vs-device agreement
// =============================================================================

type HydroSim<S, Mem> = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, S, Mem>;

fn hydro_hier<S: ExecutionSpace, Mem: MemorySpace + Sync>() -> Hierarchy<
    Newtonian, 3, 3, Cartesian, IdealGas<f64>, S, Mem,
    AdiabaticSubstrateKernelSet<Mem, f64, 3>,
> {
    let dx = 1.0 / N as f64;
    let ic = |x: [f64; 3]| {
        let r2 = x.iter().map(|&q| (q - 0.5) * (q - 0.5)).sum::<f64>();
        let pre = if r2 < 0.01 { 10.0 } else { 0.1 };
        Prim { rho: 1.0, vel: Tensor::new([0.0; 3]), pre }
    };
    let coarse = HydroSim::<S, Mem>::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(ic)
        .build();
    let ck = AdiabaticSubstrateKernelSet::<Mem, f64, 3>::new(GAMMA, CFL, &coarse.geom.allocated);
    let regions = [RefinementRegion { x_lo: [0.375; 3], x_hi: [0.625; 3] }];
    let hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| {
        AdiabaticSubstrateKernelSet::<Mem, f64, 3>::new(GAMMA, CFL, &s.geom.allocated)
    })
    .unwrap();
    coarse_set_initial(&hier.levels[1].state, ic);
    hier
}

// set_initial is bounded to f64 sims generically; inline the fill so it works
// for any (S, Mem).
fn coarse_set_initial<S: ExecutionSpace, Mem: MemorySpace>(
    sim: &HydroSim<S, Mem>,
    ic: impl Fn([f64; 3]) -> Prim<f64, 3>,
) {
    let cnrg = sim.fields.cons.nrg_field().unwrap();
    for c in sim.geom.interior.iter() {
        let p = ic(sim.geom.centroid(c));
        let cons = sim.physics.regime.to_conserved(&sim.physics.eos, &p);
        sim.fields.cons.den.view_mut().set(c, cons.den);
        for k in 0..3 {
            sim.fields.cons.mom[k].view_mut().set(c, cons.mom[k]);
        }
        cnrg.view_mut().set(c, cons.nrg);
    }
}

fn composite_mass<R, S, Mem, K>(
    hier: &Hierarchy<R, 3, 3, Cartesian, IdealGas<f64>, S, Mem, K>,
) -> f64
where
    R: symbi_hydro::regime::Regime<f64, 3>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    K: symbi::sim::evolve::KernelSet<3, 3, Mem, f64>,
{
    let mut mass = 0.0;
    for lvl in &hier.levels {
        let vol: f64 = lvl.state.geom.dx.iter().product();
        let cov = lvl.coverage.as_ref();
        for c in lvl.state.geom.interior.iter() {
            if let Some(cov) = cov {
                if cov.contains(c) {
                    continue;
                }
            }
            mass += *lvl.state.fields.cons.den.view().at(c) * vol;
        }
    }
    mass
}

#[test]
fn gpu_two_level_blast_conserves_and_matches_host() {
    let mut dev = hydro_hier::<CudaSpace, UnifiedMemory>();
    let mut host = hydro_hier::<CpuSpace, HostMemory>();

    let m0 = composite_mass(&dev);
    dev.evolve_steps(4).unwrap();
    host.evolve_steps(4).unwrap();

    // conservation on device (the register kernel path).
    let m1 = composite_mass(&dev);
    let rel = ((m1 - m0) / m0).abs();
    assert!(rel < 1e-12, "gpu composite mass drift {rel:e}");

    // device vs host agreement on the full 2-level state (same algorithm,
    // backend ULP drift only).
    for ll in 0..2 {
        let d = &dev.levels[ll].state;
        let h = &host.levels[ll].state;
        let mut max_diff = 0.0f64;
        for c in d.geom.interior.iter() {
            let a = *d.fields.cons.den.view().at(c);
            let b = *h.fields.cons.den.view().at(c);
            assert!(a.is_finite(), "gpu level {ll} {c:?}: den {a}");
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 1e-9,
            "gpu vs cpu density diverged on level {ll}: max {max_diff:e}"
        );
    }
    assert_eq!(dev.levels[0].state.iteration, host.levels[0].state.iteration);
}

// =============================================================================
// mhd: 2-level orszag-tang divB on device
// =============================================================================

type MhdSim<S, Mem> = SimState<NewtonianMhd, 3, Cartesian, IdealGas<f64>, S, Mem>;

fn fill_ot<S: ExecutionSpace, Mem: MemorySpace>(sim: &MhdSim<S, Mem>) {
    const G: f64 = 5.0 / 3.0;
    const B0: f64 = 1.0;
    const V0: f64 = 0.5;
    let mhd = sim.fields.mhd.as_ref().unwrap();
    let dy = sim.geom.dx[1];
    let dxx = sim.geom.dx[0];
    for c in &sim.geom.interior.extend(0, 0, 1) {
        let y0 = sim.geom.x_lo[1] + c[1] as f64 * dy;
        let y1 = y0 + dy;
        mhd.bface[0].view_mut().set(c, B0 * ((2.0 * PI * y1).cos() - (2.0 * PI * y0).cos()) / (2.0 * PI * dy));
    }
    for c in &sim.geom.interior.extend(1, 0, 1) {
        let x0 = sim.geom.x_lo[0] + c[0] as f64 * dxx;
        let x1 = x0 + dxx;
        mhd.bface[1].view_mut().set(c, B0 * ((4.0 * PI * x0).cos() - (4.0 * PI * x1).cos()) / (4.0 * PI * dxx));
    }
    for c in &sim.geom.interior.extend(2, 0, 1) {
        mhd.bface[2].view_mut().set(c, 0.0);
    }
    mhd.bface_initialized.store(true, Ordering::Relaxed);
    for c in sim.geom.interior.iter() {
        let xc = sim.geom.centroid(c);
        let (x, y) = (xc[0], xc[1]);
        let prim = MhdPrim {
            hydro: Prim {
                rho: G * G,
                vel: Tensor::new([-V0 * (2.0 * PI * y).sin(), V0 * (2.0 * PI * x).sin(), 0.0]),
                pre: G,
            },
            mag: Tensor::new([-B0 * (2.0 * PI * y).sin(), B0 * (4.0 * PI * x).sin(), 0.0]),
        };
        let cons = sim.physics.regime.to_conserved(&sim.physics.eos, &prim);
        sim.fields.cons.scatter(c, Cons { den: cons.den, mom: cons.mom, nrg: cons.nrg });
        mhd.bcell[0].view_mut().set(c, prim.mag[0]);
        mhd.bcell[1].view_mut().set(c, prim.mag[1]);
        mhd.bcell[2].view_mut().set(c, 0.0);
    }
}

#[test]
fn gpu_two_level_mhd_preserves_divb() {
    const G: f64 = 5.0 / 3.0;
    const NZ: usize = 2;
    let dx = 1.0 / N as f64;
    let dz = 1.0 / NZ as f64;
    // build to Ready with a trivial seed (cells + uniform faces); fill_ot then overwrites the full
    // staggered OT state raw — the same post-construction fill the old ::new path used.
    let coarse = MhdSim::<CudaSpace, UnifiedMemory>::build(NewtonianMhd, IdealGas { gamma: G }, Cartesian)
        .cells([N, N, NZ])
        .spacing([dx, dx, dz])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .unwrap()
        .set_initial(|_| MhdPrim {
            hydro: Prim { rho: 1.0, vel: Tensor::new([0.0; 3]), pre: 1.0 },
            mag: Tensor::new([0.0; 3]),
        })
        .seed_faces_uniform([0.0; 3])
        .build();
    fill_ot(&coarse);
    let ck = NewtonianMhdSubstrateKernelSet3D::<UnifiedMemory, f64>::new(G, 0.3, 1.0, &coarse.geom.allocated);
    let regions = [RefinementRegion { x_lo: [0.25, 0.25, 0.0], x_hi: [0.75, 0.75, 1.0] }];
    let mut hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| {
        NewtonianMhdSubstrateKernelSet3D::<UnifiedMemory, f64>::new(G, 0.3, 1.0, &s.geom.allocated)
    })
    .unwrap();
    fill_ot(&hier.levels[1].state);

    hier.evolve_steps(4).unwrap();

    for (ll, lvl) in hier.levels.iter().enumerate() {
        let mhd = lvl.state.fields.mhd.as_ref().unwrap();
        let inv: [f64; 3] = std::array::from_fn(|ax| 1.0 / lvl.state.geom.dx[ax]);
        let mut max_div = 0.0f64;
        let mut max_b = 0.0f64;
        for c in lvl.state.geom.interior.iter() {
            let mut div = 0.0;
            let mut bsq = 0.0;
            for aa in 0..3 {
                let lo = *mhd.bface[aa].view().at(c);
                let mut ch = c;
                ch[aa] += 1;
                div += (*mhd.bface[aa].view().at(ch) - lo) * inv[aa];
                bsq += lo * lo;
            }
            max_div = max_div.max(div.abs());
            max_b = max_b.max(bsq.sqrt());
        }
        let rel = max_div / max_b.max(1.0);
        assert!(rel < 1e-12, "gpu divB grew on level {ll}: rel {rel:e}");
    }
}
