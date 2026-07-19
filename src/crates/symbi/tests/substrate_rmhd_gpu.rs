// =============================================================================
// substrate_rmhd_gpu.rs
//
// the RMHD substrate KernelSet runs on the
// GPU through the production dispatch path. build TWO identical RMHD sims — one on
// host memory (CpuSpace), one on unified memory (CudaSpace) — and run the SAME
// `RmhdSubstrateKernelSet3D` method on each. the unified-memory `Mem` makes
// `dispatch` route every kernel to `run_gpu` (render the neutral IR -> NVRTC ->
// launch); host memory routes to the AOT CPU kernel. the device prim must match the
// host prim (modulo nvcc FMA fusion).
//
// runs on the HOST GPU (NVRTC needs no nvcc). run:
//   cargo test -p symbi --features cuda --test substrate_rmhd_gpu
// =============================================================================

#![cfg(feature = "cuda")]

use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::evolve::KernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const B0: [f64; 3] = [0.3, 0.4, 0.2];
const N: usize = 8;

// build an RMHD sim on (S, Mem) with a smooth periodic hydro state + uniform B,
// conserved via the production forward map — identical numbers for any memory space
// (unified is host-addressable, so the host init writes land on-device too).
fn build_sim<S: ExecutionSpace, Mem: MemorySpace>(
) -> SimState<Rmhd, 3, Cartesian, IdealGas<f64>, S, Mem> {
    let dx = 1.0 / N as f64;
    let pi = std::f64::consts::PI;
    let amp = 0.1;
    SimState::<Rmhd, 3, Cartesian, IdealGas<f64>, S, Mem>::build(
        Rmhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([N, N, N])
    .spacing([dx, dx, dx])
    .cfl(CFL)
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .allocate()
    .expect("sim construction failed")
    .set_initial(|[x, y, z]| MhdPrim {
        hydro: Prim {
            rho: 1.0 + amp * (2.0 * pi * x).sin(),
            vel: Tensor::new([
                0.1 * (2.0 * pi * y).cos(),
                0.1 * (2.0 * pi * z).sin(),
                0.05 * (2.0 * pi * x).cos(),
            ]),
            pre: 1.0 + amp * (2.0 * pi * y).sin(),
        },
        mag: Tensor::new(B0),
    })
    // uniform staggered B on each face (div-free, trivially staggered) — the CT path reads bface.
    .seed_faces_uniform(B0)
    .build()
}

// GPU vs CPU agree modulo nvcc FMA fusion: ULP-bounded drift.
fn assert_close(gpu: f64, cpu: f64, what: &str, coord: [isize; 3]) {
    let rel = (gpu - cpu).abs() / cpu.abs().max(1.0);
    assert!(rel < 1e-9, "{what} at {coord:?}: gpu {gpu} != cpu {cpu} (rel {rel:e})");
}

#[test]
fn substrate_rmhd_c2p_gpu_matches_cpu() {
    let host = build_sim::<CpuSpace, HostMemory>();
    let dev = build_sim::<CudaSpace, UnifiedMemory>();

    // host memory -> AOT CPU kernel; unified memory -> run_gpu (render IR + NVRTC).
    RmhdSubstrateKernelSet3D::<HostMemory>::new(GAMMA, CFL, 1.0, &host.geom.allocated).c2p(&host);
    RmhdSubstrateKernelSet3D::<UnifiedMemory>::new(GAMMA, CFL, 1.0, &dev.geom.allocated).c2p(&dev);
    symbi_xpu::cuda::ctx_sync(); // host-read barrier: run_gpu doesn't sync per-launch

    let hp = host.fields.prim.pre_field().unwrap();
    let dp = dev.fields.prim.pre_field().unwrap();
    let mut nontrivial = false;
    for coord in host.geom.interior.iter() {
        assert_close(
            *dev.fields.prim.rho.view().at(coord), *host.fields.prim.rho.view().at(coord), "rho", coord,
        );
        for k in 0..3 {
            assert_close(
                *dev.fields.prim.vel[k].view().at(coord), *host.fields.prim.vel[k].view().at(coord),
                "vel", coord,
            );
        }
        assert_close(*dp.view().at(coord), *hp.view().at(coord), "pre", coord);
        if (*host.fields.prim.rho.view().at(coord) - 1.0).abs() > 1e-6 {
            nontrivial = true;
        }
    }
    assert!(nontrivial, "c2p output trivially uniform — test would be vacuous");
}

// the harder paths on GPU: ghost_fill exercises the INT param lane (map_type/arg)
// + the lattice-map gather + in-place; flux exercises a 16-buffer reorder, the
// staggered bflux output, and the 3D buf_extent stride args. c2p -> ghost_fill ->
// flux on both, compare the face fluxes.
#[test]
fn substrate_rmhd_flux_gpu_matches_cpu() {
    let host = build_sim::<CpuSpace, HostMemory>();
    let dev = build_sim::<CudaSpace, UnifiedMemory>();
    let hset = RmhdSubstrateKernelSet3D::<HostMemory>::new(GAMMA, CFL, 1.0, &host.geom.allocated);
    let dset = RmhdSubstrateKernelSet3D::<UnifiedMemory>::new(GAMMA, CFL, 1.0, &dev.geom.allocated);

    hset.c2p(&host);
    hset.ghost_fill(&host);
    dset.c2p(&dev);
    dset.ghost_fill(&dev);

    // materialize the per-cell wave speeds the HLLE flux now reads (else it reads zeroed
    // fields -> degenerate s_l=s_r=0 -> the f_l branch, exercising nothing). this is the same
    // ordering the evolve loop uses (wave_speeds before flux).
    hset.wave_speeds(&host);
    dset.wave_speeds(&dev);

    for dir in 0..3 {
        hset.flux(&host, dir);
        dset.flux(&dev, dir);
        // run_gpu does NOT ctx_sync per-launch; a host read of device-written buffers
        // races with the in-flight kernel. unified memory makes already-resident pages (the
        // interior) look correct while freshly-written ghost pages read stale zero-init —
        // exactly the corner-ghost mismatch this test hit. sync before the host comparison.
        symbi_xpu::cuda::ctx_sync();
        let mut face = host.geom.interior.extend(dir, 0, 1);
        for ax in 0..3 {
            if ax != dir {
                face = face.expand(ax, 1);
            }
        }
        let hmhd = host.fields.mhd.as_ref().unwrap();
        let dmhd = dev.fields.mhd.as_ref().unwrap();
        let (hnrg, dnrg) =
            (host.fields.flux[dir].nrg_field().unwrap(), dev.fields.flux[dir].nrg_field().unwrap());
        for coord in face.iter() {
            assert_close(
                *dev.fields.flux[dir].den.view().at(coord), *host.fields.flux[dir].den.view().at(coord),
                "flux.den", coord,
            );
            for k in 0..3 {
                assert_close(
                    *dev.fields.flux[dir].mom[k].view().at(coord),
                    *host.fields.flux[dir].mom[k].view().at(coord), "flux.mom", coord,
                );
                assert_close(
                    *dmhd.bflux[dir][k].view().at(coord), *hmhd.bflux[dir][k].view().at(coord),
                    "bflux", coord,
                );
            }
            assert_close(*dnrg.view().at(coord), *hnrg.view().at(coord), "flux.nrg", coord);
        }
    }
}

// the FULL evolve loop on the GPU. drive the real RK2 step
// driver (cfl -> flux -> efield -> godunov -> CT -> c2p -> ghost, multiple steps)
// on host + unified sims with identical init; the substrate KernelSet routes every
// kernel to the GPU for unified memory. assert the run completes NaN-free and the
// conserved + magnetic state matches CPU. SHORT smoke (a handful of steps on an 8^3
// smooth box) — not a physics run. tolerance is loose: FMA drift compounds per step.
#[test]
fn substrate_rmhd_evolve_gpu_matches_cpu() {
    use symbi::sim::evolve::evolve;

    let mut host = build_sim::<CpuSpace, HostMemory>();
    let mut dev = build_sim::<CudaSpace, UnifiedMemory>();
    let hset = RmhdSubstrateKernelSet3D::<HostMemory>::new(GAMMA, CFL, 1.0, &host.geom.allocated);
    let dset = RmhdSubstrateKernelSet3D::<UnifiedMemory>::new(GAMMA, CFL, 1.0, &dev.geom.allocated);

    // a small t_final -> a handful of RK2 steps; dt is cfl-clamped to land both
    // backends exactly on t_final, so they take the same step count.
    let t_final = 0.3_f64;
    evolve(&mut host, &hset, t_final).expect("cpu evolve");
    evolve(&mut dev, &dset, t_final).expect("gpu evolve");
    symbi_xpu::cuda::ctx_sync(); // host-read barrier: the final step's c2p/ghost run async after
                                 // the cfl sync; without this the interior read can race them.

    assert!(host.iteration >= 3, "too few steps ({}) — smoke would be vacuous", host.iteration);
    assert_eq!(
        host.iteration, dev.iteration,
        "step count diverged: cpu {} vs gpu {}", host.iteration, dev.iteration,
    );

    // compare the conserved state + cell B over the interior: NaN-free + close.
    let hmhd = host.fields.mhd.as_ref().unwrap();
    let dmhd = dev.fields.mhd.as_ref().unwrap();
    let hnrg = host.fields.cons.nrg_field().unwrap();
    let dnrg = dev.fields.cons.nrg_field().unwrap();
    // looser than a single kernel: ~N steps of RK2 + iterative c2p compound FMA drift.
    let evolve_close = |g: f64, c: f64, what: &str, coord: [isize; 3]| {
        assert!(g.is_finite(), "{what} at {coord:?} went non-finite on GPU: {g}");
        let rel = (g - c).abs() / c.abs().max(1.0);
        assert!(rel < 1e-6, "{what} at {coord:?}: gpu {g} != cpu {c} (rel {rel:e})");
    };
    for coord in host.geom.interior.iter() {
        evolve_close(*dev.fields.cons.den.view().at(coord), *host.fields.cons.den.view().at(coord), "cons.den", coord);
        for k in 0..3 {
            evolve_close(
                *dev.fields.cons.mom[k].view().at(coord), *host.fields.cons.mom[k].view().at(coord),
                "cons.mom", coord,
            );
            evolve_close(*dmhd.bcell[k].view().at(coord), *hmhd.bcell[k].view().at(coord), "bcell", coord);
        }
        evolve_close(*dnrg.view().at(coord), *hnrg.view().at(coord), "cons.nrg", coord);
    }
}

// the Reduce-morphism proof in isolation: the GPU block-reduction max over a
// field's INTERIOR window must equal the host max — bit-exact (max has no FMA) — AND
// must ignore values outside the window (a huge ghost value is not reduced). this
// pins the device reduction + its view_t windowing independently of the full step.
#[test]
fn field_max_reduce_device_matches_host_and_respects_window() {
    use symbi::regimes::substrate_gpu::field_max_reduce;
    use symbi_algebra::{Domain, Space};
    use symbi_grid::Field;

    let alloc = Domain::new([
        Space { name: "x", lo: 0, hi: 8 },
        Space { name: "y", lo: 0, hi: 8 },
        Space { name: "z", lo: 0, hi: 8 },
    ]);
    let interior = Domain::new([
        Space { name: "x", lo: 2, hi: 6 },
        Space { name: "y", lo: 2, hi: 6 },
        Space { name: "z", lo: 2, hi: 6 },
    ]);
    let f = Field::<f64, 3, UnifiedMemory>::zeros(&alloc).unwrap();
    // smooth distinct values everywhere; a HUGE spike in a GHOST cell (outside the
    // interior) that the windowed reduction must NOT see.
    for c in alloc.iter() {
        let v = 0.1 + 0.01 * (c[0] + 3 * c[1] + 7 * c[2]) as f64;
        f.view_mut().set(c, v);
    }
    f.view_mut().set([0, 0, 0], 1.0e6); // ghost spike — outside [2,6)^3

    let mut host_max = f64::NEG_INFINITY;
    for c in interior.iter() {
        host_max = host_max.max(*f.view().at(c));
    }
    let dev_max = field_max_reduce(&f, &interior);
    assert_eq!(dev_max, host_max, "device reduction != host max over interior");
    assert!(dev_max < 1.0e6, "windowing failed: the ghost spike leaked into the reduction");
}

// the morphism is op-generic: Add/Min/Max all reduce on-device matching the host.
// min/max are exact (no FMA); add differs from the host's sequential fold only by
// reassociated rounding, so it's tolerance-checked.
#[test]
fn field_reduce_device_all_ops_match_host() {
    use symbi::regimes::substrate_gpu::field_reduce;
    use symbi_algebra::{Domain, Space};
    use symbi_grid::Field;
    use symbi_ir::emit::ReductionOp;

    let alloc = Domain::new([
        Space { name: "x", lo: 0, hi: 8 },
        Space { name: "y", lo: 0, hi: 8 },
        Space { name: "z", lo: 0, hi: 8 },
    ]);
    let interior = Domain::new([
        Space { name: "x", lo: 2, hi: 6 },
        Space { name: "y", lo: 2, hi: 6 },
        Space { name: "z", lo: 2, hi: 6 },
    ]);
    let f = Field::<f64, 3, UnifiedMemory>::zeros(&alloc).unwrap();
    for c in alloc.iter() {
        f.view_mut().set(c, 0.5 + 0.001 * (c[0] + 5 * c[1] + 11 * c[2]) as f64);
    }
    let cells: Vec<f64> = interior.iter().map(|c| *f.view().at(c)).collect();
    let hmax = cells.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let hmin = cells.iter().cloned().fold(f64::INFINITY, f64::min);
    let hsum: f64 = cells.iter().sum();
    let hprod: f64 = cells.iter().product();

    // min/max are bit-exact (no FMA); add/mul differ only by reassociated rounding.
    assert_eq!(field_reduce(&f, &interior, ReductionOp::Max), hmax, "device max");
    assert_eq!(field_reduce(&f, &interior, ReductionOp::Min), hmin, "device min");
    let dsum = field_reduce(&f, &interior, ReductionOp::Add);
    assert!((dsum - hsum).abs() < 1e-9 * hsum.abs().max(1.0), "device add {dsum} != host {hsum}");
    let dprod = field_reduce(&f, &interior, ReductionOp::Mul);
    assert!((dprod - hprod).abs() < 1e-9 * hprod.abs().max(1e-30), "device mul {dprod} != host {hprod}");
}

// keystone: a single NaN cell must survive the ON-DEVICE block reduction (the
// in-block ternary `a > b ? a : b` drops NaN unless guarded with `x != x`). this
// validates the CUDA combine matches the host NaN-propagation so a poisoned cell
// reaches the dt guard instead of being silently averaged away on the GPU.
#[test]
fn device_reduction_propagates_single_nan_cell() {
    use symbi::regimes::substrate_gpu::field_reduce;
    use symbi_algebra::{Domain, Space};
    use symbi_grid::Field;
    use symbi_ir::emit::ReductionOp;

    let alloc = Domain::new([
        Space { name: "x", lo: 0, hi: 8 },
        Space { name: "y", lo: 0, hi: 8 },
        Space { name: "z", lo: 0, hi: 8 },
    ]);
    let interior = Domain::new([
        Space { name: "x", lo: 2, hi: 6 },
        Space { name: "y", lo: 2, hi: 6 },
        Space { name: "z", lo: 2, hi: 6 },
    ]);
    let f = Field::<f64, 3, UnifiedMemory>::zeros(&alloc).unwrap();
    for c in alloc.iter() {
        f.view_mut().set(c, 0.5 + 0.001 * (c[0] + 5 * c[1] + 11 * c[2]) as f64);
    }
    f.view_mut().set([3, 3, 3], f64::NAN); // one poisoned interior cell

    assert!(field_reduce(&f, &interior, ReductionOp::Max).is_nan(), "device max dropped NaN");
    assert!(field_reduce(&f, &interior, ReductionOp::Min).is_nan(), "device min dropped NaN");
}
