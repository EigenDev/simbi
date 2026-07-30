// =============================================================================
// mhd_2p5d_bench.rs
//
// throughput comparison of the GENUINE 2.5D MHD path (spatial D=2,
// vector DOF=3) vs the legacy 3D-with-nz=1 HACK, on the same Orszag-Tang vortex.
// the 2.5D run removes the wasted z-sweep, the z-ghosts, and the 3D loop overhead;
// the cache-tiling executor (policy_for, D-generic) then tiles the genuine 2D grid.
// reports compute-only MZCS (mega-zone-cycles/sec) for both + the speedup.
//
// usage: cargo run --release --example mhd_2p5d_bench -- [nx] [steps_t_final]
//   (defaults: 256, t_final = 0.1) — keep it short; this is a throughput probe.
// =============================================================================

use std::f64::consts::PI;
use std::time::Instant;

use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_newtonian_mhd::{
    NewtonianMhdSubstrateKernelSet, NewtonianMhdSubstrateKernelSet3D,
};
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;

// backend: device (DeviceMemory + DeviceSpace) under --features cuda, else host CPU.
// the same substrate code path; this bench measures whichever the build selects.
#[cfg(not(feature = "gpu"))]
use symbi_xpu::{CpuSpace as Space, HostMemory as Mem};
#[cfg(feature = "gpu")]
use symbi_xpu::{DeviceMemory as Mem, DeviceSpace as Space};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const THETA: f64 = 1.5;

// the OT primitive vectors (vel, B 3-vectors; Bz=0) at physical position (x,y).
fn ot_vectors(x: f64, y: f64, b0: f64) -> (Tensor<f64, 3>, Tensor<f64, 3>) {
    let vel = Tensor::new([-(2.0 * PI * y).sin(), (2.0 * PI * x).sin(), 0.0]);
    let mag = Tensor::new([-b0 * (2.0 * PI * y).sin(), b0 * (4.0 * PI * x).sin(), 0.0]);
    (vel, mag)
}

fn bench_2p5d(nx: usize, t_final: f64) -> (f64, u64) {
    let dx = 1.0 / nx as f64;
    let rho0 = GAMMA * GAMMA;
    let p0 = GAMMA;
    let b0 = 1.0 / (4.0 * PI).sqrt();
    let mut sim =
        SimStateGeneric::<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, Space, Mem>::build(
            NewtonianMhd,
            IdealGas { gamma: GAMMA },
            Cartesian,
        )
        .cells([nx, nx])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("2.5d sim")
        .set_initial(|[x, y]| {
            let (vel, mag) = ot_vectors(x, y, b0);
            MhdPrim {
                hydro: Prim {
                    rho: rho0,
                    vel,
                    pre: p0,
                },
                mag,
            }
        })
        .seed_faces(|axis, [x, y]| match axis {
            0 => -b0 * (2.0 * PI * y).sin(),
            _ => b0 * (4.0 * PI * x).sin(),
        })
        .build();
    let sub =
        NewtonianMhdSubstrateKernelSet::<Mem, f64, 2>::new(GAMMA, CFL, THETA, &sim.geom.allocated)
            .with_solver(Solver::Hlld)
            .expect("HLLD is valid for Newtonian MHD");
    let cells = sim.geom.interior.volume() as u64;
    let t0 = Instant::now();
    evolve_with_callback(&mut sim, &sub, t_final, 1_000_000, |_| {}).expect("2.5d evolve");
    let secs = t0.elapsed().as_secs_f64();
    let mzcs = (sim.iteration as f64 * cells as f64) / (secs * 1.0e6);
    (mzcs, sim.iteration)
}

fn bench_3d_nz1(nx: usize, t_final: f64) -> (f64, u64) {
    let dx = 1.0 / nx as f64;
    let rho0 = GAMMA * GAMMA;
    let p0 = GAMMA;
    let b0 = 1.0 / (4.0 * PI).sqrt();
    let mut sim = SimState::<NewtonianMhd, 3, Cartesian, IdealGas<f64>, Space, Mem>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([nx, nx, 1])
    .spacing([dx, dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(CFL)
    .timestepping(Timestepping::Rk2)
    .allocate()
    .expect("3d sim")
    .set_initial(|[x, y, _z]| {
        let (vel, mag) = ot_vectors(x, y, b0);
        MhdPrim {
            hydro: Prim {
                rho: rho0,
                vel,
                pre: p0,
            },
            mag,
        }
    })
    .seed_faces(|axis, [x, y, _z]| match axis {
        0 => -b0 * (2.0 * PI * y).sin(),
        1 => b0 * (4.0 * PI * x).sin(),
        _ => 0.0,
    })
    .build();
    let sub =
        NewtonianMhdSubstrateKernelSet3D::<Mem, f64>::new(GAMMA, CFL, THETA, &sim.geom.allocated)
            .with_solver(Solver::Hlld)
            .expect("HLLD is valid for Newtonian MHD");
    let cells = sim.geom.interior.volume() as u64;
    #[cfg(feature = "gpu")]
    let launches0 = symbi::regimes::substrate_gpu::gpu_launch_count();
    let t0 = Instant::now();
    evolve_with_callback(&mut sim, &sub, t_final, 1_000_000, |_| {}).expect("3d evolve");
    let secs = t0.elapsed().as_secs_f64();
    #[cfg(feature = "gpu")]
    {
        let launches = symbi::regimes::substrate_gpu::gpu_launch_count() - launches0;
        eprintln!(
            "  [gpu] {launches} launches over {} steps = {:.0} launches/step; {:.2} ms/step",
            sim.iteration,
            launches as f64 / sim.iteration as f64,
            secs * 1e3 / sim.iteration as f64,
        );
    }
    let mzcs = (sim.iteration as f64 * cells as f64) / (secs * 1.0e6);
    (mzcs, sim.iteration)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let nx: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let t_final: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.1);

    eprintln!("[mhd_2p5d_bench] Orszag-Tang {nx}x{nx}, HLLD, t_final={t_final}");
    let (m3, i3) = bench_3d_nz1(nx, t_final);
    eprintln!("  3D-with-nz=1 (hack): {m3:7.2} MZCS  ({i3} steps)");
    // 2.5D is skipped by default: its GPU kernels have no device-validation gate.
    // set SYMBI_BENCH_2P5D=1 to include it (CPU OK).
    if std::env::var("SYMBI_BENCH_2P5D").is_ok() {
        let (m2, i2) = bench_2p5d(nx, t_final);
        eprintln!("  genuine 2.5D       : {m2:7.2} MZCS  ({i2} steps)");
        eprintln!("  speedup            : {:.2}x", m2 / m3);
    }
}
