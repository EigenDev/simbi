// =============================================================================
// mhd_2p5d_gpu.rs
//
// GPU<->CPU parity for the genuine 2.5D MHD path (D=2, DOF=3) — the carrier gate
// for the 2.5D substrate kernels (the 3D parity lives in
// substrate_rmhd_gpu.rs). builds the same Orszag-Tang-with-Bz 2.5D sim on host
// (CpuSpace/HostMemory) and device (CudaSpace/UnifiedMemory), evolves a handful of
// RK2 steps through the production loop, and asserts the conserved state + cell B
// agree over the interior. the efield save/avg/snapshot copies must be dimensioned
// `_{D}d`: a 3d copy kernel over a 2d field reads out of bounds and crashes.
// =============================================================================

#![cfg(feature = "cuda")]

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};

use std::f64::consts::PI;

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 32;
const B0: f64 = 1.0;
const BZ0: f64 = 0.4;
const V0: f64 = 0.5;

fn build_sim<S: ExecutionSpace, Mem: MemorySpace>()
-> SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, S, Mem, f64> {
    let dx = 1.0 / N as f64;
    let rho0 = GAMMA * GAMMA;
    let p0 = GAMMA;
    SimStateGeneric::<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, S, Mem, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([N, N])
    .spacing([dx, dx])
    .cfl(CFL)
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .allocate()
    .expect("2.5d sim")
    .set_initial(|[x, y]| {
        let vel = Tensor::new([-V0 * (2.0 * PI * y).sin(), V0 * (2.0 * PI * x).sin(), 0.0]);
        let mag = Tensor::new([
            -B0 * (2.0 * PI * y).sin(),
            B0 * (4.0 * PI * x).sin(),
            BZ0 * (2.0 * PI * x).cos(),
        ]);
        MhdPrim {
            hydro: Prim {
                rho: rho0,
                vel,
                pre: p0,
            },
            mag,
        }
    })
    // face-normal B: bface[0] = -B0*sin(2*pi*y) (uses the face's y), bface[1] = B0*sin(4*pi*x).
    .seed_faces(|axis, x| match axis {
        0 => -B0 * (2.0 * PI * x[1]).sin(),
        _ => B0 * (4.0 * PI * x[0]).sin(),
    })
    .build()
}

#[test]
fn nmhd_2p5d_evolve_gpu_matches_cpu() {
    let mut host = build_sim::<CpuSpace, HostMemory>();
    let mut dev = build_sim::<CudaSpace, UnifiedMemory>();
    let hset = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
        GAMMA,
        CFL,
        1.0,
        &host.geom.allocated,
    );
    let dset = NewtonianMhdSubstrateKernelSet::<UnifiedMemory, f64, 2>::new(
        GAMMA,
        CFL,
        1.0,
        &dev.geom.allocated,
    );

    let t_final = 0.05_f64;
    evolve(&mut host, &hset, t_final).expect("cpu evolve");
    evolve(&mut dev, &dset, t_final).expect("gpu evolve");
    symbi_xpu::cuda::ctx_sync(); // host-read barrier for the final async c2p/ghost.

    assert!(host.iteration >= 3, "too few steps ({})", host.iteration);
    assert_eq!(
        host.iteration, dev.iteration,
        "step count diverged: cpu {} gpu {}",
        host.iteration, dev.iteration
    );

    let hmhd = host.fields.mhd.as_ref().unwrap();
    let dmhd = dev.fields.mhd.as_ref().unwrap();
    let hnrg = host.fields.cons.nrg_field().unwrap();
    let dnrg = dev.fields.cons.nrg_field().unwrap();
    let close = |g: f64, c: f64, what: &str, coord: [isize; 2]| {
        assert!(g.is_finite(), "{what} at {coord:?} non-finite on GPU: {g}");
        let rel = (g - c).abs() / c.abs().max(1.0);
        assert!(
            rel < 1e-6,
            "{what} at {coord:?}: gpu {g} != cpu {c} (rel {rel:e})"
        );
    };
    for coord in host.geom.interior.iter() {
        close(
            *dev.fields.cons.den.view().at(coord),
            *host.fields.cons.den.view().at(coord),
            "cons.den",
            coord,
        );
        close(
            *dnrg.view().at(coord),
            *hnrg.view().at(coord),
            "cons.nrg",
            coord,
        );
        for k in 0..3 {
            close(
                *dev.fields.cons.mom[k].view().at(coord),
                *host.fields.cons.mom[k].view().at(coord),
                "cons.mom",
                coord,
            );
            // bcell carries all 3 components incl. the out-of-plane Bz.
            close(
                *dmhd.bcell[k].view().at(coord),
                *hmhd.bcell[k].view().at(coord),
                "bcell",
                coord,
            );
        }
    }
}
