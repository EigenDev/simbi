// =============================================================================
// substrate_rmhd_spherical_gpu.rs
//
// GPU<->CPU parity for 3D CURVILINEAR (spherical) RMHD — the gate for the
// curvilinear FUSED god+bcell kernel on device. 3D + curvilinear +
// energy => fusable, so on GPU this sim runs the fused gas+bcell launch whose geo source
// reads cell-B via the predictor's `bc_k` key (the codegen dedup that makes the fuse
// alias-free). builds the SAME div-free B_r = B0/r^2 + pressure-bump shell on host and
// device, evolves a handful of RK2 steps, and asserts conserved state + cell B + the
// staggered radial face B agree. (cartesian fused GPU parity lives in substrate_rmhd_gpu.)
// =============================================================================

#![cfg(feature = "cuda")]

use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Spherical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};

const N: usize = 8;
const R_LO: f64 = 1.0;
const DR: f64 = 0.1;
const T_LO: f64 = 0.6;
const DTH: f64 = 0.06;
const P_LO: f64 = 0.2;
const DPH: f64 = 0.07;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const B0: f64 = 0.1;

fn build_sim<S: ExecutionSpace, Mem: MemorySpace>(
) -> SimStateGeneric<Rmhd, 3, 3, Spherical, IdealGas<f64>, S, Mem, f64> {
    SimStateGeneric::<Rmhd, 3, 3, Spherical, IdealGas<f64>, S, Mem, f64>::build(
        Rmhd,
        IdealGas { gamma: GAMMA },
        Spherical,
    )
    .cells([N, N, N])
    .origin([R_LO, T_LO, P_LO])
    .spacing([DR, DTH, DPH])
    .cfl(CFL)
    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
    .allocate()
    .expect("spherical RMHD sim")
    .set_initial(|[r, _th, _ph]| {
        let bc = B0 / (r * r);
        let pre = 1.0 + 0.3 * (-((r - 1.4) / 0.2).powi(2)).exp();
        MhdPrim {
            hydro: Prim { rho: 1.0, vel: Tensor::new([0.0, 0.0, 0.0]), pre },
            mag: Tensor::new([bc, 0.0, 0.0]),
        }
    })
    // div-free staggered B_r = B0/r^2 on the r-faces (area weighting makes it exactly div-free);
    // B_theta / B_phi faces stay zero.
    .seed_faces(|axis, x| match axis {
        0 => B0 / (x[0] * x[0]),
        _ => 0.0,
    })
    .build()
}

#[test]
fn rmhd_spherical_evolve_gpu_matches_cpu() {
    let mut host = build_sim::<CpuSpace, HostMemory>();
    let mut dev = build_sim::<CudaSpace, UnifiedMemory>();
    let hset = RmhdSubstrateKernelSet::<HostMemory, f64, 3>::new(GAMMA, CFL, 1.0, &host.geom.allocated);
    let dset = RmhdSubstrateKernelSet::<UnifiedMemory, f64, 3>::new(GAMMA, CFL, 1.0, &dev.geom.allocated);

    let t_final = 0.03_f64;
    evolve(&mut host, &hset, t_final).expect("cpu evolve");
    evolve(&mut dev, &dset, t_final).expect("gpu evolve");
    symbi_xpu::cuda::ctx_sync();

    assert!(host.iteration >= 2, "too few steps ({})", host.iteration);
    assert_eq!(host.iteration, dev.iteration, "step count diverged: cpu {} gpu {}", host.iteration, dev.iteration);

    let hmhd = host.fields.mhd.as_ref().unwrap();
    let dmhd = dev.fields.mhd.as_ref().unwrap();
    let hnrg = host.fields.cons.nrg_field().unwrap();
    let dnrg = dev.fields.cons.nrg_field().unwrap();
    let close = |g: f64, c: f64, what: &str, coord: [isize; 3]| {
        assert!(g.is_finite(), "{what} at {coord:?} non-finite on GPU: {g}");
        let rel = (g - c).abs() / c.abs().max(1.0);
        assert!(rel < 1e-9, "{what} at {coord:?}: gpu {g} != cpu {c} (rel {rel:e})");
    };
    for coord in host.geom.interior.iter() {
        close(*dev.fields.cons.den.view().at(coord), *host.fields.cons.den.view().at(coord), "cons.den", coord);
        close(*dnrg.view().at(coord), *hnrg.view().at(coord), "cons.nrg", coord);
        for k in 0..3 {
            close(*dev.fields.cons.mom[k].view().at(coord), *host.fields.cons.mom[k].view().at(coord), "cons.mom", coord);
            close(*dmhd.bcell[k].view().at(coord), *hmhd.bcell[k].view().at(coord), "bcell", coord);
        }
    }
    // the staggered radial face B (the CT-evolved B_r) — diff over its face domain.
    for fc in hmhd.bface[0].domain().clone().iter() {
        close(*dmhd.bface[0].view().at(fc), *hmhd.bface[0].view().at(fc), "bface_r", fc);
    }
}
