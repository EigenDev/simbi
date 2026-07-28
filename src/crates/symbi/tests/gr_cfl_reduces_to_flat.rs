// =============================================================================
// gr_cfl_reduces_to_flat.rs
//
// kerr-schild at M = 0 IS minkowski (alpha = 1, beta = 0, gamma = delta), so
// the GR RMHD CFL wave-speed map must produce the SAME timestep as the flat
// map on identical primitive state. the two kernels assemble the same
// magnetosonic bound through different operation orders, so the comparison is
// roundoff-tight, not bitwise. one kernel-set call each — no evolve, no CT —
// so a wave-speed-map defect is isolated from every downstream stage.
// =============================================================================

use std::f64::consts::PI;

use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi::sim::substrate_seam::KernelSet;
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, SchwarzschildKSCartesian};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 8;
const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.3;
const B0: f64 = 0.1;

// a nonuniform magnetized state so the map's velocity-addition and alfven
// terms all carry nonzero data.
fn swirl_prim(x: f64, y: f64, z: f64) -> MhdPrim<f64, 3> {
    let s = 2.0 * PI;
    MhdPrim {
        hydro: Prim {
            rho: 1.0,
            vel: Tensor::new([
                0.1 * (s * y).sin(),
                0.1 * (s * z).sin(),
                0.1 * (s * x).sin(),
            ]),
            pre: 0.5,
        },
        mag: Tensor::new([B0, 0.0, 0.0]),
    }
}

#[test]
fn zero_mass_ks_cfl_equals_flat_cfl() {
    let dx = 1.0 / N as f64;

    type GrSim = SimState<Rmhd, 3, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
    let mut sim_gr = GrSim::build(
        Rmhd,
        IdealGas { gamma: GAMMA },
        SchwarzschildKSCartesian { mass: 0.0 },
    )
    .cells([N; 3])
    .origin([1.2; 3])
    .spacing([dx; 3])
    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
    .cfl(CFL)
    .allocate()
    .expect("gr sim")
    .set_initial(|[x, y, z]| swirl_prim(x, y, z))
    .seed_faces(|axis, _| if axis == 0 { B0 } else { 0.0 })
    .build();

    type FlatSim = SimState<Rmhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let mut flat = FlatSim::build(Rmhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([1.2; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("flat sim")
        .set_initial(|[x, y, z]| swirl_prim(x, y, z))
        .seed_faces(|axis, _| if axis == 0 { B0 } else { 0.0 })
        .build();

    let sub_gr =
        RmhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, CFL, 1.0, &sim_gr.geom.allocated)
            .with_solver(Solver::Hlld)
            .expect("hlld");
    let sub_flat =
        RmhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, CFL, 1.0, &flat.geom.allocated)
            .with_solver(Solver::Hlld)
            .expect("hlld");

    // a zero-t_final evolve runs c2p + ghost_fill and no steps: it derives the
    // primitive fields from the identical conserved state on both chains.
    evolve_with_callback(&mut sim_gr, &sub_gr, 0.0, 1, |_| {}).expect("gr prime");
    evolve_with_callback(&mut flat, &sub_flat, 0.0, 1, |_| {}).expect("flat prime");

    let dt_gr = sub_gr.cfl(&sim_gr);
    let dt_flat = sub_flat.cfl(&flat);
    let rel = (dt_gr - dt_flat).abs() / dt_flat;
    assert!(
        rel < 1e-12,
        "M = 0 kerr-schild cfl diverges from flat: gr {dt_gr:.17e} flat {dt_flat:.17e} rel {rel:e}"
    );
}
