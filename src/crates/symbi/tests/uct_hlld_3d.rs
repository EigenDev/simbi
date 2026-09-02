// =============================================================================
// uct_hlld_3d.rs
//
// the flat-cartesian 3D UCT family, end to end through the production MHD
// substrates: all three corner-EMF edge kernels (k = 0, 1, 2) dispatch and the
// staggered div(B) stays at machine zero under evolve. the IC is an
// Orszag-Tang vortex with an out-of-plane swirl (v_z != 0, B_z(x) != 0), so
// the x- and y-edge EMFs carry genuinely nonzero physics — a plain 2D OT
// leaves them identically zero and would pass with broken curls. gates:
// - NMHD under UCT-HLLD and UCT-HLL (the five-wave and the regime-generic
//   corner families), IMHD and RMHD under UCT-HLLD;
// - the dimensional-degeneracy oracle: an in-plane problem (v_z = B_z = 0)
//   run z-invariantly in 3D reproduces the 2.5D run column by column — the
//   z flux divergence cancels exactly and the x/y-edge EMFs are exact zeros,
//   so the surviving arithmetic is the 2.5D chain.
// =============================================================================

use std::f64::consts::PI;
use symbi_hydro::quantity::{Density, Pressure};

use symbi::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet3D;
use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_newtonian_mhd::{
    NewtonianMhdSubstrateKernelSet, NewtonianMhdSubstrateKernelSet3D,
};
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const NX: usize = 8;
const NY: usize = 8;
const NZ: usize = 4;
const GAMMA: f64 = 5.0 / 3.0;
const CS: f64 = 1.0;
const CFL: f64 = 0.3;
const B0: f64 = 1.0;
const T_FINAL: f64 = 0.2;
const DIVB_TOL: f64 = 1e-12;

// staggered face div(B), max over the interior, plus max |B| for the relative scale.
fn max_divb<R, E>(
    sim: &SimState<R, 3, Cartesian, E, CpuSpace, HostMemory>,
    inv_d: [f64; 3],
) -> (f64, f64)
where
    R: symbi_hydro::regime::Regime<f64, 3>,
    E: symbi_hydro::eos::Eos<f64>,
{
    let mhd = sim.fields.mhd.as_ref().expect("mhd");
    let (mut max_div, mut max_b) = (0.0_f64, 0.0_f64);
    for c in sim.geom.interior.iter() {
        let lo: [f64; 3] = std::array::from_fn(|a| *mhd.bface[a].view().at(c));
        let hi: [f64; 3] = std::array::from_fn(|a| {
            let mut n = c;
            n[a] += 1;
            *mhd.bface[a].view().at(n)
        });
        let div: f64 = (0..3).map(|a| (hi[a] - lo[a]) * inv_d[a]).sum();
        max_div = max_div.max(div.abs());
        max_b = max_b.max((lo.iter().map(|b| b * b).sum::<f64>()).sqrt());
    }
    (max_div, max_b)
}

// the swirled OT: the classic in-plane vortex plus v_z(y) and B_z(x), all
// z-independent (periodic z at any extent). div(B) = 0 analytically: B_x(y),
// B_y(x), B_z(x) each vanish under their own derivative.
fn swirl_prim(x: f64, y: f64, v0: f64, rho0: f64, p0: f64) -> MhdPrim<f64, 3> {
    MhdPrim::new(
        Prim::adiabatic(
            Density(rho0),
            Tensor::new([
                -v0 * (2.0 * PI * y).sin(),
                v0 * (2.0 * PI * x).sin(),
                0.5 * v0 * (2.0 * PI * y).sin(),
            ]),
            Pressure(p0),
        ),
        Tensor::new([
            -B0 * (2.0 * PI * y).sin(),
            B0 * (4.0 * PI * x).sin(),
            0.5 * B0 * (2.0 * PI * x).sin(),
        ]),
    )
}

fn swirl_face(axis: usize, x: f64, y: f64) -> f64 {
    match axis {
        0 => -B0 * (2.0 * PI * y).sin(),
        1 => B0 * (4.0 * PI * x).sin(),
        _ => 0.5 * B0 * (2.0 * PI * x).sin(),
    }
}

macro_rules! run_3d_divb {
    ($regime:expr, $regime_ty:ty, $set:ty, $eos_param:expr, $v0:expr, $rho0:expr, $p0:expr, $solver:expr, $what:literal) => {{
        type Sim = SimState<$regime_ty, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
        let d = [1.0 / NX as f64, 1.0 / NY as f64, 1.0 / NZ as f64];
        let mut sim = Sim::build($regime, IdealGas { gamma: GAMMA }, Cartesian)
            .cells([NX, NY, NZ])
            .spacing(d)
            .boundaries(Boundaries::uniform(BoundaryType::Periodic))
            .cfl(CFL)
            .allocate()
            .expect("sim")
            .set_initial(|[x, y, _z]| swirl_prim(x, y, $v0, $rho0, $p0))
            .seed_faces(|axis, [x, y, _z]| swirl_face(axis, x, y))
            .build();
        let inv_d = [NX as f64, NY as f64, NZ as f64];
        let (div0, b_max) = max_divb(&sim, inv_d);
        assert!(
            div0 / b_max.max(1.0) < 1e-13,
            "{}: IC not div-free: {div0:e}",
            $what
        );

        let sub = <$set>::new($eos_param, CFL, 1.0, &sim.geom.allocated)
            .with_solver($solver)
            .expect("valid solver/regime pair")
            .ct_method(CtMethod::Uct);
        let mut steps: u64 = 0;
        evolve_with_callback(&mut sim, &sub, T_FINAL, 1, |s| {
            let (max_div, max_b) = max_divb(s, inv_d);
            let rel = max_div / max_b.max(1.0);
            assert!(
                rel < DIVB_TOL,
                "{}: div(B) grew at iter {} t={:.3e}: rel {rel:e}",
                $what,
                s.iteration,
                s.time,
            );
            steps = s.iteration;
        })
        .expect("evolve");
        assert!(
            steps >= 5,
            "{}: only {steps} steps — gate barely exercised",
            $what
        );
    }};
}

#[test]
fn nmhd_uct_hlld_3d_preserves_divb() {
    run_3d_divb!(
        NewtonianMhd, NewtonianMhd, NewtonianMhdSubstrateKernelSet3D<HostMemory, f64>,
        GAMMA, 0.5, GAMMA * GAMMA, GAMMA, Solver::Hlld, "nmhd uct-hlld 3d"
    );
}

#[test]
fn nmhd_uct_hll_3d_preserves_divb() {
    run_3d_divb!(
        NewtonianMhd, NewtonianMhd, NewtonianMhdSubstrateKernelSet3D<HostMemory, f64>,
        GAMMA, 0.5, GAMMA * GAMMA, GAMMA, Solver::Hlle, "nmhd uct-hll 3d"
    );
}

#[test]
fn imhd_uct_hlld_3d_preserves_divb() {
    run_3d_divb!(
        NewtonianMhd, NewtonianMhd, IsothermalMhdSubstrateKernelSet3D<HostMemory, f64>,
        CS, 0.5, 1.0, CS * CS, Solver::Hlld, "imhd uct-hlld 3d"
    );
}

#[test]
fn rmhd_uct_hlld_3d_preserves_divb() {
    run_3d_divb!(
        Rmhd, Rmhd, RmhdSubstrateKernelSet3D<HostMemory, f64>,
        GAMMA, 0.3, 1.0, 1.0, Solver::Hlld, "rmhd uct-hlld 3d"
    );
}

// the in-plane OT (v_z = B_z = 0): every x/y-edge EMF is an exact zero (all its
// contributing products carry a zero factor) and the z flux divergence cancels
// bitwise (identical faces subtract), so the 3D chain must reproduce the 2.5D
// run column by column. dz is fat so the z axis never binds the CFL.
#[test]
fn z_invariant_uct_hlld_3d_matches_2p5d_columns() {
    const T: f64 = 0.1;
    let inplane_prim = |x: f64, y: f64| -> MhdPrim<f64, 3> {
        MhdPrim::new(
            Prim::adiabatic(
                Density(GAMMA * GAMMA),
                Tensor::new([-0.5 * (2.0 * PI * y).sin(), 0.5 * (2.0 * PI * x).sin(), 0.0]),
                Pressure(GAMMA),
            ),
            Tensor::new([-B0 * (2.0 * PI * y).sin(), B0 * (4.0 * PI * x).sin(), 0.0]),
        )
    };
    let inplane_face = |axis: usize, x: f64, y: f64| -> f64 {
        match axis {
            0 => -B0 * (2.0 * PI * y).sin(),
            1 => B0 * (4.0 * PI * x).sin(),
            _ => 0.0,
        }
    };

    type Sim3 = SimState<NewtonianMhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let (dx, dy, dz) = (1.0 / NX as f64, 1.0 / NY as f64, 10.0);
    let mut s3 = Sim3::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([NX, NY, NZ])
        .spacing([dx, dy, dz])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .expect("3d sim")
        .set_initial(|[x, y, _z]| inplane_prim(x, y))
        .seed_faces(|axis, [x, y, _z]| inplane_face(axis, x, y))
        .build();
    let k3 = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(
        GAMMA,
        CFL,
        1.0,
        &s3.geom.allocated,
    )
    .with_solver(Solver::Hlld)
    .expect("hlld")
    .ct_method(CtMethod::Uct);
    evolve_with_callback(&mut s3, &k3, T, 1, |_| {}).expect("3d evolve");

    // the 2.5D twin: 2 grid axes, 3 vector components (the DOF-lifted state).
    let mut s2 = SimStateGeneric::<
        NewtonianMhd,
        2,
        3,
        Cartesian,
        IdealGas<f64>,
        CpuSpace,
        HostMemory,
    >::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
    .cells([NX, NY])
    .spacing([dx, dy])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(CFL)
    .allocate()
    .expect("2d sim")
    .set_initial(|[x, y]| inplane_prim(x, y))
    .seed_faces(|axis, [x, y]| inplane_face(axis, x, y))
    .build();
    let k2 = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
        GAMMA,
        CFL,
        1.0,
        &s2.geom.allocated,
    )
    .with_solver(Solver::Hlld)
    .expect("hlld")
    .ct_method(CtMethod::Uct);
    evolve_with_callback(&mut s2, &k2, T, 1, |_| {}).expect("2d evolve");

    assert_eq!(
        s3.iteration, s2.iteration,
        "step counts diverged: the z axis bound the CFL"
    );

    let mhd3 = s3.fields.mhd.as_ref().expect("mhd3");
    let mhd2 = s2.fields.mhd.as_ref().expect("mhd2");
    let lo3: [isize; 3] = std::array::from_fn(|a| s3.geom.interior.spaces[a].lo);
    let lo2: [isize; 2] = std::array::from_fn(|a| s2.geom.interior.spaces[a].lo);
    let mut checked = 0usize;
    for jj in 0..NY as isize {
        for ii in 0..NX as isize {
            let c2 = [lo2[0] + ii, lo2[1] + jj];
            for kk in 0..NZ as isize {
                let c3 = [lo3[0] + ii, lo3[1] + jj, lo3[2] + kk];
                let pairs = [
                    (
                        *s3.fields.cons.den.view().at(c3),
                        *s2.fields.cons.den.view().at(c2),
                        "den",
                    ),
                    (
                        *s3.fields.cons.mom[0].view().at(c3),
                        *s2.fields.cons.mom[0].view().at(c2),
                        "mom0",
                    ),
                    (
                        *s3.fields.cons.mom[1].view().at(c3),
                        *s2.fields.cons.mom[1].view().at(c2),
                        "mom1",
                    ),
                    (
                        *mhd3.bface[0].view().at(c3),
                        *mhd2.bface[0].view().at(c2),
                        "bx",
                    ),
                    (
                        *mhd3.bface[1].view().at(c3),
                        *mhd2.bface[1].view().at(c2),
                        "by",
                    ),
                ];
                for (a, b, what) in pairs {
                    assert!(
                        a.to_bits() == b.to_bits(),
                        "{what} at ({ii},{jj},{kk}): 3d {a:e} vs 2.5d {b:e}"
                    );
                }
                let (mz, bz) = (
                    *s3.fields.cons.mom[2].view().at(c3),
                    *mhd3.bface[2].view().at(c3),
                );
                assert!(
                    mz == 0.0 && bz == 0.0,
                    "out-of-plane leaked at ({ii},{jj},{kk}): mz={mz:e} bz={bz:e}"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, NX * NY * NZ);
}
