// =============================================================================
// rmhd_divb_under_evolve.rs
//
// the full-evolve divergence-cleaning gate. rmhd_ct_curl_divb (in
// symbi-discretize) covers the CT operator in isolation — one step, one EMF.
// this test runs the entire production RMHD substrate (c2p -> ghost_fill -> flux
// per dir -> cfl -> snapshot -> godunov_euler -> post_godunov[CT] -> rk2) for
// STEPS_TARGET steps with periodic BCs and asserts the discrete staggered
// divergence stays at machine zero.
//
// IC: Orszag-Tang vortex — analytically div-free B (Bx = -B0\cdot sin(2\pi y), By =
// B0\cdot sin(4\pi x), Bz = 0) on a small 8^3 grid (nz=1 logical 2D extruded one cell).
// face-staggered storage matches examples/rmhd_orszag_tang.rs; the discrete div on
// the CT mesh is identically zero at t=0 by telescoping, and CT preserves that
// to machine epsilon every step.
//
// assertion (every step, every interior cell):
//   max_divB = max_c | (bface[0][c+ex] - bface[0][c]) / dx
//                    + (bface[1][c+ey] - bface[1][c]) / dy
//                    + (bface[2][c+ez] - bface[2][c]) / dz |
//   max_divB / max(|B|, 1) < 1e-12
//
// the 1e-12 slop (not 1e-15) absorbs ~50 steps of FP roundoff in the curl
// accumulation. a real CT regression — wrong stencil, broken edge-EMF wiring,
// view-struct sign flip — produces O(1) divB and trips the gate immediately.
// =============================================================================

use std::f64::consts::PI;
use symbi_hydro::quantity::{Density, Pressure};

use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Rmhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const NX: usize = 8;
const NY: usize = 8;
const NZ: usize = 1;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const V0: f64 = 0.5;
const B0: f64 = 1.0;
// t_final sized to span \gtrsim 10 cfl-bounded steps on dx=1/8, \lambda_max ~1 (dt \approx 0.0375).
// stays well before the OT turbulent regime (\gtrsim 0.5/cs ~ 0.75 in the example), so
// the divB invariant is exercised under the smooth-and-developing flow phase.
const T_FINAL: f64 = 0.5;
const DIVB_TOL: f64 = 1e-12;

fn make_sim() -> Sim {
    let dx = 1.0 / NX as f64;
    let dy = 1.0 / NY as f64;
    let dz = 1.0 / NZ as f64;
    let rho0 = GAMMA * GAMMA;
    let p0 = GAMMA;

    // analytically div-free staggered B. seed_faces reads face_coord (on the d-face,
    // cell-centered transverse) — no hand-written half-cell offset.
    //   Bx on x-faces: Bx = -B0\cdot sin(2\pi y),  By on y-faces: By = B0\cdot sin(4\pi x),  Bz = 0
    // cell-centered B from analytic eval at cell centers (consistent with bface); hydro:
    // v from the OT velocity field.
    Sim::build(Rmhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([NX, NY, NZ])
        .spacing([dx, dy, dz])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .expect("rmhd sim construction failed")
        .set_initial(|[x, y, _z]| {
            let vx = -V0 * (2.0 * PI * y).sin();
            let vy = V0 * (2.0 * PI * x).sin();
            let bx_c = -B0 * (2.0 * PI * y).sin();
            let by_c = B0 * (4.0 * PI * x).sin();
            MhdPrim::new(
                Prim::adiabatic(Density(rho0), Tensor::new([vx, vy, 0.0]), Pressure(p0)),
                Tensor::new([bx_c, by_c, 0.0]),
            )
        })
        .seed_faces(|axis, [x, y, _z]| match axis {
            0 => -B0 * (2.0 * PI * y).sin(),
            1 => B0 * (4.0 * PI * x).sin(),
            _ => 0.0,
        })
        .build()
}

// staggered divergence + amplitude over the interior, taken from bface.
fn max_divb_and_b(sim: &Sim, idx: f64, idy: f64, idz: f64) -> (f64, f64, [isize; 3]) {
    let mhd = sim.fields.mhd.as_ref().expect("mhd");
    let mut max_div = 0.0_f64;
    let mut max_b = 0.0_f64;
    let mut worst = [0_isize; 3];
    for c in sim.geom.interior.iter() {
        let bx_lo = *mhd.bface[0].view().at(c);
        let bx_hi = *mhd.bface[0].view().at([c[0] + 1, c[1], c[2]]);
        let by_lo = *mhd.bface[1].view().at(c);
        let by_hi = *mhd.bface[1].view().at([c[0], c[1] + 1, c[2]]);
        let bz_lo = *mhd.bface[2].view().at(c);
        let bz_hi = *mhd.bface[2].view().at([c[0], c[1], c[2] + 1]);
        let div = (bx_hi - bx_lo) * idx + (by_hi - by_lo) * idy + (bz_hi - bz_lo) * idz;
        if div.abs() > max_div {
            max_div = div.abs();
            worst = c;
        }
        let b_mag = (bx_lo * bx_lo + by_lo * by_lo + bz_lo * bz_lo).sqrt();
        if b_mag > max_b {
            max_b = b_mag;
        }
    }
    (max_div, max_b, worst)
}

#[test]
fn rmhd_orszag_tang_preserves_divb_under_full_evolve() {
    let mut sim = make_sim();
    let idx = NX as f64;
    let idy = NY as f64;
    let idz = NZ as f64;

    // sanity: the analytic IC is divergence-free to machine precision on the
    // staggered mesh before any kernel runs.
    let (div0, b0_max, _) = max_divb_and_b(&sim, idx, idy, idz);
    assert!(
        div0 / b0_max.max(1.0) < 1e-13,
        "ORSZAG-TANG IC is not divergence-free: max|divB|={:e} (rel {:e})",
        div0,
        div0 / b0_max.max(1.0),
    );

    let sub = RmhdSubstrateKernelSet3D::<HostMemory, f64>::new(
        GAMMA,
        CFL,
        /* theta */ 1.0,
        &sim.geom.allocated,
    );

    // per-step divB monitoring through the production loop (interval = 1).
    let mut max_seen_rel = 0.0_f64;
    let mut steps_seen: u64 = 0;
    evolve_with_callback(&mut sim, &sub, T_FINAL, 1, |s| {
        let (max_div, max_b, worst) = max_divb_and_b(s, idx, idy, idz);
        let rel = max_div / max_b.max(1.0);
        assert!(
            rel < DIVB_TOL,
            "DIVB GREW UNDER EVOLVE at iter {} t={:.4e} cell {:?}: \
                 max|divB|={:e}  max|B|={:e}  rel={:e}  (tol {:e}) — CT operator is broken",
            s.iteration,
            s.time,
            worst,
            max_div,
            max_b,
            rel,
            DIVB_TOL,
        );
        if rel > max_seen_rel {
            max_seen_rel = rel;
        }
        steps_seen = s.iteration;
    })
    .expect("rmhd evolve failed");

    assert!(
        steps_seen >= 5,
        "RMHD evolve produced only {} steps before t_final — divB gate barely exercised",
        steps_seen,
    );
    eprintln!(
        "[rmhd_divb] DONE iter={} t={:.4e} max rel divB seen = {:e} (tol {:e})",
        sim.iteration, sim.time, max_seen_rel, DIVB_TOL,
    );
}
