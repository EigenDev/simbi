// =============================================================================
// imhd_divb_under_evolve.rs
//
// end-to-end validation of the isothermal-mhd substrate (Mignone 2007): run the full
// no-energy KernelSet (c2p -> ghost_fill -> flux per dir -> cfl -> snapshot -> godunov
// -> post_godunov[CT] -> rk2) for ~10 steps with periodic BCs and assert
//   (a) the discrete staggered div(B) stays at machine zero (the shared CT stack is
//       correctly wired through the energy-optional mhd_substrate path, and the
//       iso `bcell_from_bface` — interpolation only, no 1/2|B|^2 correction — works), and
//   (b) the state stays physical (rho > 0, finite) — the trivial iso c2p cannot fail.
//
// IC: Orszag-Tang vortex, analytically div-free B, isothermal closure (cs = 1).
// =============================================================================

use std::f64::consts::PI;
use symbi_hydro::quantity::Density;

use symbi::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet3D;
use symbi::regimes::substrate_kernels::Solver;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::Isothermal;
use symbi_hydro::isothermal_mhd::IsothermalMhd;
use symbi_hydro::mhd_state::MhdPrimG;
use symbi_hydro::state::PrimG;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<IsothermalMhd, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;

const NX: usize = 8;
const NY: usize = 8;
const NZ: usize = 1;
const CS: f64 = 1.0;
const CFL: f64 = 0.3;
const V0: f64 = 0.5;
const B0: f64 = 1.0;
const RHO0: f64 = 1.0;
const T_FINAL: f64 = 0.5;
const DIVB_TOL: f64 = 1e-12;

fn make_sim() -> Sim {
    let dx = 1.0 / NX as f64;
    let dy = 1.0 / NY as f64;
    let dz = 1.0 / NZ as f64;

    // analytically div-free staggered B (seed_faces reads face_coord — face-midpoint sampling).
    // iso primitive: no energy slot (pre is zst). set_initial seeds the conserved (den, mom =
    // rho*v) from the prim and the cell-centered B from prim mag.
    Sim::build(IsothermalMhd, Isothermal { cs: CS }, Cartesian)
        .cells([NX, NY, NZ])
        .spacing([dx, dy, dz])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .expect("iso-MHD sim construction failed")
        .set_initial(|[x, y, _z]| {
            let vx = -V0 * (2.0 * PI * y).sin();
            let vy = V0 * (2.0 * PI * x).sin();
            let mag = Tensor::new([-B0 * (2.0 * PI * y).sin(), B0 * (4.0 * PI * x).sin(), 0.0]);
            MhdPrimG::<f64, 3, IsoModel>::new(
                PrimG::isothermal(Density(RHO0), Tensor::new([vx, vy, 0.0])),
                mag,
            )
        })
        .seed_faces(|axis, [x, y, _z]| match axis {
            0 => -B0 * (2.0 * PI * y).sin(),
            1 => B0 * (4.0 * PI * x).sin(),
            _ => 0.0,
        })
        .build()
}

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

fn run_solver(solver: Solver) {
    let mut sim = make_sim();
    let (idx, idy, idz) = (NX as f64, NY as f64, NZ as f64);

    let (div0, b0_max, _) = max_divb_and_b(&sim, idx, idy, idz);
    assert!(
        div0 / b0_max.max(1.0) < 1e-13,
        "iso OT IC is not divergence-free: max|divB|={div0:e}",
    );

    let sub = IsothermalMhdSubstrateKernelSet3D::<HostMemory, f64>::new(
        CS,
        CFL,
        1.0,
        &sim.geom.allocated,
    )
    .with_solver(solver)
    .expect("valid solver/regime pair");

    let mut max_seen_rel = 0.0_f64;
    let mut steps_seen: u64 = 0;
    evolve_with_callback(&mut sim, &sub, T_FINAL, 1, |s| {
        let (max_div, max_b, worst) = max_divb_and_b(s, idx, idy, idz);
        let rel = max_div / max_b.max(1.0);
        assert!(
            rel < DIVB_TOL,
            "DIVB GREW UNDER EVOLVE at iter {} t={:.4e} cell {:?}: max|divB|={:e} rel={:e} — CT broken",
            s.iteration, s.time, worst, max_div, rel,
        );
        if rel > max_seen_rel { max_seen_rel = rel; }
        steps_seen = s.iteration;
    }).expect("iso-MHD evolve failed");

    assert!(
        steps_seen >= 5,
        "iso evolve produced only {steps_seen} steps — gate barely exercised"
    );

    // physicality: the trivial iso c2p is rho = den; every interior cell must stay positive.
    for c in sim.geom.interior.iter() {
        let rho = *sim.fields.cons.den.view().at(c);
        assert!(rho.is_finite() && rho > 0.0, "cell {c:?}: rho = {rho}");
    }

    eprintln!(
        "[imhd_divb] {:?} DONE iter={} t={:.4e} max rel divB seen = {:e} (tol {:e})",
        solver, sim.iteration, sim.time, max_seen_rel, DIVB_TOL,
    );
}

#[test]
fn imhd_orszag_tang_preserves_divb_and_stays_physical() {
    // the full no-energy substrate pipeline under both Riemann solvers: HLLE and the
    // 3-state Mignone HLLD. divB must hold to machine precision through the shared CT.
    run_solver(Solver::Hlle);
    run_solver(Solver::Hlld);
}
