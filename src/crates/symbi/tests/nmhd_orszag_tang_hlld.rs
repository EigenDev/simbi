// =============================================================================
// nmhd_orszag_tang_hlld.rs
//
// the Newtonian-MHD HLLD payoff: the Orszag-Tang vortex evolved with the full
// 5-wave HLLD solver (`Solver::Hlld`) through the production substrate, into the
// developing-turbulence regime where current sheets form. the win is robustness:
// the algebraic NMHD c2p cannot fail the way RMHD's iterative inversion does in
// those sheets, and the closed-form HLLD resolves the contact/alfven structure
// HLLE smears. the run must stay physical (rho>0, p>0, finite) and div(B)-clean
// every step — that is the "OT-with-HLLD is stable" proof.
// =============================================================================

use std::f64::consts::PI;

use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet3D;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<NewtonianMhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const NX: usize = 32;
const NY: usize = 32;
const NZ: usize = 1;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const T_FINAL: f64 = 0.5; // well into the OT vortex roll-up + current-sheet formation
const DIVB_TOL: f64 = 1e-11;

fn make_sim() -> Sim {
    let dx = 1.0 / NX as f64;
    let dy = 1.0 / NY as f64;
    let dz = 1.0 / NZ as f64;
    // canonical Orszag-Tang: rho = gamma^2, p = gamma, v = (-sin 2pi y, sin 2pi x, 0),
    // B = (-sin 2pi y, sin 4pi x, 0)/sqrt(4pi). use B0 = 1/sqrt(4pi).
    let rho0 = GAMMA * GAMMA;
    let p0 = GAMMA;
    let b0 = 1.0 / (4.0 * PI).sqrt();

    // canonical Orszag-Tang: point-sampled cell-centered prim + div-free staggered B.
    // Bx on x-faces = -b0 sin(2pi y), By on y-faces = b0 sin(4pi x). seed_faces reads the
    // staggered face_coord (cell-centered transverse) — no half-cell typo.
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([NX, NY, NZ])
        .spacing([dx, dy, dz])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .expect("nmhd OT sim construction failed")
        .set_initial(|[x, y, _z]| {
            let vx = -(2.0 * PI * y).sin();
            let vy = (2.0 * PI * x).sin();
            let bx = -b0 * (2.0 * PI * y).sin();
            let by = b0 * (4.0 * PI * x).sin();
            MhdPrim {
                hydro: Prim {
                    rho: rho0,
                    vel: Tensor::new([vx, vy, 0.0]),
                    pre: p0,
                },
                mag: Tensor::new([bx, by, 0.0]),
            }
        })
        .seed_faces(|axis, [x, y, _z]| match axis {
            0 => -b0 * (2.0 * PI * y).sin(),
            1 => b0 * (4.0 * PI * x).sin(),
            _ => 0.0,
        })
        .build()
}

fn max_divb(sim: &Sim, idx: f64, idy: f64, idz: f64) -> (f64, f64) {
    let mhd = sim.fields.mhd.as_ref().expect("mhd");
    let (mut max_div, mut max_b) = (0.0_f64, 0.0_f64);
    for c in sim.geom.interior.iter() {
        let bx_lo = *mhd.bface[0].view().at(c);
        let bx_hi = *mhd.bface[0].view().at([c[0] + 1, c[1], c[2]]);
        let by_lo = *mhd.bface[1].view().at(c);
        let by_hi = *mhd.bface[1].view().at([c[0], c[1] + 1, c[2]]);
        let bz_lo = *mhd.bface[2].view().at(c);
        let bz_hi = *mhd.bface[2].view().at([c[0], c[1], c[2] + 1]);
        let div = (bx_hi - bx_lo) * idx + (by_hi - by_lo) * idy + (bz_hi - bz_lo) * idz;
        max_div = max_div.max(div.abs());
        max_b = max_b.max((bx_lo * bx_lo + by_lo * by_lo + bz_lo * bz_lo).sqrt());
    }
    (max_div, max_b)
}

#[test]
fn nmhd_orszag_tang_hlld_stays_physical_and_divb_clean() {
    let mut sim = make_sim();
    let (idx, idy, idz) = (NX as f64, NY as f64, NZ as f64);

    // the HLLD payoff: select the full 5-wave solver.
    let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory>::new(
        GAMMA,
        CFL,
        /* theta */ 1.5,
        &sim.geom.allocated,
    )
    .with_solver(Solver::Hlld)
    .expect("valid solver/regime pair");
    assert_eq!(sub.solver, Solver::Hlld, "must run with HLLD");

    let mut max_rel_divb = 0.0_f64;
    evolve_with_callback(&mut sim, &sub, T_FINAL, 1, |s| {
        // physical at every step — the algebraic c2p + HLLD robustness in the sheets.
        let pre = s.fields.prim.pre_field().expect("prim.pre");
        for c in s.geom.interior.iter() {
            let rho = *s.fields.prim.rho.view().at(c);
            let p = *pre.view().at(c);
            assert!(
                rho.is_finite() && rho > 0.0,
                "iter {}: rho={rho} at {c:?}",
                s.iteration
            );
            assert!(
                p.is_finite() && p > 0.0,
                "iter {}: p={p} at {c:?}",
                s.iteration
            );
        }
        let (md, mb) = max_divb(s, idx, idy, idz);
        let rel = md / mb.max(1.0);
        assert!(
            rel < DIVB_TOL,
            "iter {}: div(B) grew to rel {rel:e} (tol {DIVB_TOL:e})",
            s.iteration
        );
        max_rel_divb = max_rel_divb.max(rel);
    })
    .expect("nmhd OT HLLD evolution failed");

    assert!(
        sim.iteration >= 20,
        "OT-HLLD took only {} steps — barely exercised",
        sim.iteration
    );
    eprintln!(
        "[nmhd OT HLLD] {} steps to t={:.3} on {NX}x{NY}, max rel div(B) = {:e}",
        sim.iteration, sim.time, max_rel_divb,
    );
}
