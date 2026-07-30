// =============================================================================
// nmhd_2p5d_divb_under_evolve.rs
//
// validation of the 2.5D Newtonian-MHD substrate: run
// the ENTIRE production NMHD KernelSet on a GENUINE D=2 grid (DOF=3) — c2p ->
// ghost_fill -> flux per dir -> cfl -> snapshot -> godunov_euler -> post_godunov[CT]
// -> rk2 — for ~10 steps with periodic BCs and assert
//   (a) the discrete staggered in-plane div(B) = dBx/dx + dBy/dy stays at machine
//       zero (the 2.5D CT: single corner E_z evolving the face-staggered Bx,By), and
//   (b) the state stays PHYSICAL (rho>0, p>0, finite), and
//   (c) the out-of-plane Bz (carried CELL-CENTERED, NO face, evolved by the ordinary
//       induction-flux divergence — no CT) actually EVOLVES from its IC (so the 2.5D
//       out-of-plane path is genuinely exercised).
//
// IC: Orszag-Tang vortex with an added non-uniform out-of-plane Bz. in 2.5D d/dz=0,
// so div B only constrains the in-plane field (Bz is divergence-free for free) — the
// single corner E_z CT must hold div B to machine epsilon while Bz advects.
// =============================================================================

use std::f64::consts::PI;

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::{NewtonianMhd, nmhd_recover};
use symbi_hydro::state::{Cons, Prim};
use symbi_xpu::{CpuSpace, HostMemory};

// the genuine 2.5D MHD sim: D=2 spatial axes, DOF=3 vector components.
type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const NX: usize = 16;
const NY: usize = 16;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const V0: f64 = 0.5;
const B0: f64 = 1.0;
const BZ0: f64 = 0.4;
const T_FINAL: f64 = 0.3;
const DIVB_TOL: f64 = 1e-12;

fn make_sim() -> Sim {
    let dx = 1.0 / NX as f64;
    let dy = 1.0 / NY as f64;
    let rho0 = GAMMA * GAMMA;
    let p0 = GAMMA;

    // analytically div-free staggered in-plane B (two face fields only — no Bz face).
    // set_initial seeds cons + cell-centered B (incl. the non-uniform out-of-plane Bz,
    // cell-centered, div-free for d/dz=0) from the primitive.
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([NX, NY])
        .spacing([dx, dy])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .expect("nmhd 2.5d sim construction failed")
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
        .seed_faces(|axis, [x, y]| match axis {
            0 => -B0 * (2.0 * PI * y).sin(),
            _ => B0 * (4.0 * PI * x).sin(),
        })
        .build()
}

// in-plane staggered div(B) = dBx/dx + dBy/dy (Bz carries no face; d/dz=0 in 2.5D).
fn max_divb_and_b(sim: &Sim, idx: f64, idy: f64) -> (f64, f64, [isize; 2]) {
    let mhd = sim.fields.mhd.as_ref().expect("mhd");
    let mut max_div = 0.0_f64;
    let mut max_b = 0.0_f64;
    let mut worst = [0_isize; 2];
    for c in sim.geom.interior.iter() {
        let bx_lo = *mhd.bface[0].view().at(c);
        let bx_hi = *mhd.bface[0].view().at([c[0] + 1, c[1]]);
        let by_lo = *mhd.bface[1].view().at(c);
        let by_hi = *mhd.bface[1].view().at([c[0], c[1] + 1]);
        let div = (bx_hi - bx_lo) * idx + (by_hi - by_lo) * idy;
        if div.abs() > max_div {
            max_div = div.abs();
            worst = c;
        }
        let bz = *mhd.bcell[2].view().at(c);
        let b_mag = (bx_lo * bx_lo + by_lo * by_lo + bz * bz).sqrt();
        if b_mag > max_b {
            max_b = b_mag;
        }
    }
    (max_div, max_b, worst)
}

// the maximum cell-by-cell change in the out-of-plane Bz vs an analytic IC sample —
// confirms the cell-centered induction-flux divergence actually evolves Bz.
fn max_bz_change(sim: &Sim) -> f64 {
    let mhd = sim.fields.mhd.as_ref().expect("mhd");
    let dx = 1.0 / NX as f64;
    let mut max_dz = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let x = (c[0] as f64 + 0.5) * dx;
        let bz_ic = BZ0 * (2.0 * PI * x).cos();
        let bz = *mhd.bcell[2].view().at(c);
        max_dz = max_dz.max((bz - bz_ic).abs());
    }
    max_dz
}

#[test]
fn nmhd_2p5d_orszag_tang_preserves_divb_evolves_bz() {
    let mut sim = make_sim();
    let idx = NX as f64;
    let idy = NY as f64;

    let (div0, b0_max, _) = max_divb_and_b(&sim, idx, idy);
    assert!(
        div0 / b0_max.max(1.0) < 1e-13,
        "2.5D ORSZAG-TANG IC is not divergence-free: max|divB|={:e} (rel {:e})",
        div0,
        div0 / b0_max.max(1.0),
    );

    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
        GAMMA,
        CFL,
        /* theta */ 1.0,
        &sim.geom.allocated,
    );

    let mut max_seen_rel = 0.0_f64;
    let mut steps_seen: u64 = 0;
    evolve_with_callback(&mut sim, &sub, T_FINAL, 1, |s| {
        let (max_div, max_b, worst) = max_divb_and_b(s, idx, idy);
        let rel = max_div / max_b.max(1.0);
        assert!(
            rel < DIVB_TOL,
            "2.5D DIVB GREW UNDER EVOLVE at iter {} t={:.4e} cell {:?}: \
             max|divB|={:e}  max|B|={:e}  rel={:e}  (tol {:e}) — 2.5D CT is broken",
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
    .expect("nmhd 2.5d evolve failed");

    assert!(
        steps_seen >= 5,
        "2.5D evolve produced only {steps_seen} steps — gate barely exercised"
    );

    // PHYSICALITY: recover prims from the evolved conserved state (DOF=3 cons).
    let eos = IdealGas { gamma: GAMMA };
    let mhd = sim.fields.mhd.as_ref().expect("mhd");
    for c in sim.geom.interior.iter() {
        let cnrg = sim.fields.cons.nrg_field().expect("cons.nrg");
        let cons = symbi_hydro::mhd_state::MhdCons::<f64, 3> {
            hydro: Cons {
                chi: Default::default(),
                den: *sim.fields.cons.den.view().at(c),
                mom: Tensor::new([
                    *sim.fields.cons.mom[0].view().at(c),
                    *sim.fields.cons.mom[1].view().at(c),
                    *sim.fields.cons.mom[2].view().at(c),
                ]),
                nrg: *cnrg.view().at(c),
            },
            mag: Tensor::new([
                *mhd.bcell[0].view().at(c),
                *mhd.bcell[1].view().at(c),
                *mhd.bcell[2].view().at(c),
            ]),
        };
        let prim = nmhd_recover(&eos, &cons);
        assert!(
            prim.rho.is_finite() && prim.rho > 0.0,
            "cell {c:?}: rho = {}",
            prim.rho
        );
        assert!(
            prim.pre.is_finite() && prim.pre > 0.0,
            "cell {c:?}: p = {}",
            prim.pre
        );
    }

    // the out-of-plane Bz must have evolved (cell-centered induction-flux divergence).
    let bz_change = max_bz_change(&sim);
    assert!(
        bz_change > 1e-6,
        "out-of-plane Bz did not evolve (max change {bz_change:e}) — the 2.5D cell-centered \
         induction path is not running",
    );

    eprintln!(
        "[nmhd_2p5d] DONE iter={} t={:.4e} max rel divB = {:e} (tol {:e}) max |dBz| = {:e}",
        sim.iteration, sim.time, max_seen_rel, DIVB_TOL, bz_change,
    );
}
