// =============================================================================
// nmhd_divb_under_evolve.rs
//
// validation of the Newtonian-MHD substrate: run the ENTIRE
// production NMHD KernelSet (c2p -> ghost_fill -> flux per dir -> cfl -> snapshot ->
// godunov_euler -> post_godunov[CT] -> rk2) for ~10 steps with periodic BCs and assert
//   (a) the discrete staggered div(B) stays at machine zero (the CT stack — reused
//       from RMHD — is correctly wired for the Newtonian regime), and
//   (b) the state stays PHYSICAL (rho > 0, p > 0, finite) — the algebraic NMHD c2p
//       cannot fail the way RMHD's iterative inversion does.
//
// IC: Orszag-Tang vortex — analytically div-free B (Bx = -B0\cdot sin(2\pi y), By =
// B0\cdot sin(4\pi x), Bz = 0) on an 8^3 grid (nz=1 extruded 2D), matching the RMHD divB gate.
// the shared CT path is regime-agnostic (Faraday + the 1/2|B|^2 Newtonian magnetic-
// energy correction), so div(B) must hold to machine epsilon exactly as for RMHD.
// =============================================================================

use std::f64::consts::PI;

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet3D;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::{nmhd_recover, NewtonianMhd};
use symbi_hydro::state::{Cons, Prim};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<NewtonianMhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const NX: usize = 8;
const NY: usize = 8;
const NZ: usize = 1;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const V0: f64 = 0.5;
const B0: f64 = 1.0;
const T_FINAL: f64 = 0.5;
const DIVB_TOL: f64 = 1e-12;

fn make_sim() -> Sim {
    let dx = 1.0 / NX as f64;
    let dy = 1.0 / NY as f64;
    let dz = 1.0 / NZ as f64;
    let rho0 = GAMMA * GAMMA;
    let p0 = GAMMA;

    // analytically div-free staggered B (same as the RMHD OT gate). seed_faces reads the
    // staggered face_coord; set_initial reads the cell center.
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([NX, NY, NZ])
        .spacing([dx, dy, dz])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .expect("nmhd sim construction failed")
        .set_initial(|[x, y, _z]| {
            let vx = -V0 * (2.0 * PI * y).sin();
            let vy =  V0 * (2.0 * PI * x).sin();
            let bx_c = -B0 * (2.0 * PI * y).sin();
            let by_c =  B0 * (4.0 * PI * x).sin();
            MhdPrim {
                hydro: Prim { rho: rho0, vel: Tensor::new([vx, vy, 0.0]), pre: p0 },
                mag: Tensor::new([bx_c, by_c, 0.0]),
            }
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
        if b_mag > max_b { max_b = b_mag; }
    }
    (max_div, max_b, worst)
}

#[test]
fn nmhd_orszag_tang_preserves_divb_and_stays_physical() {
    let mut sim = make_sim();
    let idx = NX as f64;
    let idy = NY as f64;
    let idz = NZ as f64;

    let (div0, b0_max, _) = max_divb_and_b(&sim, idx, idy, idz);
    assert!(
        div0 / b0_max.max(1.0) < 1e-13,
        "ORSZAG-TANG IC is not divergence-free: max|divB|={:e} (rel {:e})",
        div0, div0 / b0_max.max(1.0),
    );

    let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, CFL, /* theta */ 1.0, &sim.geom.allocated);

    let mut max_seen_rel = 0.0_f64;
    let mut steps_seen: u64 = 0;
    evolve_with_callback(
        &mut sim,
        &sub,
        T_FINAL,
        1,
        |s| {
            let (max_div, max_b, worst) = max_divb_and_b(s, idx, idy, idz);
            let rel = max_div / max_b.max(1.0);
            assert!(
                rel < DIVB_TOL,
                "DIVB GREW UNDER EVOLVE at iter {} t={:.4e} cell {:?}: \
                 max|divB|={:e}  max|B|={:e}  rel={:e}  (tol {:e}) — CT operator is broken",
                s.iteration, s.time, worst, max_div, max_b, rel, DIVB_TOL,
            );
            if rel > max_seen_rel { max_seen_rel = rel; }
            steps_seen = s.iteration;
        },
    ).expect("nmhd evolve failed");

    assert!(
        steps_seen >= 5,
        "NMHD evolve produced only {} steps before t_final — gate barely exercised",
        steps_seen,
    );

    // PHYSICALITY: recover the primitives from the evolved conserved state via the
    // algebraic NMHD c2p; every interior cell must be physical (rho>0, p>0, finite).
    let eos = IdealGas { gamma: GAMMA };
    let mhd = sim.fields.mhd.as_ref().expect("mhd");
    for c in sim.geom.interior.iter() {
        let cnrg = sim.fields.cons.nrg_field().expect("cons.nrg");
        let cons = symbi_hydro::mhd_state::MhdCons::<f64, 3> {
            hydro: Cons {
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
        assert!(prim.rho.is_finite() && prim.rho > 0.0, "cell {c:?}: rho = {}", prim.rho);
        assert!(prim.pre.is_finite() && prim.pre > 0.0, "cell {c:?}: p = {}", prim.pre);
    }

    eprintln!(
        "[nmhd_divb] DONE iter={} t={:.4e} max rel divB seen = {:e} (tol {:e})",
        sim.iteration, sim.time, max_seen_rel, DIVB_TOL,
    );
}
