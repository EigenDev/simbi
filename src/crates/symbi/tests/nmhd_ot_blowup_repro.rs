// diagnostic: reproduce the nmhd_orszag_tang example's IC + config and watch the
// interior density. prints min/max rho at t=0 (uniform gamma^2 for a correct IC)
// and over the first steps, localizing a density blow-up to the IC, to the first
// step, or to secular growth.

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

const NX: usize = 128;
const NY: usize = 128;
const GAMMA: f64 = 5.0 / 3.0;
const V0: f64 = 1.0; // canonical Athena Newtonian OT

// (rho_min, rho_max, pre_min) over the interior.
fn rho_stats(sim: &Sim) -> (f64, f64, f64) {
    let (mut mn, mut mx, mut pmin) = (f64::MAX, f64::MIN, f64::MAX);
    let pre = sim.fields.prim.pre_field().unwrap();
    for c in sim.geom.interior.iter() {
        let r = *sim.fields.prim.rho.view().at(c);
        mn = mn.min(r);
        mx = mx.max(r);
        pmin = pmin.min(*pre.view().at(c));
    }
    (mn, mx, pmin)
}

fn make_sim() -> Sim {
    let dx = 1.0 / NX as f64;
    let dy = 1.0 / NY as f64;
    // canonical Athena Newtonian OT: cs^2 = gamma p/rho = 1, B0 = 1/sqrt(4pi).
    let rho0 = 25.0 / (36.0 * PI);
    let p0 = 5.0 / (12.0 * PI);
    let b0 = 1.0 / (4.0 * PI).sqrt();
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([NX, NY, 1])
        .spacing([dx, dy, 1.0])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .allocate()
        .unwrap()
        .set_initial(|[x, y, _z]| {
            let vx = -V0 * (2.0 * PI * y).sin();
            let vy = V0 * (2.0 * PI * x).sin();
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

// run OT to t=0.3 with a given solver; track the WORST (min) pressure + max density
// seen, and the first negative-pressure step. returns (worst_pmin, max_rho, first_neg_iter).
fn run_solver(solver: Solver) -> (f64, f64, Option<u64>) {
    let mut sim = make_sim();
    let sub =
        NewtonianMhdSubstrateKernelSet3D::<HostMemory>::new(GAMMA, 0.4, 1.0, &sim.geom.allocated)
            .with_solver(solver)
            .expect("valid solver/regime pair");
    let (mut worst_p, mut max_r, mut first_neg) = (f64::MAX, f64::MIN, None);
    evolve_with_callback(&mut sim, &sub, 0.3, 1, |s| {
        let (_, mx, pmin) = rho_stats(s);
        worst_p = worst_p.min(pmin);
        max_r = max_r.max(mx);
        if pmin <= 0.0 && first_neg.is_none() {
            first_neg = Some(s.iteration);
        }
    })
    .ok();
    (worst_p, max_r, first_neg)
}

#[test]
#[ignore = "diagnostic: 128^2 x 3 solvers to t=0.3 (~3.5s); the per-solver positivity regression. \
            run on demand with --ignored. the fast OT regression is nmhd_divb_under_evolve (8^3)."]
fn nmhd_ot_solver_comparison() {
    eprintln!("[repro] OT {NX}x{NY} to t=0.3, canonical Athena IC — per-solver pressure/density:");
    let mut results = Vec::new();
    for (label, sv) in [
        ("HLLE", Solver::Hlle),
        ("HLLC", Solver::Hllc),
        ("HLLD", Solver::Hlld),
    ] {
        let (wp, mr, fn_) = run_solver(sv);
        eprintln!(
            "  {label}: worst pmin = {wp:.3e}  max rho = {mr:.3e}  first neg-p iter = {fn_:?}"
        );
        results.push((label, wp, mr));
    }
    // diagnostic: if only one solver goes negative-pressure, the bug is solver-specific.
    let healthy: Vec<&str> = results
        .iter()
        .filter(|(_, wp, _)| *wp > 0.0)
        .map(|(l, _, _)| *l)
        .collect();
    eprintln!("[repro] solvers that stayed positive-pressure: {healthy:?}");
}
