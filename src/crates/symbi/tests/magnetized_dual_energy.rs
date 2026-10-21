// =============================================================================
// magnetized_dual_energy.rs
//
// the entropy companion of the newtonian adiabatic MHD energy, S = rho K with
// K = p rho^(-gamma), on the 2.5D cartesian grid:
// - drain: the magnetized drain scales den, mom and the gas energy by f, so the
//   conservative pressure takes p' = f p and the entropy S' = f^(2-gamma) S,
//   K' = f^(1-gamma) K, each written out here from the cell's state;
// - transport: through a stage whose first-order redo splices the mass flux,
//   every cell's outgoing-flux weight stays nonnegative and K stays inside its
//   upwind stencil's range; on an advected entropy wave the energy path holds
//   pressure equilibrium and the entropy path, upwinding the cell value of K,
//   does not (first-order transport, recorded);
// - shocks: a planar Noh problem, cold gas streaming into the midplane, against
//   the exact Rankine-Hugoniot state behind the two shocks, unmagnetized and
//   with a transverse field at upstream beta 8e-6; the cold upstream runs on the
//   entropy, the shocked gas on the energy; without the shock detector the
//   correction budget halts the run;
// - the frozen E5 stage: the 13 x 13 block of low_beta_stage_replay.rs with the
//   entropy. the center cell sits on a drained sink edge, compressing, across a
//   steep pressure gradient: the shock detector fires and the cell, its
//   conservative energy inadmissible, stays flagged for the ladder.
//
// run: cargo test -p symbi --test magnetized_dual_energy
// =============================================================================

use std::ops::ControlFlow;
use symbi::regimes::dual_energy::DualEnergy;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_sim::dual_energy_ledger;
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kern = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;

const GAMMA: f64 = 5.0 / 3.0;

/// the conservative pressure, entropy and density of cell `c`
fn cell(s: &Sim, c: [isize; 2]) -> (f64, f64, f64) {
    let cons = &s.fields.cons;
    let rho = *cons.den.at(c);
    let kin: f64 = 0.5 * (0..3).map(|k| cons.mom[k].at(c).powi(2)).sum::<f64>() / rho;
    let b = &s.fields.mhd.as_ref().expect("mhd").bcell;
    let mag: f64 = 0.5 * (0..3).map(|k| b[k].at(c).powi(2)).sum::<f64>();
    let p = (GAMMA - 1.0) * (*cons.nrg.as_ref().expect("nrg").at(c) - kin - mag);
    (p, *cons.chi_field().expect("entropy").at(c), rho)
}

fn build(
    n: [usize; 2],
    lo: [f64; 2],
    dx: f64,
    walls: Boundaries<2>,
    state: impl Fn([f64; 2]) -> ([f64; 4], [f64; 3]) + Copy + 'static,
) -> Sim {
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(n)
        .origin(lo)
        .spacing([dx, dx])
        .boundaries(walls)
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(move |x| {
            let ([rho, vx, vy, p], b) = state(x);
            MhdPrim::new(Prim::adiabatic(Density(rho), Tensor::new([vx, vy, 0.0]), Pressure(p)), Tensor::new(b))
        })
        .seed_faces(move |d, x| state(x).1[d])
        .build()
        .with_passive_scalar()
        .expect("entropy slot")
}

fn kernels(s: &Sim, cfg: DualEnergy) -> Kern {
    let k = Kern::new(GAMMA, 0.3, 1.0, &s.geom.allocated).with_dual_energy(cfg);
    k.seed_entropy(s);
    k
}

#[test]
fn the_drain_scales_pressure_by_f_and_entropy_by_f_to_the_two_minus_gamma() {
    let s = build([16, 16], [-1.0, -1.0], 0.125, Boundaries::uniform(BoundaryType::Outflow), |_| {
        ([1.0, 0.1, -0.2, 0.3], [0.5, 0.3, 0.2])
    });
    let sink = Body::black_hole(0, Tensor::new([0.0, 0.0]), Tensor::zeros(), 50.0, 0.3, 0.1, 0.0, 1.0, 0.3)
        .with_surface(SurfaceSpec::Drain);
    let s = s.with_bodies(BodyCollection::new().add(sink));
    let k = kernels(&s, DualEnergy::new(GAMMA));
    k.c2p(&s);
    let cells: Vec<[isize; 2]> = s.geom.interior.iter().collect();
    let before: Vec<(f64, f64, f64)> = cells.iter().map(|&c| cell(&s, c)).collect();
    k.penalize(&s, 0.02);
    let (mut checked, mut worst, mut deepest) = (0, 0.0f64, 1.0f64);
    for (&c, &(p0, s0, rho0)) in cells.iter().zip(&before) {
        let (p1, s1, rho1) = cell(&s, c);
        let f = rho1 / rho0;
        if f == 1.0 {
            continue;
        }
        checked += 1;
        deepest = deepest.min(f);
        let k0 = s0 / rho0;
        let k1 = s1 / rho1;
        worst = worst
            .max((p1 / (f * p0) - 1.0).abs())
            .max((s1 / (f.powf(2.0 - GAMMA) * s0) - 1.0).abs())
            .max((k1 / (f.powf(1.0 - GAMMA) * k0) - 1.0).abs());
    }
    assert!(checked >= 4, "the drain acted on {checked} cells; the gate is vacuous");
    // f^(2-gamma) and f differ by f^(1-gamma) - 1, which is what an S' = f S drain would miss
    assert!(deepest.powf(1.0 - GAMMA) - 1.0 > 0.1, "the deepest drain f = {deepest} cannot tell f^(2-gamma) from f");
    // each ratio rounds the drain factor, the power and the energy differences once (measured 2.2e-16)
    assert!(worst < 1e-14, "the drained cell departs from p' = f p, S' = f^(2-gamma) S, K' = f^(1-gamma) K by {worst:e}");
}

/// a planar Noh run: gas at density 1 and pressure 1e-6 streaming at unit speed into x = 0 with a
/// transverse field b, to time t; returns the cells' (x, rho, p, B_y), the run's accepted
/// corrections and its halt, if it stopped short of t
fn noh(b: f64, nx: usize, t: f64, cfg: DualEnergy) -> (Vec<(f64, f64, f64, f64)>, dual_energy_ledger::CorrectionTotals, Option<String>) {
    let dx = 2.0 / nx as f64;
    let walls = Boundaries([[BoundaryType::Outflow; 2], [BoundaryType::Periodic; 2]]);
    let s = build([nx, 4], [-1.0, 0.0], dx, walls, move |[x, _]| {
        ([1.0, if x < 0.0 { 1.0 } else { -1.0 }, 0.0, 1e-6], [0.0, b, 0.0])
    });
    let k = kernels(&s, cfg);
    let scope = dual_energy_ledger::open_scope();
    let mut hier = Hierarchy::single(s, k);
    hier.evolve_with_callback(t, 1_000_000, |_| ControlFlow::Continue(())).expect("noh run");
    let halt = hier.crash.as_ref().and_then(|c| c.halt.as_ref()).map(|h| format!("{h:?}"));
    let st = &hier.levels[0].state;
    let lo = st.geom.interior.spaces[1].lo;
    let row: Vec<(f64, f64, f64, f64)> = st
        .geom
        .interior
        .iter()
        .filter(|c| c[1] == lo)
        .map(|c| {
            let (p, _, rho) = cell(st, c);
            let x = -1.0 + (c[0] - st.geom.interior.spaces[0].lo) as f64 * dx + 0.5 * dx;
            (x, rho, p, *st.fields.mhd.as_ref().unwrap().bcell[1].at(c))
        })
        .collect();
    (row, scope.accepted(), halt)
}

/// the state behind a planar shock running into gas at density 1, pressure p1 and transverse field
/// b1 that streams in at unit speed, the shocked gas at rest: (compression, pressure, field, shock
/// speed), from mass, momentum, flux and energy conservation across the shock
fn rankine_hugoniot(p1: f64, b1: f64) -> (f64, f64, f64, f64) {
    let g = GAMMA;
    let state = |r: f64| {
        let s = 1.0 / (r - 1.0);
        let (w1, w2) = (1.0 + s, s);
        let b2 = b1 * r;
        let p2 = w1 * w1 + p1 + 0.5 * b1 * b1 - r * w2 * w2 - 0.5 * b2 * b2;
        let energy = (0.5 * w1 * w1 + g / (g - 1.0) * p1 + b1 * b1) * w1 - (0.5 * r * w2 * w2 + g / (g - 1.0) * p2 + b2 * b2) * w2;
        (energy, p2, b2, s)
    };
    let (mut lo, mut hi) = (1.0 + 1e-9, (g + 1.0) / (g - 1.0) - 1e-12);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if state(mid).0.signum() == state(lo).0.signum() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let r = 0.5 * (lo + hi);
    let (_, p2, b2, s) = state(r);
    (r, p2, b2, s)
}

/// the mean of (rho, p, B_y) over the shocked plateau: between the wall-heated cells at the center
/// and the shocks at |x| = s t, a margin of a tenth of the plateau on each side
fn plateau(row: &[(f64, f64, f64, f64)], front: f64) -> (f64, f64, f64, usize) {
    let (inner, outer) = (0.4 * front, 0.85 * front);
    let cells: Vec<_> = row.iter().filter(|(x, ..)| x.abs() > inner && x.abs() < outer).collect();
    let n = cells.len() as f64;
    let mean = |f: &dyn Fn(&(f64, f64, f64, f64)) -> f64| cells.iter().map(|c| f(c)).sum::<f64>() / n;
    (mean(&|c| c.1), mean(&|c| c.2), mean(&|c| c.3), cells.len())
}

/// the default switch against the exact state behind the shocks, and the switch with its shock
/// detector removed
fn noh_against_rankine_hugoniot(b: f64) {
    let t = 0.6;
    let (r, p2, b2, s) = rankine_hugoniot(1e-6, b);
    let (row, ledger, halt) = noh(b, 400, t, DualEnergy::new(GAMMA));
    assert!(halt.is_none(), "b = {b}: the run halted: {halt:?}");
    let (rho, p, by, n) = plateau(&row, s * t);
    assert!(n >= 20, "the plateau holds {n} cells; the comparison is vacuous");
    // premises: the cold upstream ran on the entropy and the detector held cells on the energy
    assert!(ledger.entropy_cells > 0, "b = {b}: no cell ran on the entropy; the cold upstream never reached the switch");
    assert!(ledger.shock_cells > 0, "b = {b}: the shock detector never held a cell on the energy");
    assert_eq!(ledger.invalid_shock_cells, 0, "b = {b}: a shocked cell carried an inadmissible energy");
    let errors = [(rho / r - 1.0).abs(), (p / p2 - 1.0).abs(), if b > 0.0 { (by / b2 - 1.0).abs() } else { 0.0 }];
    println!("b = {b}: plateau rho {rho:.5} / {r:.5}, p {p:.5} / {p2:.5}, B_y {by:.5} / {b2:.5}; corrections {:+.2e} {:+.2e}; entropy cells {}, shock {}", ledger.added, ledger.removed, ledger.entropy_cells, ledger.shock_cells);
    // the plateau of the energy-only scheme sits within 1.0e-4 (b = 0) and 4.8e-5 (b = 0.5) of the
    // exact state at 400 cells, the wall-heated center and the smeared shocks excluded; the default
    // switch measures the same
    assert!(errors.iter().all(|e| *e < 1e-3), "b = {b}: the plateau departs from rankine-hugoniot by {errors:?}");
    // without the detector the shocked cells run on the entropy, lose the shock's heating, and the
    // correction budget halts the run
    let (_, _, halt) = noh(b, 400, t, DualEnergy { eta_2: f64::INFINITY, ..DualEnergy::new(GAMMA) });
    let halt = halt.unwrap_or_default();
    assert!(halt.contains("correction budget exceeded"), "b = {b}: without the detector the run did not halt on the budget: {halt:?}");
}

#[test]
fn planar_noh_matches_rankine_hugoniot_unmagnetized() {
    noh_against_rankine_hugoniot(0.0);
}

#[test]
fn planar_noh_matches_rankine_hugoniot_at_low_beta() {
    noh_against_rankine_hugoniot(0.5);
}

mod frozen_e5 {
    use super::*;

    include!("fixtures/low_beta_stage_entry.rs");

    /// the E5 stage-entry block loaded into a 2.5D store with the entropy seeded from its energy
    fn loaded() -> (Sim, Kern, [isize; 2]) {
        let s = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
            .cells([N, N])
            .spacing([1.0 / 256.0; 2])
            .origin([0.0; 2])
            .boundaries(Boundaries::uniform(BoundaryType::Outflow))
            .cfl(0.4)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("sim")
            .set_initial(|_| MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)), Tensor::new([0.4, 0.1, 0.1])))
            .seed_faces_uniform([0.4, 0.1])
            .build()
            .with_passive_scalar()
            .expect("entropy slot");
        let lo = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo];
        let at = |i: usize, j: usize| [lo[0] + i as isize, lo[1] + j as isize];
        let cons = &s.fields.cons;
        let prim = &s.fields.prim;
        let m = s.fields.mhd.as_ref().expect("mhd");
        let cells: Vec<(&symbi_grid::Field<f64, 2, HostMemory>, &[[f64; N]; N])> = vec![
            (&cons.den, &ENTRY_DEN),
            (&cons.mom[0], &ENTRY_MOM0),
            (&cons.mom[1], &ENTRY_MOM1),
            (&cons.mom[2], &ENTRY_MOM2),
            (cons.nrg.as_ref().expect("nrg"), &ENTRY_NRG),
            (&prim.rho, &ENTRY_RHO),
            (&prim.vel[0], &ENTRY_VEL0),
            (&prim.vel[1], &ENTRY_VEL1),
            (&prim.vel[2], &ENTRY_VEL2),
            (prim.pre.as_ref().expect("pre"), &ENTRY_PRE),
            (&m.bcell[0], &ENTRY_BCELL0),
            (&m.bcell[1], &ENTRY_BCELL1),
            (&m.bcell[2], &ENTRY_BCELL2),
        ];
        for (field, values) in cells {
            for j in 0..N {
                for i in 0..N {
                    field.view_mut().set(at(i, j), values[j][i]);
                }
            }
        }
        for j in 0..N {
            for i in 0..=N {
                m.bface[0].view_mut().set(at(i, j), ENTRY_BFACE0[j][i]);
            }
        }
        for j in 0..=N {
            for i in 0..N {
                m.bface[1].view_mut().set(at(i, j), ENTRY_BFACE1[j][i]);
            }
        }
        let k = Kern::new(GAMMA, 0.4, 1.0, &s.geom.allocated).with_dual_energy(DualEnergy::new(GAMMA));
        k.ghost_fill(&s);
        k.seed_entropy(&s);
        (s, k, at(N / 2, N / 2))
    }

    /// the E5 stage with the entropy: the center cell's conservative internal energy turns
    /// negative (as in low_beta_stage_replay.rs); the synchronization either gives it the entropy
    /// pressure or, where the shock detector fires, leaves it flagged for the correction ladder
    #[test]
    fn the_frozen_e5_stage_with_the_entropy() {
        let (mut s, k, center) = loaded();
        let scope = dual_energy_ledger::open_scope();
        k.snapshot(&s);
        s.mark_stage_input_captured();
        k.snapshot_stage(&s);
        s.dt = DT;
        for d in 0..2 {
            k.flux(&s, d);
        }
        k.efield(&s);
        k.godunov_stage(&s, DT, 0.0, 1.0);
        k.stage_emf(&s, DT, 1);
        k.stage_curl(&s, DT, 1);
        // premise: before the synchronization the center cell is the production E5 state
        let m = s.fields.mhd.as_ref().expect("mhd");
        let cons = &s.fields.cons;
        let slots = [
            *cons.den.at(center),
            *cons.mom[0].at(center),
            *cons.mom[1].at(center),
            *cons.mom[2].at(center),
            *cons.nrg.as_ref().expect("nrg").at(center),
            *m.bcell[0].at(center),
            *m.bcell[1].at(center),
            *m.bcell[2].at(center),
        ];
        for kk in 0..8 {
            assert_eq!(slots[kk].to_bits(), PRODUCTION_AFTER_CURL[kk].to_bits(), "slot {kk} departs from the production E5 stage");
        }
        let (p_cons, entropy, rho) = cell(&s, center);
        let p_entropy = entropy * rho.powf(GAMMA - 1.0);
        k.c2p(&s);
        let (p_after, _, _) = cell(&s, center);
        let stage = dual_energy_ledger::pending();
        let status = *s.fields.c2p_error.at(center);
        println!(
            "center: conservative p {p_cons:.6e}, entropy p {p_entropy:.6e}, after sync p {p_after:.6e}, status {status}; stage: entropy cells {}, shock {}, invalid shock {}, added {:.3e}, removed {:.3e}, worst relative {:.3e}",
            stage.entropy_cells, stage.shock_cells, stage.invalid_shock_cells, stage.added, stage.removed, stage.worst_relative
        );
        dual_energy_ledger::step_discard();
        drop(scope);
        assert!(p_cons < 0.0, "the center cell's conservative pressure is {p_cons:e}; the stage no longer reproduces E5");
        // the synchronization's contract at an inadmissible cell: on the entropy path it takes the
        // entropy pressure and clears its status; inside a detected shock it is left untouched,
        // still flagged, for the correction ladder
        if status == 0.0 {
            assert_eq!(p_after.to_bits(), p_entropy.to_bits(), "the cleared center cell does not carry the entropy pressure");
        } else {
            assert!(stage.invalid_shock_cells >= 1, "the flagged center cell was not booked as an invalid shock");
            assert_eq!(p_after.to_bits(), p_cons.to_bits(), "the synchronization altered a cell it left to the ladder");
        }
    }
}

/// a supersonic, weakly magnetized periodic flow (Mach 3 to 8) with a hundredfold density contrast
/// and a tenfold temperature contrast in steep tanh profiles: the flow speed sets the timestep, so
/// the outgoing mass fluxes approach a cell's mass. a cell on the density front flagged troubled in
/// the corrector of step 3 sends that stage through the first-order redo
#[test]
fn entropy_transport_stays_convex_through_a_spliced_redo() {
    use std::f64::consts::TAU;
    let n = 48;
    let s = build([n, n], [0.0, 0.0], 1.0 / n as f64, Boundaries::uniform(BoundaryType::Periodic), |[x, y]| {
        let front = 0.5 * (1.0 + (20.0 * ((TAU * x).sin() * (TAU * y).cos())).tanh());
        let rho = 0.01 + 0.99 * front;
        let p = rho * 1e-2 * (1.0 + 9.0 * (1.0 - front));
        ([rho, -(TAU * x).sin(), 0.6 * (TAU * y).cos(), p], [0.01, 0.005, 0.002])
    });
    // the transport alone: every cell resynchronizes from its energy after each stage, so no
    // entropy-path correction enters and the stage update is the only thing the gate reads. a
    // cfl of 0.45 per axis brings the outgoing fluxes closer to the cell's mass than the run's 0.3
    let k = Kern::new(GAMMA, 0.45, 1.0, &s.geom.allocated).with_dual_energy(DualEnergy { eta_1: 0.0, ..DualEnergy::new(GAMMA) });
    k.seed_entropy(&s);
    let lo = s.geom.interior.spaces[0].lo;
    // the front crosses the row y = 0 near x = 1/2, where sin changes sign
    let flagged = [lo + (n / 2) as isize, lo];
    *s.workspace.troubled_once.lock().unwrap() = Some((3, 1, flagged, false));
    let scope = dual_energy_ledger::open_scope();
    let mut hier = Hierarchy::single(s, k);
    let steps = 8;
    let mut taken = 0;
    let run = hier
        .evolve_with_callback(10.0, 1, |_| {
            taken += 1;
            if taken >= steps { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
        })
        .expect("run");
    let halt = hier.crash.as_ref().and_then(|c| c.halt.as_ref()).map(|h| format!("{h:?}"));
    assert!(halt.is_none(), "the run halted: {halt:?}");
    assert!(run.guards.troubled_cells.total >= 1, "the flagged cell booked no troubled census; the redo never ran");
    let stats = hier.levels[0].kernels.entropy_transport();
    // two stages per step and one redone stage
    assert_eq!(stats.stages, 2 * steps as u64 + 1, "the entropy advanced {} times over {steps} RK2 steps with one redo", stats.stages);
    println!("min weight {:e}, max excursion {:e}, corrections {:?}", stats.min_weight, stats.max_excursion, scope.accepted());
    assert!(stats.min_weight > 0.0, "a cell's outgoing mass flux reached its mass (w / rho = {:e})", stats.min_weight);
    assert!(stats.max_excursion <= 1e-12, "K left its stencil's range by {:e}", stats.max_excursion);
}

/// an entropy wave, rho = 1 + 0.5 sin(2 pi (x + y)) at uniform pressure, velocity (1, 1) and field,
/// advected one period on the periodic unit square at `n` cells a side; the L1 error of the
/// pressure and of the density against the initial profile, with every cell on the entropy path
/// when `entropy` and on the energy path otherwise
fn advected(n: usize, entropy: bool) -> (f64, f64) {
    use std::f64::consts::TAU;
    let wave = |[x, y]: [f64; 2]| ([1.0 + 0.5 * (TAU * (x + y)).sin(), 1.0, 1.0, 1.0], [0.3, 0.2, 0.1]);
    let s = build([n, n], [0.0, 0.0], 1.0 / n as f64, Boundaries::uniform(BoundaryType::Periodic), wave);
    let cfg = if entropy { DualEnergy { eta_1: f64::INFINITY, budget_global: f64::INFINITY, budget_local: f64::INFINITY, relative_max: f64::INFINITY, ..DualEnergy::new(GAMMA) } } else { DualEnergy { eta_1: 0.0, ..DualEnergy::new(GAMMA) } };
    let k = kernels(&s, cfg);
    let scope = dual_energy_ledger::open_scope();
    let mut hier = Hierarchy::single(s, k);
    hier.evolve_with_callback(1.0, 1_000_000, |_| ControlFlow::Continue(())).expect("run");
    assert!(hier.crash.is_none(), "the advection halted: {:?}", hier.crash.as_ref().and_then(|c| c.halt.as_ref()));
    let entropy_cells = scope.accepted().entropy_cells;
    assert_eq!(entropy_cells > 0, entropy, "{entropy_cells} entropy-path cells on the {} path", if entropy { "entropy" } else { "energy" });
    let st = &hier.levels[0].state;
    let lo = [st.geom.interior.spaces[0].lo, st.geom.interior.spaces[1].lo];
    let h = 1.0 / n as f64;
    let (mut ep, mut er) = (0.0, 0.0);
    for c in st.geom.interior.iter() {
        let x = [(c[0] - lo[0]) as f64 * h + 0.5 * h, (c[1] - lo[1]) as f64 * h + 0.5 * h];
        let (p, _, rho) = cell(st, c);
        ep += (p - 1.0).abs() * h * h;
        er += (rho - wave(x).0[0]).abs() * h * h;
    }
    (ep, er)
}

/// the energy path carries the wave at uniform pressure to round-off. the entropy path upwinds the
/// cell value of K, first order in space beside the gas's second-order density, so p_S = K rho^(gamma-1)
/// leaves pressure equilibrium at the wave: measured L1 pressure errors 7.6e-2, 5.9e-2, 3.9e-2 at
/// 32, 64, 128 cells, and density errors 4 to 11 times the energy path's. recorded here as the
/// transport's accuracy; the gate asserts the energy path's equilibrium and that the entropy path's
/// errors fall with resolution
#[test]
fn the_entropy_path_advects_an_entropy_wave() {
    for entropy in [false, true] {
        let errs: Vec<(f64, f64)> = [32, 64, 128].iter().map(|&n| advected(n, entropy)).collect();
        if entropy {
            assert!(errs.windows(2).all(|w| w[1].0 < w[0].0 && w[1].1 < w[0].1), "the entropy path's errors do not fall with resolution: {errs:?}");
        } else {
            assert!(errs.iter().all(|e| e.0 < 1e-14), "the energy path left pressure equilibrium: {errs:?}");
        }
        println!(
            "{}: p {:.3e} {:.3e} {:.3e} (ratios {:.2} {:.2}); rho {:.3e} {:.3e} {:.3e} (ratios {:.2} {:.2})",
            if entropy { "entropy path" } else { "energy path" },
            errs[0].0, errs[1].0, errs[2].0, errs[0].0 / errs[1].0, errs[1].0 / errs[2].0,
            errs[0].1, errs[1].1, errs[2].1, errs[0].1 / errs[1].1, errs[1].1 / errs[2].1
        );
    }
}

/// the conserved state and entropy over the interior, as bits
fn state_bits(s: &Sim) -> Vec<u64> {
    let cons = &s.fields.cons;
    s.geom
        .interior
        .iter()
        .flat_map(|c| {
            let mut v = vec![cons.den.at(c).to_bits(), cons.nrg.as_ref().unwrap().at(c).to_bits(), cons.chi_field().unwrap().at(c).to_bits()];
            v.extend((0..3).map(|k| cons.mom[k].at(c).to_bits()));
            v
        })
        .collect()
}

/// the unmagnetized Noh run without the shock detector: its shocked cells lose the shock's heating
/// to the entropy until a stage's corrections exceed the local budget, after accepted steps. the
/// halted run holds the state of its last accepted step bit for bit, and the ledger holds the
/// accepted steps' corrections with nothing of the rejected attempt
#[test]
fn a_budget_halt_restores_the_step_entry_and_books_nothing_of_the_attempt() {
    let nx = 100;
    let walls = Boundaries([[BoundaryType::Outflow; 2], [BoundaryType::Periodic; 2]]);
    let s = build([nx, 4], [-1.0, 0.0], 2.0 / nx as f64, walls, |[x, _]| {
        ([1.0, if x < 0.0 { 1.0 } else { -1.0 }, 0.0, 1e-6], [0.0, 0.0, 0.0])
    });
    let k = kernels(&s, DualEnergy { eta_2: f64::INFINITY, ..DualEnergy::new(GAMMA) });
    let scope = dual_energy_ledger::open_scope();
    let mut hier = Hierarchy::single(s, k);
    let mut last: Option<(Vec<u64>, dual_energy_ledger::CorrectionTotals, f64)> = None;
    hier.evolve_with_callback(10.0, 1, |h| {
        last = Some((state_bits(&h.levels[0].state), dual_energy_ledger::accepted(), h.levels[0].state.time));
        ControlFlow::Continue(())
    })
    .expect("run");
    let halt = hier.crash.as_ref().and_then(|c| c.halt.as_ref()).map(|h| format!("{h:?}")).unwrap_or_default();
    assert!(halt.contains("correction budget exceeded"), "the run did not halt on the budget: {halt:?}");
    let (bits, ledger, time) = last.expect("no state observed");
    assert!(time > 0.0 && ledger.stages > 0, "the run halted in its first step; the restore of an accepted state is unobserved");
    let st = &hier.levels[0].state;
    assert_eq!(st.time.to_bits(), time.to_bits(), "the halted clock is not the last accepted step's");
    assert!(state_bits(st) == bits, "the halted state differs from the last accepted step's");
    assert_eq!(scope.accepted(), ledger, "the ledger booked corrections of the rejected attempt");
    assert_eq!(dual_energy_ledger::pending(), Default::default(), "the rejected attempt's corrections are still pending");
    println!("halted at t = {time:e}: {halt}; accepted {ledger:?}");
}
