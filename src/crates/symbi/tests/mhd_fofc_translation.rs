// =============================================================================
// mhd_fofc_translation.rs
//
// the first-order flux correction of a magnetized flow on a periodic box is
// translation invariant: shifting the complete initial state by a whole number
// of cells shifts every evolved field by the same cells, bit for bit. the
// corrected edge EMF reads four face fluxes around the edge; on an edge of the
// grid's boundary two of them lie in the ghost column, and they must be the
// same spliced fluxes an interior edge would read, so an event on the periodic
// seam is corrected exactly as the same event in the middle of the grid.
//
// the fixture is the periodic magnetized relativistic shock: two cold streams
// (v = +0.95 / -0.94, p = 1e-4, uniform transverse By) collide on one plane and
// part on another half a box away; the collision drives the high-order
// recovery unphysical and the correction fires with the freeze tier silent. one
// arm puts the collision plane in the middle of the grid, the other on the
// periodic seam.
//
// every arm asserts its premise: the correction fired, in the corrector (or
// single) stage and, under RK2, in the predictor stage as well; the two arms
// booked the same number of corrections; their troubled flags are translates of
// each other; and no cell froze. then the translated fields are compared bit
// for bit and the discrete divergence is checked in both arms.
//
// run: cargo test -p symbi --test mhd_fofc_translation
// =============================================================================

use std::ops::ControlFlow;
use symbi::prelude::SimSubstrate;
use symbi::regimes::fofc::{fofc_reset_stats, fofc_stats};
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_sim::substrate_seam::Solver;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 4.0 / 3.0;
const B0Y: f64 = 0.2;
const NX: usize = 256;
const NY: usize = 8;
const DX: f64 = 1.0 / NX as f64;

// the correction counters are process-global: one arm at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

type Sim = SimStateGeneric<Rmhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

struct Arm {
    /// every compared field on the global grid, x index first: (name, nx, values)
    fields: Vec<(String, usize, Vec<f64>)>,
    /// the troubled flag at the end of every step, the last stage's decode
    flags: Vec<Vec<f64>>,
    fallbacks: u64,
    freezes: u64,
    max_div: f64,
}

/// the stream speed and end time of one arm, chosen so the correction fires in every stage
/// the arm covers while the freeze tier stays silent.
#[derive(Clone, Copy)]
struct Fixture {
    speed: f64,
    t_final: f64,
}

/// fires from step 5 (UCT) or 37 (Contact, forward Euler) with no freeze through t = 0.06.
const MODERATE: Fixture = Fixture {
    speed: 0.95,
    t_final: 0.06,
};
/// Contact under RK2 corrects in its predictor alone at the moderate speed; at 0.98 the
/// corrector stage corrects at step 49, and the freeze tier first acts after t = 0.02.
const FAST_AND_SHORT: Fixture = Fixture {
    speed: 0.98,
    t_final: 0.02,
};

/// run with the streams colliding on the low face of cell column `collide`.
fn run(collide: usize, ct: CtMethod, ts: Timestepping, fixture: Fixture) -> Arm {
    fofc_reset_stats();
    let sim = Sim::build(Rmhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([NX, NY])
        .origin([0.0, 0.0])
        .spacing([DX, 1.0 / NY as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.1)
        .timestepping(ts)
        .allocate()
        .expect("sim")
        .set_initial(move |[x, _y]| {
            // by cell column, so both arms build the same numbers in shifted places
            let column = (x / DX).floor() as usize;
            let behind = (column + NX - collide) % NX >= NX / 2;
            let vx = if behind { fixture.speed } else { -(fixture.speed - 0.01) };
            MhdPrim::new(
                Prim::adiabatic(Density(1.0), Tensor::new([vx, 0.0, 0.0]), Pressure(1e-4)),
                Tensor::new([0.0, B0Y, 0.0]),
            )
        })
        .seed_faces_uniform([0.0, B0Y])
        .build();
    let mut kset = sim
        .substrate()
        .with_solver(Solver::Hlld)
        .expect("solver")
        .ct_method(ct);
    kset.theta = 1.5;
    let mut hier = Hierarchy::single(sim, kset);
    let mut flags = Vec::new();
    hier.evolve_with_callback(fixture.t_final, 1, |h| {
        let s = &h.levels[0].state;
        let view = s.workspace.fofc_flag.view();
        let lo = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo];
        let mut grid = vec![0.0; NX * NY];
        for c in s.geom.interior.iter() {
            grid[(c[0] - lo[0]) as usize * NY + (c[1] - lo[1]) as usize] = *view.at(c);
        }
        flags.push(grid);
        ControlFlow::Continue(())
    })
    .expect("evolve");

    let s = &hier.levels[0].state;
    let lo = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo];
    let cells = |field: &symbi_grid::Field<f64, 2, HostMemory>| -> Vec<f64> {
        let view = field.view();
        let mut grid = vec![0.0; NX * NY];
        for c in s.geom.interior.iter() {
            grid[(c[0] - lo[0]) as usize * NY + (c[1] - lo[1]) as usize] = *view.at(c);
        }
        grid
    };
    let mhd = s.fields.mhd.as_ref().expect("mhd fields");
    // a periodic face field over the NX by NY faces that are distinct: the closing face on
    // its own axis repeats the opening one
    let faces = |d: usize| -> Vec<f64> {
        let view = mhd.bface.b[d].view();
        let mut grid = vec![0.0; NX * NY];
        for i in 0..NX {
            for j in 0..NY {
                grid[i * NY + j] = *view.at([lo[0] + i as isize, lo[1] + j as isize]);
            }
        }
        grid
    };
    let mut fields = vec![
        ("den".to_string(), NX, cells(&s.fields.cons.den)),
        ("nrg".to_string(), NX, cells(s.fields.cons.nrg_field().unwrap())),
        ("bface0".to_string(), NX, faces(0)),
        ("bface1".to_string(), NX, faces(1)),
    ];
    for k in 0..3 {
        fields.push((format!("mom{k}"), NX, cells(&s.fields.cons.mom[k])));
        fields.push((format!("bcell{k}"), NX, cells(&mhd.bcell.b[k])));
    }
    let mut max_div = 0.0f64;
    let (bx, by) = (mhd.bface.b[0].view(), mhd.bface.b[1].view());
    for c in s.geom.interior.iter() {
        let div = (*bx.at([c[0] + 1, c[1]]) - *bx.at(c)) / DX
            + (*by.at([c[0], c[1] + 1]) - *by.at(c)) * NY as f64;
        max_div = max_div.max(div.abs());
    }
    let (fallbacks, freezes) = fofc_stats();
    Arm {
        fields,
        flags,
        fallbacks,
        freezes,
        max_div,
    }
}

/// `grid` moved `shift` columns toward higher x on the periodic axis.
fn shifted(grid: &[f64], shift: usize) -> Vec<f64> {
    let mut out = vec![0.0; grid.len()];
    for i in 0..NX {
        let to = (i + shift) % NX;
        out[to * NY..(to + 1) * NY].copy_from_slice(&grid[i * NY..(i + 1) * NY]);
    }
    out
}

/// `interior` and `seam` are the collision columns of the two arms: the first keeps every
/// troubled cell off the grid's first and last columns, the second puts one there.
fn assert_translation_invariant(
    ct: CtMethod,
    ts: Timestepping,
    fixture: Fixture,
    interior: usize,
    seam: usize,
    label: &str,
) {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let shift = (interior + NX - seam) % NX;
    let middle = run(interior, ct, ts, fixture);
    let seam = run(seam, ct, ts, fixture);

    // premises
    assert!(middle.fallbacks > 0, "{label}: the correction never fired");
    assert_eq!((middle.freezes, seam.freezes), (0, 0), "{label}: the freeze tier acted");
    assert_eq!(middle.flags.len(), seam.flags.len(), "{label}: step counts differ");
    let last_stage: u64 = middle.flags.iter().flatten().map(|f| *f as u64).sum();
    assert!(last_stage > 0, "{label}: the last stage of a step never corrected");
    if ts == Timestepping::Rk2 {
        assert!(
            middle.fallbacks > last_stage,
            "{label}: the predictor stage never corrected ({} total, {last_stage} in correctors)",
            middle.fallbacks
        );
    }
    // the seam arm's events sit on the seam: a flagged cell in the first or last column
    let touches_seam = |arm: &Arm| {
        arm.flags
            .iter()
            .any(|g| g[..NY].iter().chain(&g[(NX - 1) * NY..]).any(|f| *f != 0.0))
    };
    assert!(touches_seam(&seam), "{label}: no troubled cell touched the periodic seam");
    assert!(
        !touches_seam(&middle),
        "{label}: the reference arm has a troubled cell on the seam as well"
    );
    for (step, (m, s)) in middle.flags.iter().zip(&seam.flags).enumerate() {
        assert!(
            shifted(s, shift) == *m,
            "{label}: the troubled flags of step {step} are not translates"
        );
    }

    for ((name, _, m), (_, _, s)) in middle.fields.iter().zip(&seam.fields) {
        let back = shifted(s, shift);
        let differing = back.iter().zip(m).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert_eq!(differing, 0, "{label}: {name} differs from its translate in {differing} places");
    }
    assert_eq!(middle.fallbacks, seam.fallbacks, "{label}: the arms booked different corrections");
    for (arm, which) in [(&middle, "middle"), (&seam, "seam")] {
        assert!(arm.max_div < 1e-11, "{label}: {which} arm div B {:e}", arm.max_div);
    }
}

// where the last stage's troubled columns fall relative to the collision column c:
//  forward Euler, Contact:    the parting plane, c - 128 ..= c - 118 and c + 117 ..= c + 127
//  forward Euler, UCT:        the same, and c - 5 ..= c - 2 and c + 1 ..= c + 5
//  UCT under RK2:             c - 5 ..= c - 2 and c + 1 ..= c + 5
//  Contact under RK2, fast:   c - 127 alone (the parting plane, at step 49)
// the seam arm's collision column is chosen to land one of them on column 0 or 255.

#[test]
fn contact_euler() {
    assert_translation_invariant(CtMethod::Contact, Timestepping::Euler, MODERATE, NX / 4, NX / 2, "contact, euler");
}

#[test]
fn contact_rk2() {
    assert_translation_invariant(CtMethod::Contact, Timestepping::Rk2, FAST_AND_SHORT, NX / 2, NX / 2 - 1, "contact, rk2");
}

#[test]
fn uct_euler() {
    assert_translation_invariant(CtMethod::Uct, Timestepping::Euler, MODERATE, NX / 4, 2, "uct, euler");
}

#[test]
fn uct_rk2() {
    assert_translation_invariant(CtMethod::Uct, Timestepping::Rk2, MODERATE, NX / 2, 2, "uct, rk2");
}
