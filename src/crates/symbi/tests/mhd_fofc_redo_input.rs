// =============================================================================
// mhd_fofc_redo_input.rs
//
// the first-order redo of a magnetized stage reconstructs its fluxes on the
// normal B the high-order sweep read: the face field as it stood at stage
// entry, transverse ghosts included. by the time the redo runs, the stage's
// curl has advanced the live interior faces, so the redo binds its flux
// sweep to the stage-entry snapshot and never to the live faces.
//
// the gate runs one stage up to the troubled-cell mark, raises the flag on
// one physical interior cell, and then completes the stage three ways from
// identical twins:
//   clean     the stage as it stands
//   live      every live face, ghosts included, overwritten beforehand
//   snapshot  every snapshot face overwritten beforehand
// the redo's first-order gas and induction fluxes must be bit-for-bit equal
// between clean and live (the live faces are never read) and must differ
// between clean and snapshot (the snapshot is what is read). the premise is
// asserted: the stage's curl changed the live faces before the redo ran.
//
// covered: forward Euler, the RK2 predictor, and the RK2 corrector, where the
// stage-entry faces are the predictor's and differ from the step-entry base
// the constrained-transport update integrates from.
//
// run: cargo test -p symbi --test mhd_fofc_redo_input
// =============================================================================

use std::f64::consts::PI;
use symbi::prelude::SimSubstrate;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::stage::{StageArgs, fold_stage, fold_stage_from_correction, fold_stage_through_mark};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 16;
const GAMMA: f64 = 5.0 / 3.0;
const DT: f64 = 2.0e-3;
const TROUBLED: [isize; 2] = [7, 9];
const POISON: f64 = 1.0e3;

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Overwrite {
    Nothing,
    LiveFaces,
    SnapshotFaces,
}

fn build(ts: Timestepping) -> Sim {
    build_with(ts, BoundaryType::Periodic)
}

fn build_with(ts: Timestepping, x_walls: BoundaryType) -> Sim {
    let h = 1.0 / N as f64;
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .spacing([h, h])
        .boundaries(Boundaries([[x_walls; 2], [BoundaryType::Periodic; 2]]))
        .cfl(0.3)
        .timestepping(ts)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| {
            let vx = -0.5 * (2.0 * PI * y).sin();
            let vy = 0.5 * (2.0 * PI * x).sin();
            MhdPrim::new(
                Prim::adiabatic(Density(GAMMA * GAMMA), Tensor::new([vx, vy, 0.0]), Pressure(GAMMA)),
                Tensor::new([-(2.0 * PI * y).sin(), (4.0 * PI * x).sin(), 0.0]),
            )
        })
        // each face component depends on its transverse coordinate alone: divergence free
        .seed_faces(|axis, [x, y]| match axis {
            0 => -(2.0 * PI * y).sin(),
            _ => (4.0 * PI * x).sin(),
        })
        .build()
}

fn bits(field: &symbi_grid::Field<f64, 2, HostMemory>) -> Vec<u64> {
    let view = field.view();
    field.domain().iter().map(|c| view.at(c).to_bits()).collect()
}

/// the redo's spliced gas and induction fluxes of every direction, after completing stage
/// `stage` of `n_stages` with one injected troubled cell, plus whether the stage's curl had
/// changed the live faces by the time the redo ran.
fn redo_fluxes(ts: Timestepping, stage: usize, overwrite: Overwrite) -> (Vec<u64>, bool) {
    let mut sim = build(ts);
    let kernels = sim.substrate();
    let stages = ts.stages();
    let n_stages = stages.len();
    let args = |index: usize| StageArgs {
        dt: DT,
        a0: stages[index].0,
        ac: stages[index].1,
        stage: index,
        n_stages,
        injection_weight: 1.0,
        allow_elision: false,
    };
    sim.dt = DT;
    let sim = &*sim;
    kernels.c2p(sim);
    kernels.ghost_fill(sim);
    if n_stages > 1 {
        kernels.snapshot(sim);
    }
    for earlier in 0..stage {
        let _ = fold_stage(sim, &kernels, args(earlier), &mut |_| {});
    }

    let troubled = fold_stage_through_mark(sim, &kernels, args(stage), &mut |_| {});
    assert_eq!(troubled, 0, "the smooth state tripped the correction by itself");
    let mhd = sim.fields.mhd.as_ref().expect("mhd fields");
    let curled = (0..2).any(|d| bits(&mhd.bface.b[d]) != bits(&mhd.bface_stage.b[d]));

    *sim.workspace.fofc_flag.view_mut().at_mut(TROUBLED) = 1.0;
    let target = match overwrite {
        Overwrite::Nothing => None,
        Overwrite::LiveFaces => Some(&mhd.bface),
        Overwrite::SnapshotFaces => Some(&mhd.bface_stage),
    };
    if let Some(faces) = target {
        for d in 0..2 {
            let mut view = faces.b[d].view_mut();
            for c in faces.b[d].domain().iter() {
                *view.at_mut(c) += POISON;
            }
        }
    }
    let _ = fold_stage_from_correction(sim, &kernels, args(stage), false, &mut |_| {});

    let mut out = Vec::new();
    for d in 0..2 {
        out.extend(bits(&sim.fields.flux[d].den));
        out.extend(bits(sim.fields.flux[d].nrg_field().expect("energy flux")));
        for k in 0..3 {
            out.extend(bits(&sim.fields.flux[d].mom[k]));
            out.extend(bits(&mhd.bflux[d][k]));
        }
    }
    (out, curled)
}

fn assert_redo_reads_stage_entry_faces(ts: Timestepping, stage: usize, label: &str) {
    let (clean, curled) = redo_fluxes(ts, stage, Overwrite::Nothing);
    assert!(
        curled,
        "{label}: the live faces still equal the stage-entry snapshot when the redo runs; the \
         gate cannot tell them apart"
    );
    let (live, _) = redo_fluxes(ts, stage, Overwrite::LiveFaces);
    let (snapshot, _) = redo_fluxes(ts, stage, Overwrite::SnapshotFaces);
    let differing = |a: &[u64], b: &[u64]| a.iter().zip(b).filter(|(x, y)| x != y).count();
    assert_eq!(
        differing(&clean, &live),
        0,
        "{label}: overwriting the live faces changed the redo's fluxes; the sweep read them"
    );
    assert!(
        differing(&clean, &snapshot) > 0,
        "{label}: overwriting the stage-entry snapshot left the redo's fluxes unchanged; the \
         sweep does not read it"
    );
}

#[test]
fn forward_euler() {
    assert_redo_reads_stage_entry_faces(Timestepping::Euler, 0, "forward euler");
}

#[test]
fn rk2_predictor() {
    assert_redo_reads_stage_entry_faces(Timestepping::Rk2, 0, "rk2 predictor");
}

#[test]
fn rk2_corrector() {
    assert_redo_reads_stage_entry_faces(Timestepping::Rk2, 1, "rk2 corrector");
}

/// the induction-flux splice covers the flux sweep's own window: the interior faces and one
/// face beyond them on the transverse axis. the edge EMF of a boundary edge reads two
/// transverse faces in that outer column, and each carries the first-order flux when either
/// of its own two cells is flagged and the saved high-order flux otherwise, the rule of an
/// interior face. the flags of the outer column are the wall's images of the interior flags.
/// the troubled cell sits in the first interior column, against the low x wall.
fn assert_outer_column_is_spliced(x_walls: BoundaryType, label: &str) {
    let mut sim = build_with(Timestepping::Euler, x_walls);
    let kernels = sim.substrate();
    let args = StageArgs {
        dt: DT,
        a0: 0.0,
        ac: 1.0,
        stage: 0,
        n_stages: 1,
        injection_weight: 1.0,
        allow_elision: false,
    };
    sim.dt = DT;
    let sim = &*sim;
    kernels.c2p(sim);
    kernels.ghost_fill(sim);
    assert_eq!(fold_stage_through_mark(sim, &kernels, args, &mut |_| {}), 0);
    let lo = [sim.geom.interior.spaces[0].lo, sim.geom.interior.spaces[1].lo];
    let at_wall = [lo[0], lo[1] + 9];
    *sim.workspace.fofc_flag.view_mut().at_mut(at_wall) = 1.0;
    let _ = fold_stage_from_correction(sim, &kernels, args, false, &mut |_| {});

    let mhd = sim.fields.mhd.as_ref().expect("mhd fields");
    let flag = sim.workspace.fofc_flag.view();
    // the y faces of the outer column that holds the troubled cell's image, stored at the cell
    // above each face: beyond the low wall for a mirror or an outward copy, beyond the high
    // wall for the periodic wrap
    let column = if x_walls == BoundaryType::Periodic {
        lo[0] + N as isize
    } else {
        lo[0] - 1
    };
    let (mut first_order, mut high_order) = (0, 0);
    for j in lo[1]..=lo[1] + N as isize {
        let flagged = *flag.at([column, j]) != 0.0 || *flag.at([column, j - 1]) != 0.0;
        for k in 0..3 {
            let live = mhd.bflux[1][k].view().at([column, j]).to_bits();
            let saved = mhd.bflux_ho[1][k].view().at([column, j]).to_bits();
            if flagged {
                first_order += 1;
            } else {
                high_order += 1;
                assert_eq!(
                    live, saved,
                    "{label}: the unflagged outer face at row {j}, component {k}, lost its \
                     high-order induction flux"
                );
            }
        }
    }
    assert!(high_order > 0, "{label}: every outer face was flagged");
    assert!(first_order > 0, "{label}: the wall gave the outer column no flagged image");
}

#[test]
fn outer_column_spliced_at_a_periodic_wall() {
    assert_outer_column_is_spliced(BoundaryType::Periodic, "periodic");
}

#[test]
fn outer_column_spliced_at_an_outflow_wall() {
    assert_outer_column_is_spliced(BoundaryType::Outflow, "outflow");
}

#[test]
fn outer_column_spliced_at_a_reflecting_wall() {
    assert_outer_column_is_spliced(BoundaryType::Reflect, "reflect");
}

/// a correction is local: cells farther from the troubled cell than the redo's reach end the
/// stage with the conserved gas, the cell field, and the faces bit-for-bit equal to the same
/// stage run with no flag raised. every stage ends by interpolating the cell field from the
/// faces, the RK2 predictor included, so the redo re-interpolates on every stage; a redo that
/// left the cell field flux-predicted would change it wherever the field varies.
fn assert_correction_is_local(ts: Timestepping, stage: usize, label: &str) {
    let finish = |raise: bool| {
        let mut sim = build(ts);
        let kernels = sim.substrate();
        let stages = ts.stages();
        let args = |index: usize| StageArgs {
            dt: DT,
            a0: stages[index].0,
            ac: stages[index].1,
            stage: index,
            n_stages: stages.len(),
            injection_weight: 1.0,
            allow_elision: false,
        };
        sim.dt = DT;
        {
            let sim = &*sim;
            kernels.c2p(sim);
            kernels.ghost_fill(sim);
            if stages.len() > 1 {
                kernels.snapshot(sim);
            }
            for earlier in 0..stage {
                let _ = fold_stage(sim, &kernels, args(earlier), &mut |_| {});
            }
            assert_eq!(fold_stage_through_mark(sim, &kernels, args(stage), &mut |_| {}), 0);
            if raise {
                *sim.workspace.fofc_flag.view_mut().at_mut(TROUBLED) = 1.0;
            }
            let _ = fold_stage_from_correction(sim, &kernels, args(stage), false, &mut |_| {});
        }
        sim
    };
    let (clean, corrected) = (finish(false), finish(true));
    let far = |c: [isize; 2]| {
        let wrap = |d: isize| d.rem_euclid(N as isize).min((-d).rem_euclid(N as isize));
        wrap(c[0] - TROUBLED[0]).max(wrap(c[1] - TROUBLED[1])) > 4
    };
    let (a, b) = (&*clean, &*corrected);
    let (am, bm) = (a.fields.mhd.as_ref().unwrap(), b.fields.mhd.as_ref().unwrap());
    let mut pairs: Vec<(String, &symbi_grid::Field<f64, 2, HostMemory>, &symbi_grid::Field<f64, 2, HostMemory>)> = vec![
        ("den".into(), &a.fields.cons.den, &b.fields.cons.den),
        ("nrg".into(), a.fields.cons.nrg_field().unwrap(), b.fields.cons.nrg_field().unwrap()),
    ];
    for k in 0..3 {
        pairs.push((format!("bcell{k}"), &am.bcell.b[k], &bm.bcell.b[k]));
    }
    for d in 0..2 {
        pairs.push((format!("bface{d}"), &am.bface.b[d], &bm.bface.b[d]));
    }
    let mut near_changed = 0;
    for (name, x, y) in &pairs {
        let (xv, yv) = (x.view(), y.view());
        for c in a.geom.interior.iter() {
            let same = xv.at(c).to_bits() == yv.at(c).to_bits();
            if far(c) {
                assert!(same, "{label}: {name} changed at {c:?}, far from the troubled cell {TROUBLED:?}");
            } else if !same {
                near_changed += 1;
            }
        }
    }
    assert!(near_changed > 0, "{label}: the correction changed nothing near the troubled cell");
}

#[test]
fn a_correction_is_local_under_forward_euler() {
    assert_correction_is_local(Timestepping::Euler, 0, "forward euler");
}

#[test]
fn a_correction_is_local_in_the_rk2_predictor() {
    assert_correction_is_local(Timestepping::Rk2, 0, "rk2 predictor");
}

#[test]
fn a_correction_is_local_in_the_rk2_corrector() {
    assert_correction_is_local(Timestepping::Rk2, 1, "rk2 corrector");
}

// -----------------------------------------------------------------------------
// the RK2 constrained-transport update with corrections, against its formula:
//   B^*     = B^n - dt   curl(E_1)
//   B^{n+1} = B^n - dt/2 curl(E_1 + E_2)
// where E_k is the accepted EMF of stage k: first-order on an edge touching a flagged
// cell, high-order elsewhere. the accepted EMFs are obtained apart from the averaging
// under test:
//   E_1      read from the edge field when the predictor stage ends
//   E_2 high the corrector's stage-local EMF, from a twin run phase by phase up to the
//            edge-EMF phase, before the update averages it in place
//   E_2 low  on flagged edges, the raw first-order EMF of the corrector's stage input, from
//            a twin brought to the corrector's entry and folded there as a forward-Euler
//            stage: the redo of a single-stage scheme splices the Contact EMF it computes
//            with no average and no history. the first-order fluxes and the Contact EMF are
//            functions of the restored stage input alone, so the stage weights and the
//            timestep the twin uses leave them unchanged.
// -----------------------------------------------------------------------------

struct Rk2Step {
    faces_n: [Vec<f64>; 2],
    e1: Vec<f64>,
    faces_star: [Vec<f64>; 2],
    /// the edge field when the corrector ends: the EMF its single curl used
    e_effective: Vec<f64>,
    faces_new: [Vec<f64>; 2],
    /// edges touching the corrector's flagged cell
    corrector_flagged_edges: Vec<bool>,
}

/// interior faces of component `d` on the N by N periodic grid, x index first.
fn face_grid(sim: &Sim, d: usize) -> Vec<f64> {
    let view = sim.fields.mhd.as_ref().unwrap().bface.b[d].view();
    let lo = [sim.geom.interior.spaces[0].lo, sim.geom.interior.spaces[1].lo];
    (0..N * N)
        .map(|q| *view.at([lo[0] + (q / N) as isize, lo[1] + (q % N) as isize]))
        .collect()
}

/// the edge EMF at the low corner of every cell, x index first.
fn edge_grid(sim: &Sim, field: &symbi_grid::Field<f64, 2, HostMemory>) -> Vec<f64> {
    let view = field.view();
    let lo = [sim.geom.interior.spaces[0].lo, sim.geom.interior.spaces[1].lo];
    (0..N * N)
        .map(|q| *view.at([lo[0] + (q / N) as isize, lo[1] + (q % N) as isize]))
        .collect()
}

fn rk2_args(index: usize) -> StageArgs {
    let stages = Timestepping::Rk2.stages();
    StageArgs {
        dt: DT,
        a0: stages[index].0,
        ac: stages[index].1,
        stage: index,
        n_stages: 2,
        injection_weight: 1.0,
        allow_elision: false,
    }
}

/// one RK2 step with a flag raised in the chosen stages.
fn rk2_step(flag_predictor: bool, flag_corrector: bool) -> Rk2Step {
    let mut sim = build(Timestepping::Rk2);
    let kernels = sim.substrate();
    sim.dt = DT;
    let s = &*sim;
    kernels.c2p(s);
    kernels.ghost_fill(s);
    kernels.snapshot(s);
    let faces_n = [face_grid(&sim, 0), face_grid(&sim, 1)];
    let s = &*sim;

    assert_eq!(fold_stage_through_mark(s, &kernels, rk2_args(0), &mut |_| {}), 0);
    if flag_predictor {
        *s.workspace.fofc_flag.view_mut().at_mut(TROUBLED) = 1.0;
    }
    let _ = fold_stage_from_correction(s, &kernels, rk2_args(0), false, &mut |_| {});
    let mhd = s.fields.mhd.as_ref().unwrap();
    let e1 = edge_grid(&sim, &mhd.efield.e[0]);
    let faces_star = [face_grid(&sim, 0), face_grid(&sim, 1)];

    assert_eq!(fold_stage_through_mark(s, &kernels, rk2_args(1), &mut |_| {}), 0);
    let mut corrector_flagged_edges = vec![false; N * N];
    if flag_corrector {
        *s.workspace.fofc_flag.view_mut().at_mut(TROUBLED) = 1.0;
        let lo = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo];
        let cell = [(TROUBLED[0] - lo[0]) as usize, (TROUBLED[1] - lo[1]) as usize];
        // the four corner edges of the flagged cell
        for (di, dj) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            corrector_flagged_edges[((cell[0] + di) % N) * N + (cell[1] + dj) % N] = true;
        }
    }
    let _ = fold_stage_from_correction(s, &kernels, rk2_args(1), false, &mut |_| {});
    Rk2Step {
        faces_n,
        e1,
        faces_star,
        e_effective: edge_grid(&sim, &mhd.efield.e[0]),
        faces_new: [face_grid(&sim, 0), face_grid(&sim, 1)],
        corrector_flagged_edges,
    }
}

/// the corrector's stage-local high-order EMF, read before the update averages it: the
/// step is run to the corrector's entry and the corrector phase by phase up to the edge EMF.
fn corrector_high_order_emf(flag_predictor: bool) -> Vec<f64> {
    let mut sim = build(Timestepping::Rk2);
    let kernels = sim.substrate();
    sim.dt = DT;
    let s = &*sim;
    kernels.c2p(s);
    kernels.ghost_fill(s);
    kernels.snapshot(s);
    assert_eq!(fold_stage_through_mark(s, &kernels, rk2_args(0), &mut |_| {}), 0);
    if flag_predictor {
        *s.workspace.fofc_flag.view_mut().at_mut(TROUBLED) = 1.0;
    }
    let _ = fold_stage_from_correction(s, &kernels, rk2_args(0), false, &mut |_| {});
    kernels.wave_speeds(s);
    for d in 0..2 {
        kernels.flux(s, d);
    }
    kernels.efield(s);
    edge_grid(&sim, &s.fields.mhd.as_ref().unwrap().efield.e[0])
}

/// the raw first-order EMF of the corrector's stage input, with the corrector's flag raised:
/// the step is run to the corrector's entry and that stage is folded as a forward-Euler
/// stage, whose redo splices the Contact EMF as computed. meaningful on the flagged edges.
fn corrector_first_order_emf(flag_predictor: bool) -> Vec<f64> {
    let mut sim = build(Timestepping::Rk2);
    let kernels = sim.substrate();
    sim.dt = DT;
    let s = &*sim;
    kernels.c2p(s);
    kernels.ghost_fill(s);
    kernels.snapshot(s);
    assert_eq!(fold_stage_through_mark(s, &kernels, rk2_args(0), &mut |_| {}), 0);
    if flag_predictor {
        *s.workspace.fofc_flag.view_mut().at_mut(TROUBLED) = 1.0;
    }
    let _ = fold_stage_from_correction(s, &kernels, rk2_args(0), false, &mut |_| {});
    let single_stage = StageArgs {
        dt: DT,
        a0: 0.0,
        ac: 1.0,
        stage: 0,
        n_stages: 1,
        injection_weight: 1.0,
        allow_elision: false,
    };
    assert_eq!(fold_stage_through_mark(s, &kernels, single_stage, &mut |_| {}), 0);
    *s.workspace.fofc_flag.view_mut().at_mut(TROUBLED) = 1.0;
    let _ = fold_stage_from_correction(s, &kernels, single_stage, false, &mut |_| {});
    edge_grid(&sim, &s.fields.mhd.as_ref().unwrap().efield.e[0])
}

/// `B - dt curl(E)` on the periodic grid: Bx -= dt dE/dy, By += dt dE/dx, with the edge at
/// the low corner of its cell.
fn curled(faces: &[Vec<f64>; 2], e: &[f64], dt: f64) -> [Vec<f64>; 2] {
    let h = 1.0 / N as f64;
    let at = |i: usize, j: usize| e[(i % N) * N + j % N];
    let mut out = faces.clone();
    for i in 0..N {
        for j in 0..N {
            out[0][i * N + j] -= dt * (at(i, j + 1) - at(i, j)) / h;
            out[1][i * N + j] += dt * (at(i + 1, j) - at(i, j)) / h;
        }
    }
    out
}

fn max_abs_difference(a: &[Vec<f64>; 2], b: &[Vec<f64>; 2]) -> f64 {
    (0..2)
        .flat_map(|d| a[d].iter().zip(&b[d]).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}

/// the face comparison bound. the formula is evaluated here with its own operation order,
/// so it agrees with the kernels to rounding of O(1) face values: a few ulp, 1e-14. a lost
/// half weight or a stale history moves a face by dt times an EMF difference over h, which
/// the premise below requires to exceed 1e-9.
const FORMULA_ROUNDING: f64 = 1.0e-14;
const SIGNAL: f64 = 1.0e-9;

fn assert_rk2_formula(flag_predictor: bool, flag_corrector: bool, label: &str) {
    let clean = rk2_step(false, false);
    let step = rk2_step(flag_predictor, flag_corrector);
    let e2_high = corrector_high_order_emf(flag_predictor);
    let e2: Vec<f64> = if flag_corrector {
        let e2_low = corrector_first_order_emf(flag_predictor);
        // premise: on a flagged edge the first-order EMF differs from the high-order one, or
        // the two branches of the corrector's splice could not be told apart
        assert!(
            (0..N * N).any(|q| step.corrector_flagged_edges[q] && e2_low[q] != e2_high[q]),
            "{label}: the first-order and high-order corrector EMFs agree on every flagged edge"
        );
        (0..N * N)
            .map(|q| {
                if step.corrector_flagged_edges[q] {
                    e2_low[q]
                } else {
                    e2_high[q]
                }
            })
            .collect()
    } else {
        e2_high
    };

    // premises: the formula evaluator reproduces the clean step, and the correction changed
    // an EMF whose curl moves faces by more than the signal bound
    let clean_e2 = corrector_high_order_emf(false);
    let clean_sum: Vec<f64> = clean.e1.iter().zip(&clean_e2).map(|(a, b)| a + b).collect();
    let clean_formula = curled(&clean.faces_n, &clean_sum, 0.5 * DT);
    assert!(
        max_abs_difference(&clean_formula, &clean.faces_new) < FORMULA_ROUNDING,
        "{label}: the formula evaluator misses the uncorrected step by {:e}",
        max_abs_difference(&clean_formula, &clean.faces_new)
    );
    let moved = max_abs_difference(&step.faces_new, &clean.faces_new);
    assert!(
        moved > SIGNAL,
        "{label}: the correction moved the final faces by {moved:e}; nothing to discriminate"
    );
    if flag_predictor {
        let history = step.e1.iter().zip(&clean.e1).filter(|(a, b)| a != b).count();
        assert!(history > 0, "{label}: the predictor's accepted EMF equals the high-order one");
        assert!(
            max_abs_difference(&step.faces_star, &clean.faces_star) > SIGNAL,
            "{label}: the corrected predictor EMF has no curl to speak of"
        );
    }

    let star = curled(&step.faces_n, &step.e1, DT);
    assert!(
        max_abs_difference(&star, &step.faces_star) < FORMULA_ROUNDING,
        "{label}: B* misses B^n - dt curl(E_1) by {:e}",
        max_abs_difference(&star, &step.faces_star)
    );
    // the effective EMF is the half-weighted sum on every edge, flagged or not
    for q in 0..N * N {
        let want = 0.5 * (step.e1[q] + e2[q]);
        assert!(
            (step.e_effective[q] - want).abs() <= 4.0 * f64::EPSILON * want.abs().max(1.0),
            "{label}: edge {q} carries {:e}, the half-weighted sum is {want:e}",
            step.e_effective[q]
        );
    }
    let sum: Vec<f64> = step.e1.iter().zip(&e2).map(|(a, b)| a + b).collect();
    let formula = curled(&step.faces_n, &sum, 0.5 * DT);
    assert!(
        max_abs_difference(&formula, &step.faces_new) < FORMULA_ROUNDING,
        "{label}: B^(n+1) misses B^n - dt/2 curl(E_1 + E_2) by {:e}",
        max_abs_difference(&formula, &step.faces_new)
    );
}

#[test]
fn rk2_formula_with_a_predictor_correction_and_a_clean_corrector() {
    assert_rk2_formula(true, false, "predictor only");
}

#[test]
fn rk2_formula_with_a_corrector_correction_alone() {
    assert_rk2_formula(false, true, "corrector only");
}

#[test]
fn rk2_formula_with_corrections_in_both_stages() {
    assert_rk2_formula(true, true, "both stages");
}

// -----------------------------------------------------------------------------
// the transverse mass flux the corrected Contact EMF upwinds on. the edge EMF at a corner
// reads the mass flux of the four faces around it and takes each one's sign; on an edge of
// the interior's boundary two of those faces lie in the column beyond the interior. the
// first-order and high-order mass fluxes of a face can differ in sign: across a density
// step in a slow flow the high-order contact-resolving flux carries mass with the flow while
// the first-order fan diffuses it down the gradient. a corrected boundary edge whose outer
// face holds the first-order flux, while the same face one cell inside would hold the
// spliced high-order one, is then corrected differently from the same event in the interior.
// -----------------------------------------------------------------------------

const STEP_ROW: isize = 9;
const SLOW: f64 = 0.02;

/// a slow uniform transverse flow with a density step between rows STEP_ROW - 1 and STEP_ROW
/// in the last canonical column, the sinusoidal face field of the other gates, HLLD for the
/// high-order fluxes so a contact is resolved. `origin` is the grid column that carries
/// canonical column 0: every value is computed from the canonical column index, so two grids
/// with different origins hold the same numbers in shifted places, bit for bit.
fn build_step(origin: usize) -> Sim {
    let h = 1.0 / N as f64;
    // the canonical x of a cell center or of a face at grid position x
    let canonical = move |x: f64, face: bool| -> f64 {
        let index = if face { (x / h).round() } else { (x / h).floor() } as isize;
        let k = (index - origin as isize).rem_euclid(N as isize) as f64;
        (k + if face { 0.0 } else { 0.5 }) * h
    };
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .spacing([h, h])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .timestepping(Timestepping::Euler)
        .allocate()
        .expect("sim")
        .set_initial(move |[x, y]| {
            let xc = canonical(x, false);
            let column = (xc / h).floor() as usize;
            let row = (y / h).floor() as isize;
            let rho = if column == N - 1 && row >= STEP_ROW { 2.0 } else { 1.0 };
            MhdPrim::new(
                Prim::adiabatic(Density(rho), Tensor::new([0.0, SLOW, 0.0]), Pressure(GAMMA)),
                Tensor::new([-(2.0 * PI * y).sin(), (4.0 * PI * xc).sin(), 0.0]),
            )
        })
        .seed_faces(move |axis, [x, y]| match axis {
            0 => -(2.0 * PI * y).sin(),
            _ => (4.0 * PI * canonical(x, false)).sin(),
        })
        .build()
}

/// one forward-euler stage of `sim` with the flag raised at `cell`; every conserved cell and
/// face over the interior afterwards, x index first, plus the interior y-face mass fluxes
fn step_corrected(sim: &Sim, cell: [isize; 2]) -> (Vec<u64>, Vec<f64>, Vec<f64>) {
    let kernels = sim.substrate().with_solver(symbi_sim::substrate_seam::Solver::Hlld).expect("hlld");
    let args = StageArgs { dt: DT, a0: 0.0, ac: 1.0, stage: 0, n_stages: 1, injection_weight: 1.0, allow_elision: false };
    let s = &**sim;
    kernels.c2p(s);
    kernels.ghost_fill(s);
    assert_eq!(fold_stage_through_mark(s, &kernels, args, &mut |_| {}), 0, "the step state tripped the correction by itself");
    *s.workspace.fofc_flag.view_mut().at_mut(cell) = 1.0;
    let _ = fold_stage_from_correction(s, &kernels, args, false, &mut |_| {});
    let lo = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo];
    let grid = |f: &symbi_grid::Field<f64, 2, HostMemory>| -> Vec<f64> {
        let v = f.view();
        (0..N * N).map(|q| *v.at([lo[0] + (q / N) as isize, lo[1] + (q % N) as isize])).collect()
    };
    let mhd = s.fields.mhd.as_ref().unwrap();
    let mut state = Vec::new();
    for f in [&s.fields.cons.den, s.fields.cons.nrg_field().unwrap(), &s.fields.cons.mom[0], &s.fields.cons.mom[1], &mhd.bface.b[0], &mhd.bface.b[1]] {
        state.extend(grid(f).into_iter().map(f64::to_bits));
    }
    (state, grid(&s.fields.flux[1].den), grid(&s.workspace.flux_ho[1].den))
}

fn shifted_grid(grid: &[u64], shift: usize) -> Vec<u64> {
    let mut out = vec![0; grid.len()];
    for (q, v) in grid.iter().enumerate() {
        let (i, j) = (q / N, q % N);
        out[((i + shift) % N) * N + j] = *v;
    }
    out
}

#[test]
fn a_boundary_edge_upwinding_on_a_sign_changing_transverse_mass_flux_is_corrected_like_an_interior_one() {
    // the seam arm: canonical column 0 is grid column 0, so the density step sits in the last
    // column and the flagged cell in the first, and the flagged edge reads the step's face
    // through the ghost column
    let seam = build_step(0);
    let lo = [seam.geom.interior.spaces[0].lo, seam.geom.interior.spaces[1].lo];
    let (seam_state, _, _) = step_corrected(&seam, [lo[0], lo[1] + STEP_ROW]);

    // the premise, from a twin flagged on the step column itself so the splice keeps the
    // first-order flux on the step's face: the first- and high-order transverse mass fluxes
    // of that face have different signs
    let twin = build_step(0);
    let (_, fo, ho) = step_corrected(&twin, [lo[0] + N as isize - 1, lo[1] + STEP_ROW]);
    let face = (N - 1) * N + STEP_ROW as usize;
    assert!(
        fo[face] * ho[face] < 0.0,
        "the first-order ({:e}) and high-order ({:e}) transverse mass fluxes of the step's face \
         share a sign; the upwinding dependency is not exercised",
        fo[face], ho[face]
    );

    // the interior arm: everything shifted by half a box, so the same event is in the interior
    let shift = N / 2;
    let interior = build_step(shift);
    let (interior_state, _, _) = step_corrected(&interior, [lo[0] + shift as isize, lo[1] + STEP_ROW]);
    let fields = 6;
    let per = N * N;
    for f in 0..fields {
        let a = &seam_state[f * per..(f + 1) * per];
        let b = shifted_grid(&interior_state[f * per..(f + 1) * per], N - shift);
        let differing = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        assert_eq!(differing, 0, "field {f} of the seam arm differs from its interior translate in {differing} places");
    }
}
