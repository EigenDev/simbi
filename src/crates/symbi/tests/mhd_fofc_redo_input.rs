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
