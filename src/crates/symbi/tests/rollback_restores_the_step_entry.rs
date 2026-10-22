// =============================================================================
// rollback_restores_the_step_entry.rs
//
// a rejected explicit step replays from its entry. the kernel set keeps the
// step-entry conserved state when the attempt begins and, after a rejection,
// restores it with the primitives and their ghosts rebuilt from it, so the
// retry reconstructs from the state the rejected attempt started from.
//
// for each gas regime and scheme the gate keeps the entry, folds the attempt
// through the named stage, asserts that the attempt changed the conserved
// state (a restoration of an unchanged state proves nothing), restores, and
// asserts that the conserved vector and the primitives equal the entry bit
// for bit over the whole allocated domain, ghosts included. a second fold of
// the same stage from the restored state then reproduces the first attempt
// bit for bit: the restored state is a fresh entry.
//
// the same kernel sets report whether their snapshot is complete: the gas
// regimes on a plain store, and the flat magnetized set, which keeps the
// step-entry conserved state and both magnetic representations: complete on a
// plain store, incomplete on a store carrying an immersed body, whose state
// lives outside the snapshot.
//
// run: cargo test -p symbi --test rollback_restores_the_step_entry
// =============================================================================

use std::f64::consts::PI;
use symbi::prelude::SimSubstrate;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::Rhd;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::stage::{StageArgs, fold_stage};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 16;
const DT: f64 = 2.0e-3;

type Store = FieldStore<2, 2, HostMemory, f64>;

fn newtonian(ts: Timestepping) -> SimStateGeneric<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory> {
    let h = 1.0 / N as f64;
    SimStateGeneric::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([N, N])
        .spacing([h, h])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .timestepping(ts)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| {
            Prim::adiabatic(
                Density(1.0 + 0.3 * (2.0 * PI * x).sin() * (2.0 * PI * y).cos()),
                Tensor::new([0.4 * (2.0 * PI * y).sin(), -0.3 * (2.0 * PI * x).cos()]),
                Pressure(1.0 + 0.2 * (2.0 * PI * x).cos()),
            )
        })
        .build()
}

fn rhd(ts: Timestepping) -> SimStateGeneric<Rhd, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory> {
    let h = 1.0 / N as f64;
    SimStateGeneric::build(Rhd, IdealGas { gamma: 4.0 / 3.0 }, Cartesian)
        .cells([N, N])
        .spacing([h, h])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .timestepping(ts)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| {
            Prim::adiabatic(
                Density(1.0 + 0.3 * (2.0 * PI * x).sin() * (2.0 * PI * y).cos()),
                Tensor::new([0.4 * (2.0 * PI * y).sin(), -0.3 * (2.0 * PI * x).cos()]),
                Pressure(1.0 + 0.2 * (2.0 * PI * x).cos()),
            )
        })
        .build()
}

/// every conserved and primitive value over the allocated domain, ghosts included
fn state_bits(s: &Store) -> Vec<u64> {
    let mut out = Vec::new();
    let mut push = |f: &symbi_grid::Field<f64, 2, HostMemory>| {
        let v = f.view();
        out.extend(f.domain().iter().map(|c| v.at(c).to_bits()));
    };
    push(&s.fields.cons.den);
    for k in 0..2 {
        push(&s.fields.cons.mom[k]);
    }
    push(s.fields.cons.nrg_field().expect("energy"));
    push(&s.fields.prim.rho);
    for k in 0..2 {
        push(&s.fields.prim.vel[k]);
    }
    push(s.fields.prim.pre.as_ref().expect("pressure"));
    out
}

fn args(ts: Timestepping, index: usize) -> StageArgs {
    let stages = ts.stages();
    StageArgs {
        dt: DT,
        a0: stages[index].0,
        ac: stages[index].1,
        stage: index,
        n_stages: stages.len(),
        injection_weight: 1.0,
        allow_elision: false,
        exit_fraction: symbi_sim::driver::stage_exit_fraction(stages, index),
    }
}

/// the attempt: the stages through `through`, as a driver folds them
fn attempt<K: KernelSet<2, 2, HostMemory, f64>>(s: &Store, k: &K, ts: Timestepping, through: usize) {
    for index in 0..=through {
        let _ = fold_stage(s, k, args(ts, index), &mut |_| {});
    }
}

fn assert_restores<K: KernelSet<2, 2, HostMemory, f64>>(s: &Store, k: &K, ts: Timestepping, through: usize, label: &str) {
    k.c2p(s);
    k.ghost_fill(s);
    assert!(k.rollback_snapshot_is_complete(s), "{label}: the snapshot is reported incomplete");
    // the driver's order: the retry snapshot, then the step snapshot of a multistage scheme
    k.snapshot_retry(s);
    if ts.stages().len() > 1 {
        k.snapshot(s);
    }
    let entry = state_bits(s);
    attempt(s, k, ts, through);
    let rejected = state_bits(s);
    assert_ne!(entry, rejected, "{label}: the attempt left the state unchanged; restoration is vacuous");
    k.restore_step(s);
    let restored = state_bits(s);
    let differing = restored.iter().zip(&entry).filter(|(a, b)| a != b).count();
    assert_eq!(differing, 0, "{label}: {differing} values differ from the step entry after the restore");
    // a fresh attempt from the restored state reproduces the first
    if ts.stages().len() > 1 {
        k.snapshot(s);
    }
    attempt(s, k, ts, through);
    assert_eq!(state_bits(s), rejected, "{label}: the retry from the restored state differs from the first attempt");
}

#[test]
fn newtonian_euler() {
    let mut s = newtonian(Timestepping::Euler);
    s.dt = DT;
    let k = s.substrate();
    assert_restores(&*s, &k, Timestepping::Euler, 0, "newtonian, euler");
}

#[test]
fn newtonian_rk2_rejected_after_the_predictor() {
    let mut s = newtonian(Timestepping::Rk2);
    s.dt = DT;
    let k = s.substrate();
    assert_restores(&*s, &k, Timestepping::Rk2, 0, "newtonian, rk2, predictor");
}

#[test]
fn newtonian_rk2_rejected_after_the_corrector() {
    let mut s = newtonian(Timestepping::Rk2);
    s.dt = DT;
    let k = s.substrate();
    assert_restores(&*s, &k, Timestepping::Rk2, 1, "newtonian, rk2, corrector");
}

#[test]
fn rhd_euler() {
    let mut s = rhd(Timestepping::Euler);
    s.dt = DT;
    let k = s.substrate();
    assert_restores(&*s, &k, Timestepping::Euler, 0, "rhd, euler");
}

#[test]
fn rhd_rk2_rejected_after_the_predictor() {
    let mut s = rhd(Timestepping::Rk2);
    s.dt = DT;
    let k = s.substrate();
    assert_restores(&*s, &k, Timestepping::Rk2, 0, "rhd, rk2, predictor");
}

#[test]
fn rhd_rk2_rejected_after_the_corrector() {
    let mut s = rhd(Timestepping::Rk2);
    s.dt = DT;
    let k = s.substrate();
    assert_restores(&*s, &k, Timestepping::Rk2, 1, "rhd, rk2, corrector");
}

/// a flat magnetized store on the periodic square
fn magnetized() -> SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory> {
    let h = 1.0 / N as f64;
    SimStateGeneric::<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::build(
        NewtonianMhd,
        IdealGas { gamma: 5.0 / 3.0 },
        Cartesian,
    )
    .cells([N, N])
    .spacing([h, h])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .allocate()
    .expect("sim")
    .set_initial(|[x, y]| {
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.1 * (2.0 * PI * y).sin(), 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([-(2.0 * PI * y).sin(), (4.0 * PI * x).sin(), 0.0]),
        )
    })
    .seed_faces(|axis, [x, y]| match axis {
        0 => -(2.0 * PI * y).sin(),
        _ => (4.0 * PI * x).sin(),
    })
    .build()
}

/// the flat magnetized set reports its snapshot complete on a plain store and incomplete on a store
/// carrying an immersed body, whose position and receipts the snapshot does not hold
#[test]
fn the_flat_magnetized_set_reports_whether_its_snapshot_is_complete() {
    let plain = magnetized();
    let k = plain.substrate();
    assert!(k.rollback_snapshot_is_complete(&*plain), "a plain magnetized store reports its snapshot incomplete");
    let sink = symbi_ib::Body::gravitational(0, Tensor::new([0.5, 0.5]), Tensor::zeros(), 0.0, 0.1, 0.05);
    let bodied = magnetized().with_bodies(symbi_ib::BodyCollection::new().add(sink));
    let k = bodied.substrate();
    assert!(!k.rollback_snapshot_is_complete(&*bodied), "a store carrying a body reports its snapshot complete");
}
