// =============================================================================
// failure_exit_restores_entry.rs
//
// a run that ends on a failure no replay resolves leaves its last accepted
// state, and a checkpoint marked as a failure's diagnostic snapshot never
// resumes.
//
// the fixture: a periodic 2.5D newtonian MHD wave on a 32 x 32 single
// hierarchy with a drain sink at its center. every attempt of root step 2 has
// one cell's stage input poisoned beyond the reach of the first-order redo, so
// each attempt freezes it outside the admissible set, the correction rejects
// the attempt, and the sixteenth consecutive freeze panics. the gates:
//   - the step transaction turns that panic into an unrecoverable halt naming
//     the freeze, and the halted run equals bit for bit a clean run stopped
//     after step 1: conserved, primitive, cell and face fields, the clock,
//     the sink's accretion record, and the accepted guard totals;
//   - a checkpoint whose metadata carries `restartable = false` is refused by
//     the restart, and the same file without the mark restores.
//
// run: cargo test -p symbi --test failure_exit_restores_entry
// =============================================================================

use std::f64::consts::TAU;
use std::ops::ControlFlow;
use symbi::prelude::SimSubstrate;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_sim::step_transaction::StepHalt;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 32;
const H: f64 = 1.0 / N as f64;
// a cell a quarter box from the sink, outside its drain mask
const POISONED: [isize; 2] = [8, 24];
// more armed attempts than the freeze streak allows, fewer than the retry guard's rejections
const ARMED_ATTEMPTS: usize = 18;

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

fn build(armed: bool) -> Hierarchy<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, <Sim as SimSubstrate<HostMemory, f64, 2>>::KernelSet> {
    let sim = Sim::build(NewtonianMhd, IdealGas { gamma: 5.0 / 3.0 }, Cartesian)
        .cells([N, N])
        .spacing([H; 2])
        .origin([0.0; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| {
            let rho = 1.0 + 0.2 * (TAU * x).sin() * (TAU * y).sin();
            let v = [0.3 * (TAU * y).sin(), 0.3 * (TAU * x).sin(), 0.0];
            MhdPrim::new(
                Prim::adiabatic(Density(rho), Tensor::new(v), Pressure(1.0)),
                Tensor::new([-0.2 * (TAU * y).sin(), 0.2 * (2.0 * TAU * x).sin(), 0.0]),
            )
        })
        .seed_faces(|axis, [x, y]| if axis == 0 { -0.2 * (TAU * y).sin() } else { 0.2 * (2.0 * TAU * x).sin() })
        .build();
    if armed {
        // root step 2 enters at iteration 1: every attempt of it consumes one slot
        *sim.workspace.unrecoverable_once.lock().unwrap() = vec![(1, 0, POISONED, false); ARMED_ATTEMPTS];
    }
    let kernels = sim.substrate();
    let sink = BodyCollection::new().add(
        Body::black_hole(0, Tensor::new([0.5, 0.5]), Tensor::zeros(), 0.01, 0.1, 0.05, 0.0, 1.0, 0.1)
            .with_surface(SurfaceSpec::Drain)
            .with_magnetic(MagneticSpec::None),
    );
    let mut hier = Hierarchy::single(sim, kernels).with_bodies(sink);
    hier.prime();
    hier
}

/// every interior value of the conserved, primitive and cell fields and every face value, bits
fn state_bits(store: &FieldStore<2, 3>) -> Vec<Vec<u64>> {
    let bits = |f: &symbi_grid::Field<f64, 2, HostMemory>, d: &symbi_algebra::Domain<2>| -> Vec<u64> {
        let v = f.view();
        d.iter().map(|c| v.at(c).to_bits()).collect()
    };
    let interior = &store.geom.interior;
    let (cons, prim) = (&store.fields.cons, &store.fields.prim);
    let mhd = store.fields.mhd.as_ref().expect("mhd fields");
    let mut all = vec![bits(&cons.den, interior), bits(cons.nrg_field().unwrap(), interior), bits(&prim.rho, interior)];
    all.push(bits(prim.pre_field().unwrap(), interior));
    all.extend((0..3).map(|k| bits(&cons.mom[k], interior)));
    all.extend((0..3).map(|k| bits(&prim.vel[k], interior)));
    all.extend((0..3).map(|k| bits(&mhd.bcell.b[k], interior)));
    all.extend((0..2).map(|d| bits(&mhd.bface.b[d], mhd.bface.b[d].domain())));
    all
}

/// the sink's accreted mass and rate, bits
fn sink_book(store: &FieldStore<2, 3>) -> Vec<u64> {
    let mut book = Vec::new();
    store.immersed.as_ref().expect("the sink").bodies.visit_accretion(|b| {
        if let symbi_ib::BodyKind::BlackHole { total_accreted_mass, accretion_rate, .. } = &b.kind {
            book.extend([total_accreted_mass.to_bits(), accretion_rate.to_bits()]);
        }
    });
    book
}

#[test]
fn an_unrecoverable_attempt_halts_at_the_last_accepted_state() {
    let mut halted = build(true);
    let d_halted = halted.evolve_with_callback(10.0, 1, |_| ControlFlow::Continue(())).expect("the halted run");
    let crash = halted.crash.clone().expect("the armed run never halted; the gate is vacuous");
    assert!(crash.panic.is_none(), "the panic escaped the step transaction: {:?}", crash.panic);
    match &crash.halt {
        Some(StepHalt::Unrecoverable { detail }) => {
            assert!(detail.contains("FOFC last-resort freeze"), "the halt does not name the freeze: {detail}")
        }
        other => panic!("the run ended on {other:?}, the freeze streak was the armed failure"),
    }
    assert_eq!(crash.iter, 1, "the halt is reported at the entry of root step 2");

    let mut clean = build(false);
    let d_clean = clean
        .evolve_with_callback(10.0, 1, |h| if h.levels[0].state.iteration >= 1 { ControlFlow::Break(()) } else { ControlFlow::Continue(()) })
        .expect("the clean run");
    let (h, c) = (&halted.levels[0].state, &clean.levels[0].state);
    assert_eq!((h.iteration, h.time.to_bits()), (c.iteration, c.time.to_bits()), "the halted run's clock left the step entry");
    assert!(state_bits(h) == state_bits(c), "the halted run's fields differ from the state after step 1");
    assert_eq!(sink_book(h), sink_book(c), "the halted run's sink book carries the failed attempts' accretion");
    assert!(!sink_book(c).is_empty() && sink_book(c)[0] != 0, "the sink accreted nothing; its book is no evidence");
    assert_eq!(
        (d_halted.guards.troubled_cells.total, d_halted.guards.frozen_cells.total),
        (d_clean.guards.troubled_cells.total, d_clean.guards.frozen_cells.total),
        "the failed attempts' guard acts were accepted"
    );
}

#[test]
fn a_diagnostic_checkpoint_is_refused_by_the_restart() {
    let dir = std::env::temp_dir().join(format!("symbi_failure_exit_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut run = build(false);
    run.evolve_with_callback(10.0, 1, |h| if h.levels[0].state.iteration >= 1 { ControlFlow::Break(()) } else { ControlFlow::Continue(()) })
        .expect("the run");
    let states: Vec<_> = run.levels.iter().map(|l| &l.state).collect();
    let diagnostic = dir.join("diagnostic.h5");
    let plain = dir.join("plain.h5");
    let meta = symbi_io::Metadata::new();
    symbi_sim::checkpoint::write_hierarchy_checkpoint(&states, diagnostic.to_str().unwrap(), &meta.clone().with("restartable", false))
        .expect("write the diagnostic snapshot");
    symbi_sim::checkpoint::write_hierarchy_checkpoint(&states, plain.to_str().unwrap(), &meta).expect("write the checkpoint");

    let refused = build(false).restore_from_checkpoint(diagnostic.to_str().unwrap());
    assert!(
        matches!(&refused, Err(e) if e.contains("diagnostic snapshot of a failed step")),
        "the restart accepted a diagnostic snapshot: {refused:?}"
    );
    build(false).restore_from_checkpoint(plain.to_str().unwrap()).expect("the unmarked checkpoint restores");
    let _ = std::fs::remove_dir_all(&dir);
}
