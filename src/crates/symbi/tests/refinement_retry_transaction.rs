//! A rejected fine substep must be a transaction over hierarchy side-cars, not only gas fields.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use symbi_hydro::quantity::{Density, Pressure};

use symbi::sim::decomp::LocalCopy;
use symbi::sim::refinement::{
    Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed,
};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, BodyKind};
use symbi_sim::state::FieldStore;
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

struct RejectOnce {
    calls: Arc<AtomicUsize>,
    reject_at: usize,
}

impl KernelSet<1, 1, HostMemory, f64> for RejectOnce {
    fn reconstruction_reach(&self) -> u8 {
        1
    }

    fn flux(&self, _store: &FieldStore<1, 1, HostMemory>, _dir: usize) {}

    fn c2p(&self, store: &FieldStore<1, 1, HostMemory>) {
        for cell in store.geom.interior.iter() {
            let rho = *store.fields.cons.den.view().at(cell);
            let mom = *store.fields.cons.mom[0].view().at(cell);
            let nrg = *store.fields.cons.nrg_field().unwrap().view().at(cell);
            store.fields.prim.rho.view_mut().set(cell, rho);
            store.fields.prim.vel[0].view_mut().set(cell, mom / rho);
            store
                .fields
                .prim
                .pre_field()
                .unwrap()
                .view_mut()
                .set(cell, (GAMMA - 1.0) * (nrg - 0.5 * mom * mom / rho));
        }
        store.mark_primitives_recovered();
    }

    fn godunov_stage(&self, _store: &FieldStore<1, 1, HostMemory>, _dt: f64, _a0: f64, _ac: f64) {}

    fn cfl(&self, _store: &FieldStore<1, 1, HostMemory>) -> f64 {
        0.1
    }
    fn ghost_fill(&self, _store: &FieldStore<1, 1, HostMemory>) {}
    fn snapshot(&self, _store: &FieldStore<1, 1, HostMemory>) {}

    fn fofc_active(&self) -> bool {
        true
    }
    fn fofc(
        &self,
        _store: &FieldStore<1, 1, HostMemory>,
        _dt: f64,
        _a0: f64,
        _ac: f64,
        _stage: u8,
        _redo_update: &mut dyn FnMut(),
    ) -> symbi_sim::substrate_seam::FofcReport {
        use symbi_sim::substrate_seam::{FofcDecision, FofcReport, SourceReplayOutcome};
        // root, first fine substep, then second fine substep: reject only the latter.
        let reject = self.calls.fetch_add(1, Ordering::SeqCst) + 1 == self.reject_at;
        if reject {
            // a coherent rejected pass: one troubled cell whose exterior
            // freeze act is the retry evidence.
            FofcReport::of_pass(
                1,
                1,
                1,
                SourceReplayOutcome::SharedRedo,
                FofcDecision::RetryStep,
            )
        } else {
            FofcReport::inactive()
        }
    }

    fn horizon_accretion(
        &self,
        _store: &FieldStore<1, 1, HostMemory>,
        _diagnostic_radius: f64,
    ) -> Option<symbi_sim::driver::HorizonPartial> {
        Some(symbi_sim::driver::HorizonPartial::from_rates(1.0, 2.0))
    }
}

#[test]
fn rejected_second_fine_substep_does_not_double_book_horizon_receipts() {
    let calls = Arc::new(AtomicUsize::new(0));
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([16])
        .spacing([1.0 / 16.0])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .timestepping(Timestepping::Euler)
        .cfl(0.4)
        .allocate()
        .unwrap()
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::zeros(), Pressure(1.0)))
        .build();
    let mut hierarchy = Hierarchy::with_refinement(
        coarse,
        RejectOnce {
            calls: calls.clone(),
            reject_at: 3,
        },
        &[RefinementRegion {
            x_lo: [0.25],
            x_hi: [0.75],
        }],
        ProlongOrder::Plm,
        |_| RejectOnce {
            calls: calls.clone(),
            reject_at: 3,
        },
    )
    .unwrap()
    .with_bodies(BodyCollection::new().add(Body::horizon(0, 0.01, 0.1)));

    hierarchy.evolve_steps(1).unwrap();

    let body = hierarchy.levels[1]
        .state
        .immersed
        .as_ref()
        .unwrap()
        .bodies
        .get(0);
    let BodyKind::Horizon {
        total_accreted_mass,
        total_accreted_energy,
        ..
    } = body.kind
    else {
        unreachable!()
    };
    // the root owns the horizon ledger: the rejected dt = 0.1 attempt books 0.1 at its root tail
    // before the second fine substep rejects, and the accepted replay at dt = 0.05 books its one
    // root receipt of 0.05 alone, on every level's copy of the body
    assert_eq!(hierarchy.levels[0].state.dt, 0.05);
    assert!((total_accreted_mass - 0.05).abs() < 8.0 * f64::EPSILON);
    assert!((total_accreted_energy - 0.10).abs() < 8.0 * f64::EPSILON);
}

#[test]
fn decomposed_rejection_rolls_every_tile_back_collectively() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut tiles = Vec::new();
    for tile in 0..2 {
        let origin = tile as f64 * 0.5;
        let boundaries = Boundaries::per_axis([[
            if tile == 0 {
                BoundaryType::Outflow
            } else {
                BoundaryType::CoarseFine
            },
            if tile == 1 {
                BoundaryType::Outflow
            } else {
                BoundaryType::CoarseFine
            },
        ]]);
        let root = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
            .cells([16])
            .origin([origin])
            .spacing([1.0 / 32.0])
            .boundaries(boundaries)
            .timestepping(Timestepping::Euler)
            .cfl(0.4)
            .allocate()
            .unwrap()
            .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::zeros(), Pressure(1.0)))
            .build();
        let root_kernel = RejectOnce {
            calls: calls.clone(),
            reject_at: 4,
        };
        let hierarchy = if tile == 0 {
            Hierarchy::with_refinement(
                root,
                root_kernel,
                &[RefinementRegion {
                    x_lo: [0.125],
                    x_hi: [0.375],
                }],
                ProlongOrder::Plm,
                |_| RejectOnce {
                    calls: calls.clone(),
                    reject_at: 4,
                },
            )
            .unwrap()
        } else {
            Hierarchy::single(root, root_kernel)
        }
        .with_bodies(BodyCollection::new().add(Body::horizon(0, 0.01, 0.1)));
        tiles.push(hierarchy);
    }
    // calls 1-2 are the two roots; call 3 is fine substep one. the refined tile rejects its second
    // fine substep after it has booked the first receipt.
    let devices = [0, 0];
    evolve_hierarchy_decomposed(
        &mut tiles,
        [2],
        &devices,
        &LocalCopy,
        Timestepping::Euler,
        0.0,
        0.05,
        u64::MAX,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
    // premises: the fourth correction pass ran and rejected, and the replay took two accepted root
    // steps at the halved dt (a run without the rejection reaches t = 0.05 in one step)
    assert!(calls.load(Ordering::SeqCst) >= 4, "the rejecting correction pass never ran");
    for hierarchy in &tiles {
        assert_eq!(hierarchy.levels[0].state.iteration, 2, "the root did not replay the rejected step at half the timestep");
    }
    // the root owns the horizon's receipts: each tile's root contributes its partial (rate one),
    // the merged rate two is booked once per accepted root step on every tile, so the accepted
    // steps' receipts total 2 x 0.05 on every level of every tile and the rejected attempt's none
    for (tile, hierarchy) in tiles.iter().enumerate() {
        let want = 0.1;
        let body = hierarchy.levels[hierarchy.levels.len() - 1]
            .state
            .immersed
            .as_ref()
            .unwrap()
            .bodies
            .get(0);
        let BodyKind::Horizon {
            total_accreted_mass,
            total_accreted_energy,
            ..
        } = body.kind
        else {
            unreachable!()
        };
        assert!(
            (total_accreted_mass - want).abs() < 8.0 * f64::EPSILON,
            "tile {tile} holds {total_accreted_mass}, the accepted steps' receipts are {want}"
        );
        assert!((total_accreted_energy - 2.0 * want).abs() < 8.0 * f64::EPSILON);
    }
}

/// a kernel set without FOFC whose every stage adds its timestep to the interior density, with a
/// retry snapshot of that density. a no-op additive source makes the stage capture its input,
/// which the covered-core hold of a refined level reads
struct Accumulate {
    entry: std::sync::Mutex<Vec<f64>>,
}

impl Accumulate {
    fn new() -> Self {
        Self { entry: std::sync::Mutex::new(Vec::new()) }
    }
}

impl KernelSet<1, 1, HostMemory, f64> for Accumulate {
    fn reconstruction_reach(&self) -> u8 {
        1
    }
    fn flux(&self, _store: &FieldStore<1, 1, HostMemory>, _dir: usize) {}
    fn c2p(&self, store: &FieldStore<1, 1, HostMemory>) {
        for cell in store.geom.interior.iter() {
            let rho = *store.fields.cons.den.view().at(cell);
            store.fields.prim.rho.view_mut().set(cell, rho);
            store.fields.prim.vel[0].view_mut().set(cell, 0.0);
            store.fields.prim.pre_field().unwrap().view_mut().set(cell, 1.0);
        }
        store.mark_primitives_recovered();
    }
    fn godunov_stage(&self, store: &FieldStore<1, 1, HostMemory>, dt: f64, _a0: f64, _ac: f64) {
        for cell in store.geom.interior.iter() {
            let rho = *store.fields.cons.den.view().at(cell);
            store.fields.cons.den.view_mut().set(cell, rho + dt);
        }
    }
    fn cfl(&self, _store: &FieldStore<1, 1, HostMemory>) -> f64 {
        0.1
    }
    fn ghost_fill(&self, _store: &FieldStore<1, 1, HostMemory>) {}
    fn snapshot(&self, _store: &FieldStore<1, 1, HostMemory>) {}
    fn has_additive_source(&self) -> bool {
        true
    }
    fn snapshot_stage(&self, store: &FieldStore<1, 1, HostMemory>) {
        for cell in store.geom.allocated.iter() {
            store.stage_input().den.view_mut().set(cell, *store.fields.cons.den.view().at(cell));
        }
    }
    fn snapshot_retry(&self, store: &FieldStore<1, 1, HostMemory>) {
        *self.entry.lock().unwrap() = store.geom.interior.iter().map(|cell| *store.fields.cons.den.view().at(cell)).collect();
    }
    fn restore_step(&self, store: &FieldStore<1, 1, HostMemory>) {
        let entry = self.entry.lock().unwrap();
        for (cell, &rho) in store.geom.interior.iter().zip(entry.iter()) {
            store.fields.cons.den.view_mut().set(cell, rho);
        }
        self.c2p(store);
    }
    fn horizon_accretion(&self, _store: &FieldStore<1, 1, HostMemory>, _diagnostic_radius: f64) -> Option<symbi_sim::driver::HorizonPartial> {
        Some(symbi_sim::driver::HorizonPartial::from_rates(0.0, 0.0))
    }
}

/// a forced fine rejection on a kernel set without FOFC replays the root step from its entry: the
/// rejection is a source of its own, so the step captures the root's retry snapshot and the replay
/// at half the candidate leaves an uncovered root cell at 1 + 0.05, the rejected attempt's 0.1
/// gone
#[test]
fn a_forced_rejection_without_fofc_replays_from_the_step_entry() {
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([16])
        .spacing([1.0 / 16.0])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .timestepping(Timestepping::Euler)
        .cfl(0.4)
        .allocate()
        .unwrap()
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::zeros(), Pressure(1.0)))
        .build();
    let region = RefinementRegion { x_lo: [0.25], x_hi: [0.75] };
    let mut hierarchy = Hierarchy::with_refinement(coarse, Accumulate::new(), &[region], ProlongOrder::Plm, |_| Accumulate::new()).unwrap();
    // the only stage of the second fine substep of the first root step
    hierarchy.reject_fine_once_at(0, 1, 0);
    hierarchy.evolve_steps(1).unwrap();
    assert!(!hierarchy.forced_fine_rejection_pending(), "premise: the forced rejection never fired");
    let root = &hierarchy.levels[0].state;
    assert_eq!(root.time, 0.05, "premise: the step was accepted at half its candidate");
    let uncovered = root.geom.interior.iter().next().unwrap();
    assert_eq!(*root.fields.cons.den.view().at(uncovered), 1.0 + 0.05, "the replay started from the rejected attempt's root");
}
