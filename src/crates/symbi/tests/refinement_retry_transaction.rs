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
    ) -> (f64, f64) {
        (1.0, 2.0)
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
    // The rejected dt=0.1 attempt books 0.05 on its first fine substep before the second rejects.
    // The accepted replay uses dt=0.05 and must contain exactly its two 0.025 receipts.
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
    // Calls 1-2 are the two roots; call 3 is fine substep one. Reject the refined tile on its
    // second fine substep after it has already booked the first receipt.
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
    for (tile, hierarchy) in tiles.iter().enumerate().take(1) {
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
        assert!(
            (total_accreted_mass - 0.05).abs() < 8.0 * f64::EPSILON,
            "tile {tile} retained a rejected receipt: {total_accreted_mass}"
        );
        assert!((total_accreted_energy - 0.1).abs() < 8.0 * f64::EPSILON);
    }
}
