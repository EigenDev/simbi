// =============================================================================
// refinement_mhd_tracers.rs
//
// mass-transport tracer receipts on a constrained-transport hierarchy. a
// uniform magnetized translation crosses coarse-fine interfaces while tracer
// identity, ownership, and magnetic divergence remain intact.
// =============================================================================

use symbi::prelude::*;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_sim::mass_transport::{ContainerId, ItoOrder};
use symbi_sim::tracers::{
    ContinuousTracerRecord, ContinuousTracerSet, cell_container_address,
};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.35;
const B0: [f64; 3] = [0.2, 0.1, 0.0];

type Sim =
    SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 3>;

fn divergence_max(sim: &Sim) -> f64 {
    let magnetic = sim.fields.mhd.as_ref().unwrap();
    let mut maximum = 0.0_f64;
    for cell in sim.geom.interior.iter() {
        let mut divergence = 0.0;
        for axis in 0..3 {
            let mut upper = cell;
            upper[axis] += 1;
            divergence +=
                (*magnetic.bface[axis].view().at(upper)
                    - *magnetic.bface[axis].view().at(cell))
                    / sim.geom.dx[axis];
        }
        maximum = maximum.max(divergence.abs());
    }
    maximum
}

#[test]
fn translating_nmhd_flow_crosses_refinement_interfaces_without_tracer_loss() {
    let coarse = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([32, 8, 4])
        .bounds([0.0; 3], [1.0; 3])
        .boundaries(BoundaryType::Periodic)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .unwrap()
        .set_initial(|_| MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.4, 0.0, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new(B0),
        })
        .seed_faces_uniform(B0)
        .build();
    let kernels = Kern::new(GAMMA, CFL, 1.0, &coarse.geom.allocated);
    let mut hierarchy = Hierarchy::with_refinement(
        coarse,
        kernels,
        &[RefinementRegion {
            x_lo: [0.25, 0.0, 0.0],
            x_hi: [0.75, 1.0, 1.0],
        }],
        ProlongOrder::Ppm,
        |state| Kern::new(GAMMA, CFL, 1.0, &state.geom.allocated),
    )
    .unwrap();
    hierarchy.seed_fine_from_coarse().unwrap();
    hierarchy.attach_mass_tracers(4096);
    let initial: std::collections::BTreeMap<_, _> = hierarchy
        .levels
        .iter()
        .flat_map(|level| {
            let tracers = level.state.tracers.as_ref().unwrap();
            tracers
                .id
                .iter()
                .copied()
                .zip(tracers.owner.iter().copied())
        })
        .collect();

    hierarchy.evolve_steps(2).unwrap();

    let mut ids = Vec::new();
    let mut crossed_level = 0usize;
    for (level_index, level) in hierarchy.levels.iter().enumerate() {
        let tracers = level.state.tracers.as_ref().unwrap();
        for ((&id, &owner), flags) in tracers.id.iter().zip(&tracers.owner).zip(&tracers.flags) {
            ids.push(id);
            crossed_level += usize::from(
                cell_container_address(initial[&id]).unwrap().0
                    != cell_container_address(owner).unwrap().0,
            );
            assert!(!flags.escaped);
            assert!(!flags.crossed_sink);
            assert_eq!(
                cell_container_address(owner).unwrap().0 as usize,
                level_index,
                "tracer {id} is stored on the wrong hierarchy level"
            );
        }
        assert!(
            divergence_max(&level.state) < 1.0e-12,
            "level {level_index} magnetic divergence drifted"
        );
    }
    ids.sort_unstable();
    assert_eq!(ids, (0..4096).collect::<Vec<_>>());
    assert!(
        crossed_level > 0,
        "no MHD tracer crossed a refinement interface"
    );
}

#[test]
fn continuous_tracer_record_moves_to_finest_active_level_without_state_loss() {
    let coarse = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([16, 4, 2])
        .bounds([0.0; 3], [1.0; 3])
        .boundaries(BoundaryType::Periodic)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .unwrap()
        .set_initial(|_| MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.4, 0.0, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new(B0),
        })
        .seed_faces_uniform(B0)
        .build();
    let kernels = Kern::new(GAMMA, CFL, 1.0, &coarse.geom.allocated);
    let mut hierarchy = Hierarchy::with_refinement(
        coarse,
        kernels,
        &[RefinementRegion {
            x_lo: [0.25, 0.0, 0.0],
            x_hi: [0.75, 1.0, 1.0],
        }],
        ProlongOrder::Ppm,
        |state| Kern::new(GAMMA, CFL, 1.0, &state.geom.allocated),
    )
    .unwrap();
    hierarchy.seed_fine_from_coarse().unwrap();
    let mut coarse_tracers =
        ContinuousTracerSet::<3, HostMemory>::allocate(1, ItoOrder::Three).unwrap();
    coarse_tracers.weight = 0.125;
    coarse_tracers.run_seed = 91;
    coarse_tracers.next_id = 43;
    coarse_tracers.injection_remainder = 0.03125;
    coarse_tracers
        .push_host(ContinuousTracerRecord {
            x: [0.5, 0.4, 0.6],
            step_x: [0.49, 0.39, 0.59],
            id: 42,
            cohort: 7,
            owner: ContainerId(0),
            escaped: 0,
            crossed_sink: 0,
            crossing_time: 1.25,
            random_counter: 19,
        })
        .unwrap();
    hierarchy.levels[0].state.continuous_tracers = Some(coarse_tracers);
    hierarchy.levels[1].state.continuous_tracers =
        Some(ContinuousTracerSet::allocate(0, ItoOrder::Three).unwrap());

    assert_eq!(hierarchy.migrate_continuous_tracers_to_finest().unwrap(), 1);
    assert_eq!(
        hierarchy.levels[0]
            .state
            .continuous_tracers
            .as_ref()
            .unwrap()
            .len,
        0
    );
    let fine = hierarchy.levels[1]
        .state
        .continuous_tracers
        .as_mut()
        .unwrap();
    assert_eq!(fine.len, 1);
    assert_eq!(fine.order, ItoOrder::Three);
    assert_eq!(fine.weight, 0.125);
    assert_eq!(fine.run_seed, 91);
    assert_eq!(fine.next_id, 43);
    assert_eq!(fine.injection_remainder, 0.03125);
    let record = fine.swap_remove_host(0).unwrap();
    assert_eq!(record.id, 42);
    assert_eq!(record.cohort, 7);
    assert_eq!(record.x, [0.5, 0.4, 0.6]);
    assert_eq!(record.step_x, [0.49, 0.39, 0.59]);
    assert_eq!(record.crossing_time, 1.25);
    assert_eq!(record.random_counter, 19);
    assert_eq!(cell_container_address(record.owner).unwrap().0, 1);
    fine.push_host(ContinuousTracerRecord {
        x: [0.9, 0.4, 0.6],
        ..record
    })
    .unwrap();

    assert_eq!(hierarchy.migrate_continuous_tracers_to_finest().unwrap(), 1);
    let coarse = hierarchy.levels[0]
        .state
        .continuous_tracers
        .as_mut()
        .unwrap();
    assert_eq!(coarse.len, 1);
    let record = coarse.swap_remove_host(0).unwrap();
    assert_eq!(record.id, 42);
    assert_eq!(record.cohort, 7);
    assert_eq!(record.x, [0.9, 0.4, 0.6]);
    assert_eq!(record.step_x, [0.49, 0.39, 0.59]);
    assert_eq!(record.crossing_time, 1.25);
    assert_eq!(record.random_counter, 19);
    assert_eq!(cell_container_address(record.owner).unwrap().0, 0);
    assert_eq!(coarse.order, ItoOrder::Three);
    assert_eq!(coarse.weight, 0.125);
    assert_eq!(coarse.run_seed, 91);
    assert_eq!(coarse.next_id, 43);
    assert_eq!(coarse.injection_remainder, 0.03125);
    coarse.push_host(record).unwrap();
    hierarchy.levels[1]
        .state
        .continuous_tracers
        .as_mut()
        .unwrap()
        .push_host(ContinuousTracerRecord {
            x: [0.5, 0.4, 0.6],
            step_x: [0.5, 0.4, 0.6],
            id: 44,
            cohort: 8,
            owner: ContainerId(0),
            escaped: 0,
            crossed_sink: 0,
            crossing_time: 0.0,
            random_counter: 29,
        })
        .unwrap();

    hierarchy.evolve_steps(1).unwrap();

    let fine_state = &hierarchy.levels[1].state;
    let coefficient_ghost = [
        fine_state.geom.interior.spaces[0].hi,
        fine_state.geom.interior.spaces[1].lo,
        fine_state.geom.interior.spaces[2].lo,
    ];
    assert!(
        *fine_state
            .ito_coefficients
            .as_ref()
            .unwrap()
            .drift[0]
            .view()
            .at(coefficient_ghost)
            > 0.1,
        "fine coarse/fine coefficient ghost did not inherit parent transport"
    );
    let edge_rates = fine_state
        .ito_coefficients
        .as_ref()
        .unwrap()
        .interpolate(&fine_state.geom, [0.751, 0.4, 0.6])
        .unwrap();
    assert!(
        edge_rates[0].drift > 0.1,
        "a trajectory crossing the patch edge sampled an unresolved halo"
    );
    let coarse = hierarchy.levels[0]
        .state
        .continuous_tracers
        .as_ref()
        .unwrap();
    let fine = hierarchy.levels[1]
        .state
        .continuous_tracers
        .as_ref()
        .unwrap();
    assert_eq!(coarse.len, 1);
    assert_eq!(fine.len, 1);
    unsafe {
        assert_eq!(*coarse.id.as_ptr::<u64>(), 42);
        assert_eq!(*coarse.random_counter.as_ptr::<u64>(), 20);
        assert_eq!(*fine.id.as_ptr::<u64>(), 44);
        assert_eq!(*fine.random_counter.as_ptr::<u64>(), 31);
    }
}
