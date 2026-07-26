// =============================================================================
// refinement_tracer_seeding.rs
//
// composite-hierarchy tracer seeding gate. covered coarse cells contribute no
// duplicate mass; their fine children own that material with level-aware cell
// addresses, while uncovered root cells remain root-owned.
// =============================================================================

use symbi::prelude::*;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi_hydro::state::Prim;
use symbi_sim::tracers::cell_container_address;

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 16;

type Sim =
    SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;

#[test]
fn composite_seed_uses_uncovered_coarse_and_covered_fine_mass_once() {
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .bounds([0.0], [1.0])
        .allocate()
        .unwrap()
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0]),
            pre: 1.0,
        })
        .build();
    let kernels = Kern::new(GAMMA, CFL, &coarse.geom.allocated);
    let mut hierarchy = Hierarchy::with_refinement(
        coarse,
        kernels,
        &[RefinementRegion {
            x_lo: [0.25],
            x_hi: [0.75],
        }],
        ProlongOrder::Ppm,
        |state| Kern::new(GAMMA, CFL, &state.geom.allocated),
    )
    .unwrap();
    for coord in hierarchy.levels[1].state.geom.interior.iter() {
        hierarchy.levels[1]
            .state
            .fields
            .cons
            .den
            .view_mut()
            .set(coord, 2.0);
    }

    let tracers = hierarchy.seed_mass_tracers(150);
    let root_count = tracers
        .owner
        .iter()
        .filter(|&&owner| cell_container_address(owner).unwrap().0 == 0)
        .count();
    let fine_count = tracers.len() - root_count;

    assert!((tracers.weight - 0.01).abs() < 1.0e-14);
    assert_eq!((root_count, fine_count), (50, 100));
    for (&owner, position) in tracers.owner.iter().zip(&tracers.x) {
        let (level, linear) = cell_container_address(owner).unwrap();
        if level == 0 {
            assert!(
                !(4..12).contains(&linear),
                "covered root cell {linear} received a tracer"
            );
            assert!(position[0] < 0.25 || position[0] >= 0.75);
        } else {
            assert_eq!(level, 1);
            assert!((0.25..0.75).contains(&position[0]));
        }
    }

    hierarchy.attach_mass_tracers(150);
    let mut ids = Vec::new();
    for (level, data) in hierarchy.levels.iter().enumerate() {
        let attached = data.state.tracers.as_ref().unwrap();
        assert!(
            attached.owner.iter().all(|&owner| {
                cell_container_address(owner).unwrap().0 as usize == level
            }),
            "a tracer was attached to a level that does not own its cell"
        );
        assert_eq!(attached.weight, 0.01);
        assert_eq!(attached.next_id, 150);
        ids.extend(attached.id.iter().copied());
    }
    ids.sort_unstable();
    assert_eq!(ids, (0..150).collect::<Vec<_>>());
}

#[test]
fn translating_flow_crosses_refinement_interfaces_without_losing_tracers() {
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([32])
        .bounds([0.0], [1.0])
        .boundaries(BoundaryType::Periodic)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .unwrap()
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.4]),
            pre: 1.0,
        })
        .build();
    let kernels = Kern::new(GAMMA, CFL, &coarse.geom.allocated);
    let mut hierarchy = Hierarchy::with_refinement(
        coarse,
        kernels,
        &[RefinementRegion {
            x_lo: [0.25],
            x_hi: [0.75],
        }],
        ProlongOrder::Ppm,
        |state| Kern::new(GAMMA, CFL, &state.geom.allocated),
    )
    .unwrap();
    hierarchy.seed_fine_from_coarse().unwrap();
    hierarchy.attach_mass_tracers(4096);
    let initial: std::collections::BTreeMap<_, _> = hierarchy
        .levels
        .iter()
        .filter_map(|level| level.state.tracers.as_ref())
        .flat_map(|tracers| tracers.id.iter().copied().zip(tracers.owner.iter().copied()))
        .collect();

    hierarchy.evolve_steps(2).unwrap();

    let mut ids = Vec::new();
    let mut moved = 0usize;
    let mut crossed_level = 0usize;
    for (level_index, level) in hierarchy.levels.iter().enumerate() {
        let tracers = level.state.tracers.as_ref().unwrap();
        assert_eq!(tracers.id.len(), tracers.owner.len());
        for ((&id, &owner), flags) in tracers.id.iter().zip(&tracers.owner).zip(&tracers.flags) {
            ids.push(id);
            moved += usize::from(initial[&id] != owner);
            crossed_level += usize::from(
                cell_container_address(initial[&id]).unwrap().0
                    != cell_container_address(owner).unwrap().0,
            );
            assert!(!flags.escaped);
            assert_eq!(
                cell_container_address(owner).unwrap().0 as usize,
                level_index,
                "tracer {id} is stored on the wrong hierarchy level"
            );
        }
    }
    ids.sort_unstable();
    assert_eq!(ids, (0..4096).collect::<Vec<_>>());
    assert!(moved > 0, "no tracer crossed a cell face");
    assert!(crossed_level > 0, "no tracer crossed a refinement interface");
}
