// =============================================================================
// refinement_tracer_seeding.rs
//
// composite-hierarchy tracer seeding gate. covered coarse cells contribute no
// duplicate mass; their fine children own that material with level-aware cell
// addresses, while uncovered root cells remain root-owned.
// =============================================================================

use symbi::prelude::*;
use symbi::regimes::substrate_kernels::GradientBc;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi_hydro::NEWTONIAN_SPEC;
use symbi_hydro::state::Prim;
use symbi_sim::tracers::cell_container_address;
use symbi_source_compile::SourceConfig;
use symbi_source_compile::expr_bridge::{build_boundary_dag, build_user_source};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 16;

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;

#[test]
fn composite_seed_uses_uncovered_coarse_and_covered_fine_mass_once() {
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .bounds([0.0], [1.0])
        .allocate()
        .unwrap()
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0]), Pressure(1.0)))
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
            attached
                .owner
                .iter()
                .all(|&owner| { cell_container_address(owner).unwrap().0 as usize == level }),
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
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.4]), Pressure(1.0)))
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
        .flat_map(|tracers| {
            tracers
                .id
                .iter()
                .copied()
                .zip(tracers.owner.iter().copied())
        })
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
    assert!(
        crossed_level > 0,
        "no tracer crossed a refinement interface"
    );
}

#[test]
fn refined_density_source_spawns_only_the_composite_added_mass() {
    let source = SourceConfig::from_json(
        r#"{
            "kind": "raw", "dim": 1, "outputs": [0], "params": [],
            "target": "den",
            "nodes": [ {"op": "CONSTANT", "value": 0.2} ]
        }"#,
    )
    .unwrap();
    let make_kernels = |state: &Sim| {
        Kern::new(GAMMA, CFL, &state.geom.allocated).with_runtime_source(
            build_user_source(&source, &NEWTONIAN_SPEC).unwrap(),
            source.params.clone(),
        )
    };
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([32])
        .bounds([0.0], [1.0])
        .boundaries(BoundaryType::Periodic)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .unwrap()
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0]), Pressure(1.0)))
        .build();
    let kernels = make_kernels(&coarse);
    let mut hierarchy = Hierarchy::with_refinement(
        coarse,
        kernels,
        &[RefinementRegion {
            x_lo: [0.25],
            x_hi: [0.75],
        }],
        ProlongOrder::Ppm,
        make_kernels,
    )
    .unwrap();
    hierarchy.seed_fine_from_coarse().unwrap();
    hierarchy.attach_mass_tracers(1000);

    hierarchy.evolve_steps(2).unwrap();

    let mut composite_mass = 0.0;
    for level in &hierarchy.levels {
        let geometry = level.state.geom.block_geometry(level.state.physics.metric);
        for coord in level.state.geom.interior.iter() {
            if level
                .coverage
                .as_ref()
                .is_some_and(|coverage| coverage.contains(coord))
            {
                continue;
            }
            composite_mass +=
                *level.state.fields.cons.den.view().at(coord) * geometry.volume(coord);
        }
    }
    let tracer_count: usize = hierarchy
        .levels
        .iter()
        .map(|level| level.state.tracers.as_ref().unwrap().len())
        .sum();
    let root_tracers = hierarchy.levels[0].state.tracers.as_ref().unwrap();
    let represented_added =
        (tracer_count - 1000) as f64 * root_tracers.weight + root_tracers.injection_remainder;

    assert!((represented_added - (composite_mass - 1.0)).abs() < 1.0e-12);
    let mut ids: Vec<_> = hierarchy
        .levels
        .iter()
        .flat_map(|level| level.state.tracers.as_ref().unwrap().id.iter().copied())
        .collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(
        ids.len(),
        tracer_count,
        "source spawning duplicated tracer ids"
    );
}

#[test]
fn refined_driven_inflow_spawns_only_the_composite_entering_mass() {
    let boundary = SourceConfig::from_json(
        r#"{
            "kind": "dirichlet", "dim": 1, "outputs": [0, 1, 2], "params": [],
            "nodes": [
                {"op": "CONSTANT", "value": 2.0},
                {"op": "CONSTANT", "value": 0.5},
                {"op": "CONSTANT", "value": 1.0}
            ]
        }"#,
    )
    .unwrap();
    let make_kernels = |state: &Sim| {
        let built = build_boundary_dag(&boundary, &NEWTONIAN_SPEC).unwrap();
        Kern::new(GAMMA, CFL, &state.geom.allocated)
            .with_driven_boundary(built, boundary.params.clone())
            .0
    };
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([32])
        .bounds([0.0], [1.0])
        .boundaries(Boundaries::per_axis([[
            BoundaryType::Driven(0),
            BoundaryType::Outflow,
        ]]))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .unwrap()
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0]), Pressure(1.0)))
        .build();
    let kernels = make_kernels(&coarse);
    let mut hierarchy = Hierarchy::with_refinement(
        coarse,
        kernels,
        &[RefinementRegion {
            x_lo: [0.0],
            x_hi: [0.5],
        }],
        ProlongOrder::Ppm,
        make_kernels,
    )
    .unwrap();
    hierarchy.seed_fine_from_coarse().unwrap();
    hierarchy.attach_mass_tracers(1000);

    hierarchy.evolve_steps(2).unwrap();

    let mut composite_mass = 0.0;
    for level in &hierarchy.levels {
        let geometry = level.state.geom.block_geometry(level.state.physics.metric);
        for coord in level.state.geom.interior.iter() {
            if level
                .coverage
                .as_ref()
                .is_some_and(|coverage| coverage.contains(coord))
            {
                continue;
            }
            composite_mass +=
                *level.state.fields.cons.den.view().at(coord) * geometry.volume(coord);
        }
    }
    let tracer_count: usize = hierarchy
        .levels
        .iter()
        .map(|level| level.state.tracers.as_ref().unwrap().len())
        .sum();
    let root_tracers = hierarchy.levels[0].state.tracers.as_ref().unwrap();
    let represented_added =
        (tracer_count - 1000) as f64 * root_tracers.weight + root_tracers.injection_remainder;

    assert!(
        represented_added > 0.0,
        "driven boundary injected no tracers"
    );
    assert!((represented_added - (composite_mass - 1.0)).abs() < 1.0e-12);
}

#[test]
fn refined_neumann_inflow_spawns_only_the_accepted_entering_mass() {
    let make_kernels = |state: &Sim| {
        Kern::new(GAMMA, CFL, &state.geom.allocated)
            .with_gradient_boundary(GradientBc::Neumann(vec![0.0, 20.0, 0.0]))
            .0
    };
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([32])
        .bounds([0.0], [1.0])
        .boundaries(Boundaries([[
            BoundaryType::Neumann(0),
            BoundaryType::Outflow,
        ]]))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .unwrap()
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0]), Pressure(1.0)))
        .build();
    let kernels = make_kernels(&coarse);
    let mut hierarchy = Hierarchy::with_refinement(
        coarse,
        kernels,
        &[RefinementRegion {
            x_lo: [0.0],
            x_hi: [0.5],
        }],
        ProlongOrder::Ppm,
        make_kernels,
    )
    .unwrap();
    hierarchy.seed_fine_from_coarse().unwrap();
    hierarchy.attach_mass_tracers(1000);

    hierarchy.evolve_steps(2).unwrap();

    let mut composite_mass = 0.0;
    for level in &hierarchy.levels {
        let geometry = level.state.geom.block_geometry(level.state.physics.metric);
        for coord in level.state.geom.interior.iter() {
            if level
                .coverage
                .as_ref()
                .is_some_and(|coverage| coverage.contains(coord))
            {
                continue;
            }
            composite_mass +=
                *level.state.fields.cons.den.view().at(coord) * geometry.volume(coord);
        }
    }
    let tracer_count: usize = hierarchy
        .levels
        .iter()
        .map(|level| level.state.tracers.as_ref().unwrap().len())
        .sum();
    let root_tracers = hierarchy.levels[0].state.tracers.as_ref().unwrap();
    let represented_added =
        (tracer_count - 1000) as f64 * root_tracers.weight + root_tracers.injection_remainder;

    assert!(
        tracer_count > 1000,
        "the refined Neumann boundary produced no tracer inflow"
    );
    assert!((represented_added - (composite_mass - 1.0)).abs() < 1.0e-12);
}
