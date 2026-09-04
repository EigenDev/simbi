// =============================================================================
// decomp_refine_tracers.rs
//
// global tracer ownership on a refined hierarchy decomposed across a root
// cut. complete records migrate between tiles while refinement-interface
// receipts migrate them between levels.
// =============================================================================

use symbi::prelude::*;
use symbi::regimes::substrate_kernels::GradientBc;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::LocalCopy;
use symbi::sim::refinement::{
    Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed,
    seed_decomposed_hierarchy_tracers,
};
use symbi_hydro::NEWTONIAN_SPEC;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_source_compile::SourceConfig;
use symbi_source_compile::expr_bridge::{build_boundary_dag, build_user_source};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32;
const T_FINAL: f64 = 0.2;
const N_TRACERS: usize = 4096;

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;
type Hier = Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

fn kernels(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &sim.geom.allocated)
}

fn source_kernels(sim: &Sim) -> Kern {
    let source = SourceConfig::from_json(
        r#"{
            "kind": "raw", "dim": 1, "outputs": [0], "params": [],
            "vocabulary":{"reads":[],"params":[]},
            "target": "den",
            "nodes": [ {"op": "CONSTANT", "value": 0.2} ]
        }"#,
    )
    .unwrap();
    kernels(sim).with_runtime_source(
        build_user_source(&source, &NEWTONIAN_SPEC).unwrap(),
        source.params,
    )
}

fn driven_kernels(sim: &Sim) -> Kern {
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
    kernels(sim)
        .with_driven_boundary(
            build_boundary_dag(&boundary, &NEWTONIAN_SPEC).unwrap(),
            boundary.params,
        )
        .0
}

fn gradient_kernels(sim: &Sim) -> Kern {
    kernels(sim)
        .with_gradient_boundary(GradientBc::Neumann(vec![0.0, 20.0, 0.0]))
        .0
}

fn root(cells: usize, origin: f64, boundaries: Boundaries<1>) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([cells])
        .origin([origin])
        .spacing([1.0 / N as f64])
        .boundaries(boundaries)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .unwrap()
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.4]), Pressure(1.0)))
        .build()
}

fn resting_root(cells: usize, origin: f64, boundaries: Boundaries<1>) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([cells])
        .origin([origin])
        .spacing([1.0 / N as f64])
        .boundaries(boundaries)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .unwrap()
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0]), Pressure(1.0)))
        .build()
}

fn region() -> RefinementRegion<1> {
    RefinementRegion {
        x_lo: [0.125],
        x_hi: [0.375],
    }
}

fn global() -> Hier {
    let coarse = root(N, 0.0, Boundaries::uniform(BoundaryType::Periodic));
    let hierarchy = Hier::with_refinement(
        coarse,
        kernels(&root(N, 0.0, Boundaries::uniform(BoundaryType::Periodic))),
        &[region()],
        ProlongOrder::Ppm,
        kernels,
    )
    .unwrap();
    hierarchy.seed_fine_from_coarse().unwrap();
    hierarchy
}

fn source_global() -> Hier {
    let coarse = root(N, 0.0, Boundaries::uniform(BoundaryType::Periodic));
    let coarse_kernels = source_kernels(&coarse);
    let hierarchy = Hier::with_refinement(
        coarse,
        coarse_kernels,
        &[region()],
        ProlongOrder::Ppm,
        source_kernels,
    )
    .unwrap();
    hierarchy.seed_fine_from_coarse().unwrap();
    hierarchy
}

fn driven_global() -> Hier {
    let boundaries = Boundaries([[BoundaryType::Driven(0), BoundaryType::Outflow]]);
    let coarse = resting_root(N, 0.0, boundaries);
    let coarse_kernels = driven_kernels(&coarse);
    let hierarchy = Hier::with_refinement(
        coarse,
        coarse_kernels,
        &[region()],
        ProlongOrder::Ppm,
        driven_kernels,
    )
    .unwrap();
    hierarchy.seed_fine_from_coarse().unwrap();
    hierarchy
}

fn gradient_global() -> Hier {
    let boundaries = Boundaries([[BoundaryType::Neumann(0), BoundaryType::Outflow]]);
    let coarse = resting_root(N, 0.0, boundaries);
    let coarse_kernels = gradient_kernels(&coarse);
    let hierarchy = Hier::with_refinement(
        coarse,
        coarse_kernels,
        &[region()],
        ProlongOrder::Ppm,
        gradient_kernels,
    )
    .unwrap();
    hierarchy.seed_fine_from_coarse().unwrap();
    hierarchy
}

fn tiles() -> Vec<Hier> {
    let mut result = Vec::new();
    for tile in 0..2 {
        let boundaries = if tile == 0 {
            Boundaries([[BoundaryType::Periodic, BoundaryType::CoarseFine]])
        } else {
            Boundaries([[BoundaryType::CoarseFine, BoundaryType::Periodic]])
        };
        let coarse = root(N / 2, tile as f64 * 0.5, boundaries);
        let coarse_kernels = kernels(&coarse);
        let mut hierarchy = if tile == 0 {
            let hierarchy = Hier::with_refinement(
                coarse,
                coarse_kernels,
                &[region()],
                ProlongOrder::Ppm,
                kernels,
            )
            .unwrap();
            hierarchy.seed_fine_from_coarse().unwrap();
            hierarchy
        } else {
            Hier::single(coarse, coarse_kernels)
        };
        hierarchy.set_tracer_root_layout([N], [tile * (N / 2)], [false]);
        hierarchy.prime();
        result.push(hierarchy);
    }
    result
}

fn source_tiles() -> Vec<Hier> {
    let mut result = Vec::new();
    for tile in 0..2 {
        let boundaries = if tile == 0 {
            Boundaries([[BoundaryType::Periodic, BoundaryType::CoarseFine]])
        } else {
            Boundaries([[BoundaryType::CoarseFine, BoundaryType::Periodic]])
        };
        let coarse = root(N / 2, tile as f64 * 0.5, boundaries);
        let coarse_kernels = source_kernels(&coarse);
        let mut hierarchy = if tile == 0 {
            let hierarchy = Hier::with_refinement(
                coarse,
                coarse_kernels,
                &[region()],
                ProlongOrder::Ppm,
                source_kernels,
            )
            .unwrap();
            hierarchy.seed_fine_from_coarse().unwrap();
            hierarchy
        } else {
            Hier::single(coarse, coarse_kernels)
        };
        hierarchy.set_tracer_root_layout([N], [tile * (N / 2)], [false]);
        hierarchy.prime();
        result.push(hierarchy);
    }
    result
}

fn driven_tiles() -> Vec<Hier> {
    let mut result = Vec::new();
    for tile in 0..2 {
        let boundaries = if tile == 0 {
            Boundaries([[BoundaryType::Driven(0), BoundaryType::CoarseFine]])
        } else {
            Boundaries([[BoundaryType::CoarseFine, BoundaryType::Outflow]])
        };
        let coarse = resting_root(N / 2, tile as f64 * 0.5, boundaries);
        let coarse_kernels = driven_kernels(&coarse);
        let mut hierarchy = if tile == 0 {
            let hierarchy = Hier::with_refinement(
                coarse,
                coarse_kernels,
                &[region()],
                ProlongOrder::Ppm,
                driven_kernels,
            )
            .unwrap();
            hierarchy.seed_fine_from_coarse().unwrap();
            hierarchy
        } else {
            Hier::single(coarse, coarse_kernels)
        };
        hierarchy.set_tracer_root_layout([N], [tile * (N / 2)], [false]);
        hierarchy.prime();
        result.push(hierarchy);
    }
    result
}

fn gradient_tiles() -> Vec<Hier> {
    let mut result = Vec::new();
    for tile in 0..2 {
        let boundaries = if tile == 0 {
            Boundaries([[BoundaryType::Neumann(0), BoundaryType::CoarseFine]])
        } else {
            Boundaries([[BoundaryType::CoarseFine, BoundaryType::Outflow]])
        };
        let coarse = resting_root(N / 2, tile as f64 * 0.5, boundaries);
        let coarse_kernels = gradient_kernels(&coarse);
        let mut hierarchy = if tile == 0 {
            let hierarchy = Hier::with_refinement(
                coarse,
                coarse_kernels,
                &[region()],
                ProlongOrder::Ppm,
                gradient_kernels,
            )
            .unwrap();
            hierarchy.seed_fine_from_coarse().unwrap();
            hierarchy
        } else {
            Hier::single(coarse, coarse_kernels)
        };
        hierarchy.set_tracer_root_layout([N], [tile * (N / 2)], [false]);
        hierarchy.prime();
        result.push(hierarchy);
    }
    result
}

fn composite_mass(hierarchy: &[Hier]) -> f64 {
    hierarchy
        .iter()
        .flat_map(|tile| &tile.levels)
        .map(|level| {
            let geometry = level.state.geom.block_geometry(level.state.physics.metric);
            level
                .state
                .geom
                .interior
                .iter()
                .filter(|coord| {
                    !level
                        .coverage
                        .as_ref()
                        .is_some_and(|coverage| coverage.contains(*coord))
                })
                .map(|coord| *level.state.fields.cons.den.view().at(coord) * geometry.volume(coord))
                .sum::<f64>()
        })
        .sum()
}

fn owners(
    hierarchy: &[Hier],
) -> std::collections::BTreeMap<u64, symbi_sim::mass_transport::ContainerId> {
    hierarchy
        .iter()
        .flat_map(|tile| &tile.levels)
        .flat_map(|level| {
            let tracers = level.state.tracers.as_ref().unwrap();
            tracers
                .id
                .iter()
                .copied()
                .zip(tracers.owner.iter().copied())
        })
        .collect()
}

#[test]
fn decomposed_refined_tracers_migrate_across_level_and_tile_cuts() {
    let global_seed = global();
    let mut decomposed = tiles();
    seed_decomposed_hierarchy_tracers(&global_seed, &mut decomposed, N_TRACERS);
    let initial_tile: std::collections::BTreeMap<_, _> = decomposed
        .iter()
        .enumerate()
        .flat_map(|(tile_index, tile)| {
            tile.levels.iter().flat_map(move |level| {
                level
                    .state
                    .tracers
                    .as_ref()
                    .unwrap()
                    .id
                    .iter()
                    .copied()
                    .map(move |id| (id, tile_index))
            })
        })
        .collect();
    let initial_owner = owners(&decomposed);
    evolve_hierarchy_decomposed(
        &mut decomposed,
        [2],
        &[0, 0],
        &LocalCopy,
        Timestepping::Rk2,
        0.0,
        T_FINAL,
        u64::MAX,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
    let actual = owners(&decomposed);

    assert_eq!(actual.len(), N_TRACERS);
    assert_eq!(
        actual.keys().copied().collect::<Vec<_>>(),
        (0..N_TRACERS as u64).collect::<Vec<_>>()
    );
    let mut crossed_tile = 0usize;
    let mut crossed_level = 0usize;
    for (tile_index, tile) in decomposed.iter().enumerate() {
        for (level_index, level) in tile.levels.iter().enumerate() {
            let tracers = level.state.tracers.as_ref().unwrap();
            for (&id, &owner) in tracers.id.iter().zip(&tracers.owner) {
                crossed_tile += usize::from(initial_tile[&id] != tile_index);
                crossed_level += usize::from(
                    symbi_sim::tracers::cell_container_address(initial_owner[&id])
                        .unwrap()
                        .0 as usize
                        != level_index,
                );
                assert_eq!(
                    symbi_sim::tracers::cell_container_address(owner).unwrap().0 as usize,
                    level_index,
                    "tracer {id} is stored on the wrong refinement level"
                );
            }
        }
    }
    assert!(
        crossed_tile > 0,
        "no complete tracer record migrated across a decomposition cut"
    );
    assert!(
        crossed_level > 0,
        "no tracer crossed a refinement interface"
    );
}

#[test]
fn decomposed_refined_source_spawning_matches_composite_added_mass() {
    let global_seed = source_global();
    let mut decomposed = source_tiles();
    seed_decomposed_hierarchy_tracers(&global_seed, &mut decomposed, N_TRACERS);
    let mass_before = composite_mass(&decomposed);

    evolve_hierarchy_decomposed(
        &mut decomposed,
        [2],
        &[0, 0],
        &LocalCopy,
        Timestepping::Rk2,
        0.0,
        0.04,
        u64::MAX,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );

    let all_ids: Vec<_> = decomposed
        .iter()
        .flat_map(|tile| &tile.levels)
        .flat_map(|level| level.state.tracers.as_ref().unwrap().id.iter().copied())
        .collect();
    let unique: std::collections::BTreeSet<_> = all_ids.iter().copied().collect();
    let tracers = decomposed[0].levels[0].state.tracers.as_ref().unwrap();
    let represented_added =
        (all_ids.len() - N_TRACERS) as f64 * tracers.weight + tracers.injection_remainder;
    let fluid_added = composite_mass(&decomposed) - mass_before;
    let expected_added = 0.2 * 0.04;

    assert_eq!(
        unique.len(),
        all_ids.len(),
        "spawned tracer IDs are not global"
    );
    assert!(
        all_ids.len() > N_TRACERS,
        "the decomposed source spawned no tracers"
    );
    assert!(
        (represented_added - expected_added).abs() < 1.0e-12,
        "represented source mass {represented_added:e} != analytic source integral \
         {expected_added:e}"
    );
    // the decomposed periodic halo reduction closes the fluid budget to about
    // 2.5e-11 on this grid; tracer apportionment above remains exact.
    assert!(
        (fluid_added - expected_added).abs() < 1.0e-10,
        "composite fluid addition {fluid_added:e} != analytic source integral \
         {expected_added:e}"
    );
    for tile in &decomposed {
        for level in &tile.levels {
            let state = level.state.tracers.as_ref().unwrap();
            assert_eq!(state.next_id, tracers.next_id);
            assert_eq!(
                state.injection_remainder.to_bits(),
                tracers.injection_remainder.to_bits()
            );
        }
    }
}

#[test]
fn decomposed_refined_driven_inflow_uses_one_global_spawn_stream() {
    let global_seed = driven_global();
    let mut decomposed = driven_tiles();
    seed_decomposed_hierarchy_tracers(&global_seed, &mut decomposed, N_TRACERS);
    let mass_before = composite_mass(&decomposed);

    evolve_hierarchy_decomposed(
        &mut decomposed,
        [2],
        &[0, 0],
        &LocalCopy,
        Timestepping::Rk2,
        0.0,
        0.04,
        u64::MAX,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );

    let all_ids: Vec<_> = decomposed
        .iter()
        .flat_map(|tile| &tile.levels)
        .flat_map(|level| level.state.tracers.as_ref().unwrap().id.iter().copied())
        .collect();
    let unique: std::collections::BTreeSet<_> = all_ids.iter().copied().collect();
    let tracers = decomposed[0].levels[0].state.tracers.as_ref().unwrap();
    let represented_added =
        (all_ids.len() - N_TRACERS) as f64 * tracers.weight + tracers.injection_remainder;
    let fluid_added = composite_mass(&decomposed) - mass_before;

    assert_eq!(
        unique.len(),
        all_ids.len(),
        "inflow tracer IDs are not global"
    );
    assert!(
        all_ids.len() > N_TRACERS,
        "the decomposed driven boundary spawned no tracers"
    );
    assert!(
        (represented_added - fluid_added).abs() < 1.0e-10,
        "represented inflow mass {represented_added:e} != composite fluid addition \
         {fluid_added:e}"
    );
}

#[test]
fn decomposed_refined_gradient_inflow_uses_one_global_spawn_stream() {
    let global_seed = gradient_global();
    let mut decomposed = gradient_tiles();
    seed_decomposed_hierarchy_tracers(&global_seed, &mut decomposed, N_TRACERS);
    let mass_before = composite_mass(&decomposed);

    evolve_hierarchy_decomposed(
        &mut decomposed,
        [2],
        &[0, 0],
        &LocalCopy,
        Timestepping::Rk2,
        0.0,
        0.04,
        u64::MAX,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );

    let all_ids: Vec<_> = decomposed
        .iter()
        .flat_map(|tile| &tile.levels)
        .flat_map(|level| level.state.tracers.as_ref().unwrap().id.iter().copied())
        .collect();
    let unique: std::collections::BTreeSet<_> = all_ids.iter().copied().collect();
    let tracers = decomposed[0].levels[0].state.tracers.as_ref().unwrap();
    let represented_added =
        (all_ids.len() - N_TRACERS) as f64 * tracers.weight + tracers.injection_remainder;
    let fluid_added = composite_mass(&decomposed) - mass_before;

    assert_eq!(
        unique.len(),
        all_ids.len(),
        "gradient-inflow tracer IDs are not global"
    );
    assert!(
        all_ids.len() > N_TRACERS,
        "the decomposed gradient boundary spawned no tracers"
    );
    assert!(
        (represented_added - fluid_added).abs() < 1.0e-10,
        "represented gradient inflow mass {represented_added:e} != composite fluid addition \
         {fluid_added:e}"
    );
}
