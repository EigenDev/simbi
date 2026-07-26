// =============================================================================
// decomp_refine_tracers.rs
//
// global tracer ownership on a refined hierarchy decomposed across a root and
// fine-level cut. the decomposed driver must reproduce monolithic owners while
// migrating complete records between tiles.
// =============================================================================

use symbi::prelude::*;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::LocalCopy;
use symbi::sim::refinement::{
    Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed,
    seed_decomposed_hierarchy_tracers,
};

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
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.4]),
            pre: 1.0,
        })
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
    let hierarchy =
        Hier::with_refinement(coarse, kernels(&root(N, 0.0, Boundaries::uniform(BoundaryType::Periodic))), &[region()], ProlongOrder::Plm, kernels)
            .unwrap();
    hierarchy.seed_fine_from_coarse().unwrap();
    hierarchy
}

fn tiles() -> Vec<Hier> {
    let mut result = Vec::new();
    for tile in 0..2 {
        let boundaries = if tile == 0 {
            Boundaries([[
                BoundaryType::Periodic,
                BoundaryType::CoarseFine,
            ]])
        } else {
            Boundaries([[
                BoundaryType::CoarseFine,
                BoundaryType::Periodic,
            ]])
        };
        let coarse = root(N / 2, tile as f64 * 0.5, boundaries);
        let coarse_kernels = kernels(&coarse);
        let mut hierarchy = if tile == 0 {
            let hierarchy = Hier::with_refinement(
                coarse,
                coarse_kernels,
                &[region()],
                ProlongOrder::Plm,
                kernels,
            )
            .unwrap();
            hierarchy.seed_fine_from_coarse().unwrap();
            hierarchy
        } else {
            Hier::single(coarse, coarse_kernels)
        };
        hierarchy.set_tracer_root_layout([N], [tile * (N / 2)]);
        hierarchy.prime();
        result.push(hierarchy);
    }
    result
}

fn owners(hierarchy: &[Hier]) -> std::collections::BTreeMap<u64, symbi_sim::mass_transport::ContainerId> {
    hierarchy
        .iter()
        .flat_map(|tile| &tile.levels)
        .flat_map(|level| {
            let tracers = level.state.tracers.as_ref().unwrap();
            tracers.id.iter().copied().zip(tracers.owner.iter().copied())
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
                    symbi_sim::tracers::cell_container_address(owner)
                        .unwrap()
                        .0 as usize,
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
