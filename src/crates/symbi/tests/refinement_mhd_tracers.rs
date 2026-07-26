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
use symbi_sim::tracers::cell_container_address;

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
