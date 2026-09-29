// =============================================================================
// retry_exclusion_propagation.rs
//
// the retry exclusion is the run's configured policy, and every tile and level
// of a run carries the same one:
// - a decomposed run whose tiles carry different policies refuses to run,
//   naming both;
// - a hierarchy whose levels carry different policies refuses to run;
// - a fine level takes its parent's policy when it is built;
// - a checkpoint restore leaves the configured policy on every level, whatever
//   policy the written run carried.
//
// run: cargo test -p symbi --test retry_exclusion_propagation
// =============================================================================

use std::ops::ControlFlow;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, Schedule, Topology, evolve_scheduled, unflatten};
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi::sim::substrate_seam::RetryExclusion;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::Newtonian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_io::Metadata;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim2 = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern2 = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Sim3 = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern3 = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;

const GAMMA: f64 = 5.0 / 3.0;
const SPHERE: RetryExclusion = RetryExclusion::LegacyEuclideanSphere { radius: 0.25 };

fn panic_message(f: impl FnOnce() + std::panic::UnwindSafe) -> Option<String> {
    std::panic::catch_unwind(f).err().map(|p| {
        p.downcast_ref::<String>()
            .cloned()
            .or_else(|| p.downcast_ref::<&str>().map(|m| m.to_string()))
            .unwrap_or_default()
    })
}

#[test]
fn tiles_carrying_different_policies_refuse_to_run() {
    let n = 16;
    let h = 1.0 / n as f64;
    let partition = Partition::explicit([n, n], [vec![n / 2], Vec::new()]).expect("cut");
    let counts = partition.counts();
    let message = panic_message(|| {
        let mut sims: Vec<Sim2> = (0..partition.n_tiles())
            .map(|f| {
                let ext = partition.tile_extents(unflatten(f, counts));
                let mut s = Sim2::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
                    .cells([ext[0].1, ext[1].1])
                    .spacing([h, h])
                    .origin([ext[0].0 as f64 * h, ext[1].0 as f64 * h])
                    .boundaries(Boundaries([
                        [BoundaryType::CoarseFine; 2],
                        [BoundaryType::Periodic; 2],
                    ]))
                    .cfl(0.3)
                    .timestepping(Timestepping::Rk2)
                    .allocate()
                    .expect("sim")
                    .set_initial(|_| {
                        Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0))
                    })
                    .build();
                s.retry_exclusion = if f == 0 { SPHERE } else { RetryExclusion::None };
                s
            })
            .collect();
        let ks: Vec<Kern2> = sims
            .iter()
            .map(|s| Kern2::new(GAMMA, 0.3, &s.geom.allocated))
            .collect();
        let mut stores: Vec<&mut FieldStore<2, 2, HostMemory, f64>> =
            sims.iter_mut().map(|s| &mut **s).collect();
        let kernel_refs: Vec<&Kern2> = ks.iter().collect();
        let schedule = Schedule::derive(counts, stores[0].geom.ng, &Topology::wrapping([true; 2]));
        evolve_scheduled(
            &mut stores,
            &kernel_refs,
            &schedule,
            &[0, 0],
            Timestepping::Rk2,
            0.0,
            1e-3,
            1,
            &LocalCopy,
            |_, _, _| ControlFlow::Continue(()),
        );
    });
    assert!(
        message
            .as_deref()
            .is_some_and(|m| m.contains("carry different retry exclusions")),
        "the split-policy run was not refused by name: {message:?}"
    );
}

/// a two-level hierarchy whose root carries `policy`
fn refined(
    policy: RetryExclusion,
) -> Hierarchy<Newtonian, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern3> {
    let mut coarse = Sim3::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([8; 3])
        .origin([0.0; 3])
        .spacing([1.0 / 8.0; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .unwrap()
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0; 3]), Pressure(1.0)))
        .build();
    coarse.retry_exclusion = policy;
    let ck = Kern3::new(GAMMA, 0.3, &coarse.geom.allocated);
    let region = RefinementRegion {
        x_lo: [0.25; 3],
        x_hi: [0.75; 3],
    };
    Hierarchy::with_refinement(coarse, ck, &[region], ProlongOrder::Ppm, |s| {
        Kern3::new(GAMMA, 0.3, &s.geom.allocated)
    })
    .expect("refined hierarchy")
}

#[test]
fn a_fine_level_takes_its_parent_policy_and_disagreeing_levels_refuse_to_run() {
    let hier = refined(SPHERE);
    assert_eq!(hier.levels.len(), 2, "the refinement built no fine level");
    assert_eq!(
        hier.levels[1].state.retry_exclusion, SPHERE,
        "the fine level did not take its parent's policy"
    );
    let message = panic_message(|| {
        let mut hier = refined(SPHERE);
        hier.levels[1].state.retry_exclusion = RetryExclusion::None;
        let _ = hier.evolve_with_callback(1e-3, 1, |_| ControlFlow::Continue(()));
    });
    assert!(
        message
            .as_deref()
            .is_some_and(|m| m.contains("carry different retry exclusions")),
        "the split-policy hierarchy was not refused by name: {message:?}"
    );
}

#[test]
fn a_checkpoint_restore_keeps_the_configured_policy() {
    let path = std::env::temp_dir().join(format!(
        "symbi_retry_exclusion_restore_{}.h5",
        std::process::id()
    ));
    let written = refined(SPHERE);
    let levels: Vec<&Sim3> = written.levels.iter().map(|l| &l.state).collect();
    symbi::sim::checkpoint::write_hierarchy_checkpoint(
        &levels,
        path.to_str().unwrap(),
        &Metadata::new(),
    )
    .expect("written");
    for configured in [SPHERE, RetryExclusion::None] {
        let mut resumed = refined(configured);
        resumed
            .restore_from_checkpoint(path.to_str().unwrap())
            .expect("restored");
        for (ii, level) in resumed.levels.iter().enumerate() {
            assert_eq!(
                level.state.retry_exclusion, configured,
                "level {ii} left the restore with another policy"
            );
        }
    }
    let _ = std::fs::remove_file(&path);
}
